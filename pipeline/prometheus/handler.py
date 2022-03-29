import logging
from datetime import datetime, timedelta

import requests
from dateutil import rrule
from kubernetes import client, config
from prometheus_pandas import query
import pandas as pd


class PrometheusHandler:

    def __init__(self, prometheus_url):

        self.prometheus_url = prometheus_url
        self.prometheus = query.Prometheus(self.prometheus_url)
        self.date_format_str = "%Y-%m-%dT%H:%M:%SZ"

    def chunk_datetime(self, start_time_str: str, end_time_str: str, interval: int = 3):
        """
        Chunk given period into intervals

        Parameters
        ----------
        start_time_str : str
            start time in '%Y-%m-%dT%H:%M:%SZ' format
        end_time_str : str
            end time in '%Y-%m-%dT%H:%M:%SZ' format
        interval : int
            interval in hours

        Returns
        -------

        """

        def hours_aligned(start, end, interval, inc=True):
            if inc: yield start
            rule = rrule.rrule(rrule.HOURLY, interval=interval, byminute=0,
                               bysecond=0, dtstart=start)
            for x in rule.between(start, end, inc=inc):
                yield x
            if inc: yield end

        start_time = datetime.strptime(start_time_str, self.date_format_str)
        end_time = datetime.strptime(end_time_str, self.date_format_str)
        time_list = list(hours_aligned(start_time, end_time, interval))

        result = []
        for i in range(len(time_list) - 1):
            if i == 0:
                result.append([time_list[i], time_list[i + 1]])
                continue
            result.append([time_list[i] + timedelta(seconds=1), time_list[i + 1]])

        return result

    def fetch_queries(self, start_time_str: str,
                      end_time_str: str,
                      queries: list,
                      columns: list = [],
                      single_column_query: bool = True,
                      step: str = "1s"
                      ):
        result= None
        chunked_datetime = self.chunk_datetime(start_time_str, end_time_str)
        df_list = []
        for c, query_str in enumerate(queries):
            logging.info(query_str)
            chunked_df_list = []
            for i, cd in enumerate(chunked_datetime):
                try:
                    df = self.prometheus.query_range(query_str, cd[0], cd[1], step)
                except RuntimeError as re:
                    logging.error(str(re))
                    continue
                if single_column_query and len(df.columns) != 1:
                    logging.error("Query: " + query_str + " has more than one value")
                    continue

                if single_column_query:
                    if columns:
                        df.rename(columns={df.columns[0]: columns[c]}, inplace=True)
                    else:
                        df.rename(columns={df.columns[0]: query}, inplace=True)
                chunked_df_list.append(df)

            try:
                df_list.append(pd.concat(chunked_df_list, axis=0))
            except Exception as e:
                logging.error("could not concat dataframes: " + str(e))
        try:
            result = pd.concat(df_list, axis=1)
            result.index.name = "Time"
        except Exception as e:
            logging.error("could not concat dataframes: " + str(e))

        return result

    def change_fetching_state(self,
                              enable: bool,
                              group="monitoring.coreos.com",
                              version="v1",
                              namespace="default",
                              plural="servicemonitors",
                              name="prometheus-prometheus-node-exporter",
                              ) -> bool:
        logging.debug("Changing fetching state to:" + str(enable))
        config.load_kube_config()
        api_custom = client.CustomObjectsApi()
        try:
            conf = api_custom.get_namespaced_custom_object(
                group=group,
                version=version,
                namespace=namespace,
                plural=plural,
                name=name,
            )
        except Exception as e:
            logging.error("Could not get the monitoring service: " + str(e))
            return False

        if enable:
            conf['metadata']['labels']['release'] = "prometheus"
        else:
            # setting to a random string, so the prometheus does not scrape it
            conf['metadata']['labels']['release'] = "disable"

        try:
            api_custom.patch_namespaced_custom_object(
                group=group,
                version=version,
                namespace=namespace,
                plural=plural,
                name=name,
                body=conf)
        except Exception as e:
            logging.error("Could not patch the monitoring service: " + str(e))
            return False

        r = requests.post(self.prometheus_url + '/-/reload')
        if r.status_code != 200:
            logging.error("Reloading the prometheus config was not successful: " + str(r.status_code))
            return False

        logging.debug("Changed fetching state to:" + str(enable))

        return True
