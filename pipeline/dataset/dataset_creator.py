import glob
import logging
import os
from datetime import timedelta, datetime
from pathlib import Path

import dateutil.rrule as rrule
import pandas as pd
from prometheus_api_client import PrometheusConnect
from prometheus_pandas import query


class DatasetCreator:
    """
    Create a pandas dataframe by fetching from prometheus

    """

    def __init__(self):
        self.date_format_str = '%Y-%m-%dT%H:%M:%SZ'
        self.merged_csv = None

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
            rule = rrule.rrule(rrule.HOURLY, interval=interval, byminute=0, bysecond=0, dtstart=start)
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

    def create_prometheus_df(self, start_time_str: str,
                             end_time_str: str,
                             start_metric_name: str,
                             save_dir: str = None,
                             step: str = "1s",
                             prometheus_url: str = "http://127.0.0.1:9090/") -> pd.DataFrame:
        """

        Parameters
        ----------

        start_time_str : str
            start time in '%Y-%m-%dT%H:%M:%SZ' format
        end_time_str : str
            end time in '%Y-%m-%dT%H:%M:%SZ' format
        start_metric_name : str
            start string of metrics name to filter from all the metrics,
            e.g. if filter_name is rabbitmq then it will select all metrics
            that start with rabbitmq
        save_dir : str
            Directory path for saving the dataset
        step : str
            step size for querying prometheus. e.g. 1s
        prometheus_url : str
            host url of prometheus

        Returns
        -------
        Pandas DataFrame
            created dataframe
        """
        prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)
        all_metrics_name = prom.all_metrics()
        filter_metrics_name = [s for s in all_metrics_name if s.startswith(start_metric_name)]

        # prometheus unique save dir
        prom_save_dir = start_time_str
        # final save dir
        save_dir = os.path.join(save_dir, prom_save_dir)
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        chunked_datetime = self.chunk_datetime(start_time_str, end_time_str)

        for metric_name in filter_metrics_name:
            logging.debug(metric_name)
            for i, cd in enumerate(chunked_datetime):
                p = query.Prometheus(prometheus_url)
                res = p.query_range(metric_name, cd[0], cd[1], step)
                save_path = os.path.join(save_dir, (metric_name + ".csv"))
                if i == 0:
                    res.to_csv(save_path)
                else:
                    res.to_csv(save_path, mode='a', header=False)

    def merge_prometheus_dfs(self, dir_path: str, drop_constant_cols: bool = True) -> pd.DataFrame:

        """
        Merge  prometheus dataframes

        Parameters
        ----------
        drop_constant_cols : bool
            If to drop constant/same value columns in the dataframes
        dir_path : str
            directory path where prometheus csv files locate

        Returns
        -------
        pd.DataFrame
            merged dataframe
        """
        csv_files = glob.glob(dir_path + "/*.csv")

        df_merge_list = []

        for csv in csv_files:
            metric_name = os.path.basename(csv).replace(".csv", "")
            metric_df = pd.read_csv(csv, index_col=0)
            if len(metric_df.columns) == 1:
                metric_df.rename(columns={metric_df.columns[0]: metric_name}, inplace=True)
                df_merge_list.append(metric_df)
            else:
                logging.error("Metric dataframe could not be parsed and merged for metric name: " + metric_name)

        for i in range(len(df_merge_list) - 1):
            if not (df_merge_list[i].index.equals(df_merge_list[i + 1].index)):
                logging.error("These two dataframes have different index:"
                              + df_merge_list[i].columns[0] + " , " + df_merge_list[i + 1].columns[0])

        try:
            result = pd.concat(df_merge_list, axis=1)
            result.index.name = "Time"
            if drop_constant_cols:
                result = result.loc[:, (result != result.iloc[0]).any()]
            result.to_csv(os.path.join(dir_path, "merged.csv"))
            self.merged_csv = result
        except Exception as e:
            logging.error("could not concat dataframes: " + str(e))


if __name__ == '__main__':
    logging.basicConfig(filename='dataset_creator.log', level=logging.DEBUG)
    dc = DatasetCreator()
    dc.create_prometheus_df(start_time_str="2022-02-18T11:30:00Z",
                            end_time_str="2022-02-25T11:30:00Z",
                            start_metric_name="node_",
                            save_dir="data/prometheus",
                            step="1s")
    dc.merge_rabbitmq_prometheus_dfs("data/prometheus/2022-02-18T11:30:00Z")
