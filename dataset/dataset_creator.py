import os

import pandas as pd
from enum import Enum
from prometheus_api_client import PrometheusConnect, MetricsList, Metric, MetricSnapshotDataFrame, MetricRangeDataFrame
from prometheus_api_client.utils import parse_datetime
from datetime import timedelta, datetime
import logging


class CreationMethod(Enum):
    """
    Enum for different methods of creating dataset

    """
    # __init__ = 'value __doc__'
    ADDITION_1 = 1, 'Each new data value is last data value plus 1'
    PROMETHEUS = 2, "Data from prometheus queries"


class DatasetCreator:
    """
    Create a pandas dataframe based the given method.

    """

    @staticmethod
    def create_addition_1_method_df(length: int,
                                    frequency: str,
                                    save_dir: str = None,
                                    start_time: str = "2000") -> pd.DataFrame:
        """
        Create a pandas dataframe using addition_1 method

        Parameters
        ----------

        length : number of rows in the dataset
        frequency : Frequency of time series e.g. 'S' which is every second.
                    See https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
                    for a list of frequency aliases
        save_dir : Directory path for saving the dataset
        start_time: start time of time series e.g. 2021-01-01 23:23:23 or datetime object

        Returns
        -------
        created pandas DataFrame
        """
        idx = pd.date_range(start_time, periods=length, freq=frequency)
        ts = pd.Series(range(len(idx)), index=idx)
        df = pd.DataFrame({'Time': ts.index, 'Value': ts.values})
        df.set_index('Time', inplace=True)
        if save_dir:
            save_file_name = "ADDITION_1" + "_" \
                             + start_time + "_" + str(length) \
                             + "_" + frequency + ".csv"
            df.to_csv(os.path.join(save_dir, save_file_name))
        return df

    @staticmethod
    def create_prometheus_df(start_time: datetime, end_time: datetime,
                             chunk_size: int, start_metric_name: str,
                             save_dir: str = None,
                             rabbitmq_modules: list = None,
                             prometheus_url: str = "http://127.0.0.1:9090/") -> pd.DataFrame:
        """
        Create a pandas dataframe by querying prometheus metrics

        Parameters
        ----------

        start_time : start time of querying metrics
        end_time: end time of querying metrics
        chunk_size: interval time to query metrics
        save_dir : Directory path for saving the dataset
        prometheus_url : host url of prometheus
        start_metric_name : start string of metrics name to filter from all the metrics,
                            e.g. if filter_name is rabbitmq then it will select all metrics
                            that start with rabbitmq
        rabbitmq_modules : If metrics are from rabbitmq, then list of exporter modules should be provided

        Returns
        -------
        created pandas DataFrame
        """
        prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)
        all_metrics_name = prom.all_metrics()
        filter_metrics_name = [s for s in all_metrics_name if s.startswith(start_metric_name)]
        df_merge_list = []


        # prometheus unique save dir
        prom_save_dir =  str(start_time) + " to " + str(end_time)
        # final save dir
        save_dir = os.path.join(save_dir, prom_save_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for metric_name in filter_metrics_name:
            logging.debug("creating dataframe for: " + metric_name)
            metric_data = prom.get_metric_range_data(
                metric_name,
                start_time=start_time,
                end_time=end_time,
                chunk_size=chunk_size,
            )
            try:
                metric_df = MetricRangeDataFrame(metric_data)
            except KeyError as e:
                logging.error("could not create MetricRangeDataFrame for metric: " +
                              metric_name + " " + str(e))
                print(metric_data)
                continue

            save_file_name = "PROMETHEUS" + "_" \
                             + metric_name + ".csv"
            metric_df.to_csv(os.path.join(save_dir, save_file_name))

            # These two dataframe has values for each rabbitmq modules
            if metric_name == "rabbitmq_module_scrape_duration_seconds" or \
                    metric_name == "rabbitmq_module_up":
                if 'module' in metric_df.columns:
                    for rabbitmq_module in rabbitmq_modules:
                        module_df = metric_df.loc[metric_df['module'] == rabbitmq_module].copy()
                        module_df.drop(module_df.columns.difference(['value']), 1, inplace=True)
                        module_df.rename(columns={'value': metric_name + "_" + rabbitmq_module}, inplace=True)
                        df_merge_list.append(module_df)
                else:
                    logging.error("could not find module column in: " + metric_name)
                    continue
            else:
                metric_df.drop(metric_df.columns.difference(['value']), 1, inplace=True)
                metric_df.rename(columns={'value': metric_name}, inplace=True)
                df_merge_list.append(metric_df)

        try:
            result = pd.concat(df_merge_list, axis=1)
            result.to_csv(os.path.join(save_dir, "PROMETHEUS_merged.csv"))
        except Exception as e:
            logging.error("could not concat dataframes: " + str(e))


if __name__ == '__main__':
    # DatasetCreator.create_addition_1_method_df(12 * 60 * 60, 'S', "../data")
    logging.basicConfig(filename='dataset_creator.log', encoding='utf-8', level=logging.DEBUG)
    start_time = datetime(year=2021, month=12, day=21, hour=3, minute=00, second=00)
    end_time = datetime(year=2021, month=12, day=21, hour=4, minute=00, second=00)
    # start_time = datetime(year=2021, month=12, day=20, hour=22, minute=00, second=00)
    # end_time = datetime(year=2021, month=12, day=21, hour=10, minute=00, second=00)
    chunk_size = timedelta(seconds=1)

    DatasetCreator.create_prometheus_df(start_time=start_time, end_time=end_time,
                                        chunk_size=chunk_size,
                                        start_metric_name="rabbitmq",
                                        save_dir="../data",
                                        rabbitmq_modules=["connections", "shovel",
                                                          "federation", "exchange",
                                                          "node", "queue", "memory"])
