import glob
import os
from pathlib import Path

import pandas as pd
from enum import Enum
from prometheus_api_client import PrometheusConnect, MetricsList, Metric, MetricSnapshotDataFrame, MetricRangeDataFrame
from prometheus_api_client.utils import parse_datetime
from datetime import timedelta, datetime
import logging
from prometheus_pandas import query


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
                                    start_time: str = "2020") -> pd.DataFrame:
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
    def create_prometheus_df(start_time_str: str,
                             timedelta_hours: int,
                             num_timedelta_hours: int,
                             start_metric_name: str,
                             save_dir: str = None,
                             step: str = "1s",
                             prometheus_url: str = "http://127.0.0.1:9090/") -> pd.DataFrame:
        """

        Parameters
        ----------
        start_time_str : str
            start time in '%Y-%m-%dT%H:%M:%SZ' format
        timedelta_hours : int
            scrape size in hours for each query. e.g. 11000 is the maximum size of
            timeseries query, if step is 1s then it would be approximately 3 hours
        num_timedelta_hours : int
            number of addining timedelta_hours to the start time. e.g.
            if timedelta_hours is 3 hours and num_timedelta_hours is 16 then it would scrape for 48 hours/2day
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

        date_format_str = '%Y-%m-%dT%H:%M:%SZ'

        # prometheus unique save dir
        prom_save_dir = start_time_str
        # final save dir
        save_dir = os.path.join(save_dir, prom_save_dir)
        Path("save_dir").mkdir(parents=True, exist_ok=True)

        # if not os.path.exists(save_dir):
        #     os.mkdir(save_dir)

        for metric_name in filter_metrics_name:
            start_time = datetime.strptime(start_time_str, date_format_str)
            for i in range(num_timedelta_hours):
                end_time = start_time + timedelta(hours=timedelta_hours)
                start_time_str_tmp = start_time.strftime(date_format_str)
                end_time_str = end_time.strftime(date_format_str)
                p = query.Prometheus(prometheus_url)
                res = p.query_range(metric_name, start_time_str_tmp, end_time_str, step)
                save_path = os.path.join(save_dir, (metric_name + ".csv"))
                if i == 0:
                    res.to_csv(save_path)
                else:
                    res.to_csv(save_path, mode='a', header=False)
                start_time = end_time

    @staticmethod
    def merge_rabbitmq_prometheus_dfs(dir_path: str, drop_constant_cols: bool = True) -> pd.DataFrame:
        """
        Merge rabbitmq prometheus dataframes

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

        rabbitmq_modules = ["connections", "shovel",
                            "federation", "exchange",
                            "node", "queue", "memory"]
        # exchanges = [""]
        for csv in csv_files:
            metric_name = os.path.basename(csv).replace(".csv", "")
            metric_df = pd.read_csv(csv, index_col=0)
            # These two dataframe have values for each rabbitmq modules
            if metric_name == "rabbitmq_module_scrape_duration_seconds" or \
                    metric_name == "rabbitmq_module_up":
                for rabbitmq_module in rabbitmq_modules:
                    for column in metric_df.columns:
                        if "module=\"" + rabbitmq_module + "\"" in column:
                            module_df = metric_df[[column]].copy()
                            module_df.rename(columns={column: metric_name + "_" + rabbitmq_module}, inplace=True)
                            df_merge_list.append(module_df)
            # These two dataframe have values for each rabbitmq exchanges
            # elif metric_name == "rabbitmq_exchange_messages_published_in_total" or \
            #         metric_name == "rabbitmq_exchange_messages_published_out_total":
            #     if 'exchange' in metric_df.columns:
            #         for exchange in exchanges:
            #             # fill nan with empty string
            #             metric_df['exchange'] = metric_df['exchange'].fillna('')
            #             exchange_df = metric_df.loc[metric_df['exchange'] == exchange].copy()
            #             exchange_df.drop(exchange_df.columns.difference(['value']), 1, inplace=True)
            #             exchange_df.rename(columns={'value': metric_name + "_" + exchange}, inplace=True)
            #             df_merge_list.append(exchange_df)
            #     else:
            #         logging.error("could not find module column in: " + metric_name)
            #         continue
            elif len(metric_df.columns) == 1:
                metric_df.rename(columns={metric_df.columns[0]: metric_name}, inplace=True)
                df_merge_list.append(metric_df)
            else:
                logging.error("Metric dataframe could not be parsed and merged for metric name: " + metric_name)
        try:

            result = pd.concat(df_merge_list, axis=1)
            result.index.name = "Time"
            if drop_constant_cols:
                result = result.loc[:, (result != result.iloc[0]).any()]
            result.to_csv(os.path.join(dir_path, "PROMETHEUS_merged.csv"))
        except Exception as e:
            logging.error("could not concat dataframes: " + str(e))


if __name__ == '__main__':
    # DatasetCreator.create_addition_1_method_df(2 * 24 * 60 * 60, 'S', "../data")
    # logging.basicConfig(filename='dataset_creator.log', level=logging.DEBUG)
    # # start_time = datetime(year=2021, month=12, day=21, hour=3, minute=00, second=00)
    # # end_time = datetime(year=2021, month=12, day=21, hour=4, minute=00, second=00)
    # start_time = datetime(year=2021, month=1, day=6, hour=23, minute=30, second=00)
    # end_time = datetime(year=2022, month=1, day=7, hour=10, minute=30, second=00)
    # chunk_size = timedelta(seconds=1)
    #
    DatasetCreator.create_prometheus_df(start_time_str="2022-01-24 21:10:00",
                                        timedelta_hours=1,
                                        num_timedelta_hours=12,
                                        start_metric_name="rabbitmq",
                                        save_dir="data/prometheus",
                                        step="1s")
    # DatasetCreator.create_prometheus_df(start_time=start_time, end_time=end_time,
    #
    #                                     start_metric_name="rabbitmq",
    #                                     save_dir="../data/prometheus")

    # DatasetCreator.merge_rabbitmq_prometheus_dfs("data/prometheus/2022-01-06T23:30:00Z")
