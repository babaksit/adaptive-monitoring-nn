import json
import logging
import os
import time
from pathlib import Path
import pandas as pd

from pipeline.prometheus.handler import PrometheusHandler


class DatasetCreator:
    """
    Create a pandas dataframe by fetching from prometheus

    """

    def __init__(self):
        self.date_format_str = '%Y-%m-%dT%H:%M:%SZ'
        self.df = None
        self.config = None
        self.prometheus_handler = None

    def shift_df_index(self, date_time: str = None):
        """
        Shift dataframe index to the beginning of the day or given date_time
        Returns
        -------

        """

        if date_time:
            day_start = pd.Timestamp(self.df.index[0].strftime(date_time))
        else:
            day_start = pd.Timestamp(self.df.index[0].strftime('%Y-%m-%d 00:00:00'))
        self.df.index = self.df.index.shift(periods=1, freq=(day_start - self.df.index[0]))

    def create_prometheus_queries_df(self, config_path: str, save: bool = True,
                                     shift_index: bool = True) -> pd.DataFrame:

        with open(config_path) as f:
            self.config = json.load(f)

        start_time_str = self.config["start_time"]
        end_time_str = self.config["end_time"]
        save_dir = self.config["save_dir"]
        step = self.config["step"]
        prometheus_url = self.config["prometheus_url"]
        queries = [query['query'] for query in self.config["queries"]]
        columns = [query['column_name'] for query in self.config["queries"]]

        if len(columns) != len(queries):
            raise ValueError("column names should have same size as queries, check config!")
            return None

        self.prometheus_handler = PrometheusHandler(prometheus_url)

        # prometheus unique save dir
        save_time = str(time.ctime()) + ".csv"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        # final save dir
        save_path = os.path.join(save_dir, save_time)

        self.df = self.prometheus_handler.fetch_queries(start_time_str, end_time_str,
                                                        queries, columns, step)
        #
        if shift_index:
            self.shift_df_index()
        if save:
            self.df.to_csv(save_path)
        return self.df


if __name__ == '__main__':
    logging.basicConfig(filename='dataset_creator.log', level=logging.DEBUG)
    dc = DatasetCreator()
    dc.create_prometheus_queries_df("/home/bsi/adaptive-monitoring-nn/pipeline/configs/dataset.json")
