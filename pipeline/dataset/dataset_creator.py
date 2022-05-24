import json
import logging
import os
import time
from pathlib import Path
import pandas as pd
import re
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

    @staticmethod
    def drop_labels(df: pd.DataFrame, drop_labels: tuple):
        if not drop_labels:
            return
        columns = df.columns
        new_columns = []
        for col in columns:
            labels = col[col.find('{') + len('{'):col.rfind('}')]
            labels = labels.split(",")
            new_label = []
            for label in labels:
                if label.startswith(drop_labels):
                    continue
                new_label.append(label)

            if new_label:
                sub_tmp = '{' + ",".join(new_label) + '}'
            else:
                sub_tmp = ""
            new_columns.append(re.sub(r'\{.*?\}', sub_tmp, col))
        df.columns = new_columns

    def create_csv_exporter_df(self, config_path: str) -> pd.DataFrame:

        with open(config_path) as f:
            self.config = json.load(f)

        start_time_str = self.config["start_time"]
        end_time_str = self.config["end_time"]
        save_dir = self.config["save_dir"]
        step = self.config["step"]
        prometheus_url = self.config["prometheus_url"]
        queries = [query['query'] for query in self.config["queries"]]

        self.prometheus_handler = PrometheusHandler(prometheus_url)

        # prometheus unique save dir
        save_time = str(time.ctime()) + ".csv"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        # final save dir
        save_path = os.path.join(save_dir, save_time)

        self.df = self.prometheus_handler.fetch_queries(start_time_str=start_time_str,
                                                        end_time_str=end_time_str,
                                                        queries=queries,
                                                        single_column_query=False,
                                                        step=step)

        self.drop_labels(self.df, tuple(self.config["drop_labels"]))
        self.df.to_csv(save_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    dc = DatasetCreator()
    # dc.create_prometheus_queries_df("../configs/dataset.json")
    dc.create_csv_exporter_df("../configs/csv_exporter_dataset.json")