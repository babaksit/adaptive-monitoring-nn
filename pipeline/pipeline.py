import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Union

import pandas as pd
from darts import TimeSeries

from pipeline.models.forecast_models import TFTModel, NBeatsModel, ForecastModel
from pipeline.prometheus.handler import PrometheusHandler


class Pipeline:
    """
    Pipeline class

    """
    forecast_model: ForecastModel

    def __init__(self, config_path: str):
        """
        Parameters
        ----------
        config_path: Path to the pipeline config file

        Variables
        ----------
        model_name : str
            The name of the model to create
        dataset_path : str
            Path to the dataset
        predict_length : int
            Prediction length in minutes
        target_cols : list
            list of target columns
        time_col : str
            Time column
        frequency : str
            Frequency in which prediction should be made
        augment : bool
            If to augment the dataset
        fetching_offset: int
            Number of seconds to trigger fetching metrics
            before the actual desired time to start fetching
        query_delta_time: str
            string format of time according to "https://pandas.pydata.org/pandas-docs/stable/user_guide/timedeltas.html"
            in order to shift the queries which are fetched during runtime, so it matches the scenario time
        """

        with open(config_path) as f:
            self.config = json.load(f)

        # self.dataset_loader = DatasetLoader(self.config["dataset_path"],
        #                                     self.config["time_col"],
        #                                     self.config["target_cols"],
        #                                     resample_freq=self.config["frequency"],
        #                                     augment=self.config["augment"]
        #                                     )
        self.model_dict = {"TFT": TFTModel(), "NBeats": NBeatsModel()}
        # self.create_forecast_model(self.config["model_name"])
        self.forecast_model = self.load_forecast_model(self.config["model_name"], self.config["model_path"])

        self.predict_length = self.config["predict_length"]
        self.fetching_duration = self.config["fetching_duration"]
        self.predict_frequency = self.config["predict_frequency"]
        self.queries = [query['query'] for query in self.config["queries"]]
        self.queries_column_names = [query['column_name'] for query in self.config["queries"]]
        self.fetching_offset = self.config["fetching_offset"]
        self.query_delta_time = self.config["query_delta_time"]
        self.stop_pipeline = False
        self.prometheus_url = self.config["prometheus_url"]
        self.prometheus_handler = PrometheusHandler(self.prometheus_url)
        self.date_format_str = '%Y-%m-%dT%H:%M:%SZ'
        self.save_new_queries_dir = "prometheus_new_queries"

    def create_forecast_model(self, model_name) -> bool:
        """
        Create forecast model

        Parameters
        ----------
        model_name : str
            Name of forecast model to create

        Returns
        -------
        bool
            If model creation was successful
        """

        try:
            self.forecast_model = self.model_dict[model_name]
        except KeyError:
            logging.error("The model name could not be recognized")
            return False
        self.forecast_model.create_model()
        return True

    # def train(self):
    #     train, val, test = self.dataset_loader.get_train_val_test()
    #     self.forecast_model.fit(train, val)

    def load_forecast_model(self, model_name, model_path):
        pass

    @staticmethod
    def get_series_to_predict(first_series: TimeSeries, second_series: TimeSeries):
        return first_series.append(second_series)

    def predict(self, series):
        return self.forecast_model.predict(self.predict_length, series)

    def is_high_confidence(self, prediction):
        pass

    def save_new_queries_df(self, df, start_time, end_time):

        unique_file_name = start_time.strftime(self.date_format_str) \
                           + "-" + end_time.strftime(self.date_format_str)

        save_path = os.path.join(self.save_new_queries_dir, (unique_file_name + ".csv"))
        df.to_csv(save_path)

    @staticmethod
    def get_start_time(server_time_diff: int = 1):
        """

        Parameters
        ----------
        server_time_diff : int
            difference time between server and current system

        Returns
        -------

        """

        return datetime.now() + timedelta(hours=server_time_diff)

    def shift_df(self, df):
        df.index = df.index.shift(freq=pd.Timedelta(self.query_delta_time))

    def fetch_queries(self, duration_minutes, return_series=True) -> Union[pd.DataFrame,
                                                                           TimeSeries]:

        self.prometheus_handler.change_fetching_state(enable=True)
        start_time = self.get_start_time()
        end_time = start_time + timedelta(minutes=duration_minutes)
        time.sleep(duration_minutes * 60)

        # TODO check if we should disable fetching metrics after fetching metrics
        self.prometheus_handler.change_fetching_state(enable=False)
        df = self.prometheus_handler.fetch_queries(start_time.strftime(self.date_format_str),
                                                   end_time.strftime(self.date_format_str),
                                                   self.queries,
                                                   self.queries_column_names)
        self.shift_df(df)
        self.save_new_queries_df(df, start_time, end_time)

        # TODO check if this works
        if return_series:
            return TimeSeries.from_dataframe(df)
        return df

    def run(self):
        fetched_series = self.fetch_queries(self.fetching_duration)
        series_to_predict = fetched_series
        last_high_confidence_prediction_time: datetime = None
        time_diff_sec = 0.0
        while not self.stop_pipeline:
            prediction = self.predict(series_to_predict)
            if self.is_high_confidence(prediction):
                last_high_confidence_prediction_time = prediction.end_time().to_pydatetime()
                series_to_predict = self.get_series_to_predict(series_to_predict, prediction)
                # crop series_to_predict to the length of fetching_duration/model input_chunk_length
                # since greater would be useless and cause memory usage
                if len(series_to_predict) > self.fetching_duration:
                    series_to_predict = series_to_predict[:-self.fetching_duration]
                continue
            if last_high_confidence_prediction_time:
                time_diff_sec = (last_high_confidence_prediction_time
                                 - datetime.now()).total_seconds()
                # trigger start fetching metrics sooner,
                # since it takes some time for prometheus to discover the enabled ServiceMonitor
                time_diff_sec -= self.fetching_offset

            if time_diff_sec > 0.0:
                time.sleep(time_diff_sec)

            series_to_predict = self.fetch_queries(self.predict_length)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)-4s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    parser = argparse.ArgumentParser(description="Pipeline")
    parser.add_argument('--pipeline-config', type=str,
                        help='Path to the pipeline config file', default="configs/pipeline.json")
    args = parser.parse_args()

    pipeline = Pipeline(args.pipeline_config)
    pipeline.run()