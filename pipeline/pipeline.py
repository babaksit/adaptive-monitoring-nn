import logging
import os
import time
from datetime import datetime, timedelta
from typing import Union

from darts import TimeSeries
from prometheus_pandas import query

from pipeline.dataset.dataset_loader import DatasetLoader
from pipeline.models.forecast_models import TFTModel, NBeatsModel, ForecastModel
import pandas as pd
import subprocess
from kubernetes import client
import requests
from pipeline.prometheus.handler import PrometheusHandler


class Pipeline:
    """
    Pipeline class

    """
    forecast_model: ForecastModel

    def __init__(self, model_name, dataset_path: str,
                 predict_length: int, fetching_duration: int,
                 queries: list, queries_column_names: list,
                 target_cols: list, time_col: str = "Time",
                 frequency: str = "1min", augment: bool = True,
                 fetching_offset: int = 15,
                 prometheus_url="http://127.0.0.1:9090/"):
        """

        Parameters
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
        """
        self.dataset_loader = DatasetLoader(dataset_path,
                                            time_col, target_cols,
                                            resample_freq=frequency,
                                            augment=augment
                                            )
        self.forecast_model = None
        self.model_dict = {"TFT": TFTModel(), "NBeats": NBeatsModel()}
        self.create_forecast_model(model_name)
        self.predict_length = predict_length
        self.fetching_duration = fetching_duration
        self.predict_frequency = frequency
        self.queries = queries
        self.queries_column_names = queries_column_names
        self.fetching_offset = fetching_offset
        self.stop_pipeline = False
        self.prometheus_url = prometheus_url
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

    def train(self):
        train, val, test = self.dataset_loader.get_train_val_test()
        self.forecast_model.fit(train, val)

    # def get_series_to_predict(self, start_time, length=None, freq=None):
    #     if not length:
    #         length = self.predict_length
    #     if not freq:
    #         freq = self.predict_frequency
    #
    #     rng = pd.date_range(start_time, periods=length, freq=freq)
    #     df = pd.DataFrame({'Val': 0}, index=rng)
    #     return TimeSeries.from_dataframe(df)
    #

    @staticmethod
    def get_series_to_predict(first_series: TimeSeries, second_series: TimeSeries):
        return first_series.append(second_series)

    def predict(self, series):
        return self.forecast_model.predict(self.predict_length, series)

    def is_high_confidence(self, prediction):
        pass

    def __get_queries(self) -> list:
        pass

    def enable_monitoring(self):
        pass

    def save_new_queries_df(self, df, start_time, end_time):

        unique_file_name = start_time.strftime(self.date_format_str) \
                           + "-" + end_time.strftime(self.date_format_str)

        save_path = os.path.join(self.save_new_queries_dir, (unique_file_name + ".csv"))
        df.to_csv(save_path)

    def fetch_metrics(self, duration_minutes, return_series=True) -> Union[pd.DataFrame,
                                                                           TimeSeries]:

        self.prometheus_handler.change_fetching_state(enable=True)
        start_time = datetime.now()
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        time.sleep(duration_minutes * 60)

        # TODO check if we should disable fetching metrics after fetching metrics
        self.prometheus_handler.change_fetching_state(enable=False)
        df = self.prometheus_handler.fetch_queries(start_time.strftime(self.date_format_str),
                                                   end_time.strftime(self.date_format_str),
                                                   self.queries,
                                                   self.queries_column_names)
        self.save_new_queries_df(df, start_time, end_time)

        # TODO check if this works
        if return_series:
            return TimeSeries.from_dataframe(df)
        return df


    def run(self):

        fetched_series = self.fetch_metrics(self.fetching_duration)
        series_to_predict = fetched_series
        last_high_confidence_prediction_time: datetime = None
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

            series_to_predict = self.fetch_metrics(self.predict_length)


if __name__ == '__main__':
    p = Pipeline("TFT", ...)
    p.train()
    p.run()
