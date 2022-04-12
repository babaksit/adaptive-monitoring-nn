import argparse
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union, Sequence, Dict, Type

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import (
    BlockRNNModel,
    NBEATSModel,
    TFTModel
)
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

from pipeline.dataset.dataset_loader import DatasetLoader
from pipeline.models.forecast_models import ForecastModel
from pipeline.prometheus.exporter_api_handler import ExporterApi
from pipeline.prometheus.handler import PrometheusHandler
from pipeline.utils.util import progressbar_sleep


def generate_encoders(idxs):
    days = ((idxs.second + idxs.minute * 60 + idxs.hour * 60 * 60 + idxs.dayofweek * 24 * 60 * 60) // (24 * 60)) % 7
    encoders = []
    for day in days:
        if day == 0:
            encoders.append(1)
        elif day == 1 or day == 2 or day == 3 or day == 4:
            encoders.append(2)
        elif day == 5 or day == 6:
            encoders.append(3)
    return encoders


class Pipeline:
    """
    Pipeline class

    """
    model_dict: Dict[str, Type[Union[NBEATSModel, TFTModel, BlockRNNModel]]]
    forecast_model: TorchForecastingModel

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

        self.model_dict = {"TFT": TFTModel, "NBeats": NBEATSModel, "LSTM": BlockRNNModel}
        # self.create_forecast_model(self.config["model_name"])
        self.forecast_model = None
        self.load_forecast_model()
        self.pipeline_created_time = str(datetime.now())
        self.prediction_queries_save_dir = os.path.join("logs", self.pipeline_created_time,
                                                        self.config["prediction_queries_save_dir"])
        self.prometheus_url = self.config["prometheus_url"]
        self.prometheus_handler = PrometheusHandler(self.prometheus_url)
        self.drop_metrics = self.config["drop_metrics"]
        self.keep_metric_url = self.config["keep_metric_url"]
        self.drop_metric_url = self.config["drop_metric_url"]
        self.start_csv_exporter_url = self.config["start_csv_exporter_url"]
        self.clear_keep_list_url = self.config["clear_keep_list_url"]
        self.predict_length = self.config["predict_length"]
        self.fetching_duration = self.config["fetching_duration"]
        self.predict_frequency = self.config["frequency"]
        self.std_threshold = self.config["std_threshold"]
        self.cols = []
        self.col_query_dict = {}
        self.create_query_dict(self.config["queries"])
        self.exporter_api = ExporterApi(col_query_dict=self.col_query_dict,
                                        clear_keep_list_url=self.clear_keep_list_url,
                                        keep_metric_url=self.keep_metric_url,
                                        drop_metric_url=self.drop_metric_url,
                                        start_csv_exporter_url=self.start_csv_exporter_url)
        # self.queries = [query['query'] for query in self.config["queries"]]
        # self.queries_column_names = [query['column_name'] for query in self.config["queries"]]
        self.exporter_api.drop_metrics(self.config["drop_metrics"])
        self.fetching_offset = self.config["fetching_offset"]
        now = pd.to_datetime(datetime.now())
        ref = pd.to_datetime(self.config["query_delta_time"])
        self.query_delta_time = pd.Timedelta(now - ref)
        self.stop_pipeline = False
        self.date_format_str = '%Y-%m-%dT%H:%M:%SZ'
        self.save_new_queries_dir = "prometheus_new_queries"
        self.merged_series: TimeSeries = None

    def load_forecast_model(self):
        self.forecast_model = ForecastModel.load_model(self.config["model_path"],
                                                       self.config["torch_device"])

    def create_query_dict(self, queries):
        for query in queries:
            self.col_query_dict[query['column_name']] = {"query": query['query'],
                                                         "metrics": query['metrics']}
            self.cols.append(query['column_name'])

    @staticmethod
    def get_series_to_predict(first_series: TimeSeries, second_series: TimeSeries):
        return DatasetLoader.series_append(first_series, second_series)

    def save_prediction_queries(self, df: pd.DataFrame, type_str: str):

        output_file = str(df.index[0]) + ".csv"
        output_dir = Path(self.prediction_queries_save_dir
                          , type_str)
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / output_file)
        logging.debug("saved " + type_str + " for df with start time of " + str(df.index[0]))

    def save_merged_series(self):
        output_file = "merged.csv"
        output_dir = Path(self.prediction_queries_save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.merged_series.to_csv(output_dir / output_file)
        logging.debug("saved merged series")

    def predict(self, series, single_pred: bool, save_df: bool = False) -> TimeSeries:

        series_to_predict = self.shift_series(series, shift_back=False)

        if single_pred:
            prediction = self.forecast_model.predict(series=series_to_predict,
                                                     n=self.predict_length,
                                                     num_samples=1)
        else:
            prediction = self.forecast_model.predict(series=series_to_predict,
                                                     n=self.predict_length,
                                                     num_samples=100)
        if save_df:
            if not single_pred:
                prediction_to_save = self.forecast_model.predict(series=series_to_predict,
                                                                 n=self.predict_length,
                                                                 num_samples=1)
                self.save_prediction_queries(prediction_to_save.pd_dataframe(copy=False),
                                             type_str="prediction")
            else:
                self.save_prediction_queries(prediction.pd_dataframe(copy=False),
                                             type_str="prediction")

        return self.shift_series(prediction, shift_back=True)

    def get_cols_to_fetch(self, prediction: Union[TimeSeries, Sequence[TimeSeries]]):
        cols_to_fetch = []
        for i, component in enumerate(prediction.components):
            pred = prediction.univariate_component(i)
            pred = pred.all_values()  # Time X Components X samples
            pred = np.squeeze(pred)  # Time X samples
            std = np.mean(np.std(pred, axis=1))
            if std > self.std_threshold:
                logging.warning("Component: " + str(component) + " has std value more than threshold:"
                                + str(self.std_threshold) + "std: " + str(std))
                cols_to_fetch.append(component)
            else:
                logging.debug("Component: " + str(component) + " has std value less than threshold:"
                              + str(self.std_threshold) + "std: " + str(std))
        return cols_to_fetch

    def save_new_queries_df(self, df, start_time, end_time):

        unique_file_name = start_time.strftime(self.date_format_str) \
                           + "-" + end_time.strftime(self.date_format_str)

        save_path = os.path.join(self.save_new_queries_dir, (unique_file_name + ".csv"))
        df.to_csv(save_path)

    @staticmethod
    def get_current_time(server_time_diff: int = 0):
        """

        Parameters
        ----------
        server_time_diff : int
            difference time between server and current system

        Returns
        -------

        """

        return datetime.now()

    def shift_series(self, series: TimeSeries, shift_back=False) -> TimeSeries:
        coef = -1 if shift_back else 1
        time_index = series.time_index.shift(freq=coef * self.query_delta_time)
        return TimeSeries.from_times_and_values(time_index, series.all_values(), columns=series.columns)

    def fetch_queries(self, duration_seconds, cols: list, start_time: datetime,
                      return_series=True, save_df: bool = False) -> Union[pd.DataFrame, TimeSeries]:

        end_time = start_time + timedelta(seconds=duration_seconds)

        # tmp = start_time
        # start_time = end_time
        # end_time = tmp

        logging.debug("starting fetching queries for components: " + str(cols))
        logging.debug("start time: " + str(start_time) + " - end time: " + str(end_time))
        self.prometheus_handler.change_fetching_state(enable=True)
        # TODO get target_service_name from config file
        self.prometheus_handler.check_target_exits(target_service_name="csv-exporter")
        self.exporter_api.keep_metrics(cols, keep_all=len(self.cols) == len(cols))

        queries = []
        for col in cols:
            queries.append(self.col_query_dict[col]["query"])

        # TODO uncomment this line
        progressbar_sleep(duration_seconds)
        df = self.prometheus_handler.fetch_queries(
            start_time.strftime(self.date_format_str),
            end_time.strftime(self.date_format_str),
            queries,
            cols
        )
        # TODO check if we should disable fetching metrics after fetching metrics
        self.prometheus_handler.change_fetching_state(enable=False)
        # TODO check if it is needed to save the metrics, as prometheus already have it
        # self.save_new_queries_df(df, start_time, end_time)
        if save_df:
            self.save_prediction_queries(df, "queries")
        # TODO check if this works
        if return_series:
            return TimeSeries.from_dataframe(df)
        return df

    def wait_before_fetching(self, last_pred_time: datetime):

        time_diff_sec = 0.0
        if last_pred_time:
            time_diff_sec = (last_pred_time
                             - self.get_current_time()).total_seconds()
            # trigger start fetching metrics sooner,
            # since it takes some time for prometheus to discover the enabled ServiceMonitor
            time_diff_sec -= self.fetching_offset

        if time_diff_sec > 0.0:
            logging.debug("last prediction time: " + str(last_pred_time))
            logging.debug("current time: " + str(self.get_current_time()))
            logging.debug("waiting before fetching metrics again for: "
                          + str(time_diff_sec) + " seconds")
            progressbar_sleep(int(time_diff_sec))
            # time.sleep(time_diff_sec)
        logging.debug("waiting for fetching metrics has finished")

    @staticmethod
    def merge_prediction_fetched_series(prediction: TimeSeries, fetched_series: TimeSeries) \
            -> TimeSeries:

        logging.debug("Merging prediction and fetched series")
        if prediction.start_time() != fetched_series.start_time():
            logging.warning("prediction and fetched series start time are different: " \
                            + str(prediction.start_time()) + "<>" + str(fetched_series.start_time()))

            if prediction.start_time() < fetched_series.start_time():
                # TODO check this one
                num_timesteps = prediction.slice(prediction.start_time(),
                                                 fetched_series.start_time())[:-1].n_timesteps
                diff_series = fetched_series.shift(-num_timesteps)
                diff_series = diff_series[0]
                for i in range(1, num_timesteps):
                    diff_series = diff_series.append_values(diff_series[0].values())
                logging.warning("extended series: " + str(diff_series))
                logging.warning("fetched_series: " + str(fetched_series))
                fetched_series = diff_series.concatenate(fetched_series)
                logging.warning("new prediction and fetched series start time are: " \
                                + str(prediction.start_time()) + "<>" + str(fetched_series.start_time()))

        if prediction.end_time() < fetched_series.end_time():
            raise NotImplementedError

        prediction_components = prediction.components.to_list()
        fetched_series_components = fetched_series.components.to_list()
        diff_components = list(set(prediction_components) - set(fetched_series_components))
        logging.debug("diff_components: " + str(diff_components))
        logging.debug("prediction_components" + str(prediction_components))
        prediction_diff_series = prediction[diff_components].slice(fetched_series.start_time(),
                                                                   fetched_series.end_time())
        return prediction_diff_series.concatenate(fetched_series, axis=1)

    def merge_series(self, series):
        if self.merged_series is None:
            self.merged_series = series
            return
        self.merged_series = DatasetLoader.series_append(self.merged_series, series)
        self.save_merged_series()

    def run(self):
        self.exporter_api.start_csv_exporter()
        series_to_predict = self.fetch_queries(self.fetching_duration, self.cols,
                                               self.get_current_time())
        logging.debug(series_to_predict.pd_dataframe(copy=False))
        self.merge_series(series_to_predict)
        last_pred_time: datetime = None
        while not self.stop_pipeline:
            prediction = self.predict(series_to_predict, single_pred=False)
            cols_to_fetch = self.get_cols_to_fetch(prediction)
            # The prediction was high confidence for all the metrics
            if not cols_to_fetch:
                prediction = self.predict(series_to_predict, single_pred=True, save_df=False)
                last_pred_time = prediction.end_time().to_pydatetime()
                logging.debug("last prediction time: " + str(last_pred_time))
                logging.debug("current time: " + str(self.get_current_time()))
                self.merge_series(prediction)
                series_to_predict = self.get_series_to_predict(series_to_predict, prediction)
                # crop series_to_predict to the length of fetching_duration/model input_chunk_length
                # since greater would be useless and cause memory usage
                if len(series_to_predict) > self.fetching_duration:
                    series_to_predict = series_to_predict[-self.fetching_duration:]
                continue
            self.wait_before_fetching(last_pred_time)
            last_pred_time = None
            fetched_series = self.fetch_queries(self.fetching_duration, cols_to_fetch,
                                                prediction.start_time().to_pydatetime())
            # All components are fetched no need for prediction
            if len(cols_to_fetch) == len(self.cols):
                series_to_predict = fetched_series
                self.merge_series(series_to_predict)
                continue
            single_prediction = self.predict(series_to_predict, single_pred=True, save_df=False)
            series_to_predict = self.merge_prediction_fetched_series(single_prediction, fetched_series)
            self.merge_series(series_to_predict)
            # if not series_to_predict:
            #     logging.error("could not merge prediction and fetched series -> using last prediction series")
            #     series_to_predict = single_prediction


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
