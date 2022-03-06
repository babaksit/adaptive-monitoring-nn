import logging
import time

from darts import TimeSeries
from prometheus_pandas import query

from pipeline.dataset.dataset_loader import DatasetLoader
from pipeline.models.forecast_models import TFTModel, NBeatsModel, ForecastModel
import pandas as pd
import subprocess
from kubernetes import client
import requests

class Pipeline:
    """
    Pipeline class

    """
    forecast_model: ForecastModel

    def __init__(self, model_name, dataset_path: str,
                 predict_length: int,
                 target_cols: list, time_col: str = "Time",
                 frequency: str = "1min", augment: bool = True,
                 prometheus_url="http://127.0.0.1:9090/"):
        """

        Parameters
        ----------
        model_name : str
            The name of the model to create
        dataset_path : str
            Path to the dataset
        predict_length : int
            Prediction length
        target_cols : list
            list of target columns
        time_col : str
            Time column
        frequency : str
            Frequency in which prediction should be made
        augment : bool
            If to augment the dataset

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
        self.predict_frequency = frequency
        self.stop_pipeline = False
        self.prometheus_url = prometheus_url
        self.prometheus = query.Prometheus(prometheus_url)

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

    def get_series_to_predict(self, start_time, length=None, freq=None):
        if not length:
            length = self.predict_length
        if not freq:
            freq = self.predict_frequency

        rng = pd.date_range(start_time, periods=length, freq=freq)
        df = pd.DataFrame({'Val': 0}, index=rng)
        return TimeSeries.from_dataframe(df)
        # series = TimeSeries()
        # return series

    def change_fetching_state(self, disable: bool,
                              group="monitoring.coreos.com",
                              version="v1",
                              namespace="default",
                              plural="servicemonitors",
                              name="prometheus-prometheus-node-exporter",
                              ) -> bool:
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
            logging.error("Could not get the monitoring service: "+str(e))
            return False

        if disable:
            # setting to a random string, so prometheus do not scrape it
            conf["spec"]["selector"]["matchLabels"]["release"] = "disable"
        else:
            conf["spec"]["selector"]["matchLabels"]["release"] = "prometheus"

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

        r = requests.post(self.prometheus_url+'/-/reload')
        if r.status_code != 200:
            logging.error("Reloading the prometheus config was not successful: " + str(r.status_code))
            return False
        return True

    def predict(self, start_time):
        # series = TimeSeries()
        # self.forecast_model.predict(self.predict_length, series)
        pass

    def is_high_confidence(self, prediction):
        pass

    def __get_queries(self) -> list:
        pass

    def enable_monitoring(self):
        pass

    def fetch_metrics(self, duration):
        self.change_fetching_state(disable=False)
        time.sleep(duration)
        # for query in self.__get_queries():
        #     res = p.query_range(query, cd[0], cd[1], "1s")
        self.change_fetching_state(disable=True)


    def get_start_time(self):
        pass

    def plot_on_prometheus(self):
        pass

    def run(self):
        start_time = self.get_start_time()
        series = self.get_series_to_predict(start_time)
        while not self.stop_pipeline:
            prediction = self.predict(series, start_time)
            if self.is_high_confidence(prediction):
                start_time = self.get_start_time("next_interval")
                series = self.get_series_to_predict()
                continue
            metrics = self.fetch_metrics(self.predict_length)
            series = self.get_series_to_predict(metrics)



if __name__ == '__main__':
    p = Pipeline("TFT", ...)
    p.train()
    p.run()
