from typing import Union, Sequence

import matplotlib.pyplot as plt
from darts import models, TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.utils.likelihood_models import QuantileRegression


class ForecastModel:
    """
    Forecasting Model

    """

    def __init__(self, input_length: int, predict_length: int,
                 quantiles: list, n_epochs: int, batch_size: int
                 ):
        """
        Initialize models

        Parameters
        ----------

        input_length : int
            Input/Window length
        predict_length : int
            Prediction length
        n_epochs : int
            Number of epochs
        batch_size: int
            Batch Size

        """

        self.model_name = None
        self.model = None
        self.input_length = input_length
        self.predict_length = predict_length
        self.quantiles = quantiles
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = 42
        self.log_dir = "logs"
        self.torch_device_str = 'cuda:0'
        self.save_checkpoints = True
        self.log_tensorboard = True

    def create_model(self):
        """
        Abstract function for creating the model

        Returns
        -------

        """
        raise NotImplementedError("Create model has not been implemented")

    def fit(self, train: TimeSeries, val: TimeSeries) -> None:
        """
        Fit the model on train series and evaluating using a validation TimeSeries

        Parameters
        ----------
        train : TimeSeries
            Train TimeSeries data
        val : TimeSeries
            Validation TimeSeries data

        Returns
        -------
        None
        """
        self.model.fit(train, val_series=val, verbose=True)

    def predict(self, predict_length: int, series: TimeSeries,
                num_samples: int = 100, plot: bool = False,
                low_quantile: float = 0.1, high_quantile: float = 0.9) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """
        Predict given series

        Parameters
        ----------
        predict_length : int
            Length of prediction
        series : TimeSeries
            Series to predict
        num_samples : int
            Number of samples to drawn
        plot : bool
            If plot the prediction
        low_quantile : float
            Low quantile for the prediction
        high_quantile :
            High quantile for the prediction

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            One or several time series containing the forecasts of `series`, or the forecast of the training series
                if `series` is not specified and the model has been trained on a single series.


        """

        pred = self.model.predict(series=series[:-predict_length],
                                  n=predict_length,
                                  num_samples=num_samples)
        if plot:
            for i in range(pred.n_components):
                series[-predict_length:].univariate_component(i).plot()
                pred.univariate_component(i).plot(low_quantile=low_quantile,
                                                  high_quantile=high_quantile)
                plt.show()

        return pred

    def load_model(self, path):
        raise NotImplementedError


class TFTModel(ForecastModel):

    def __init__(self, input_length: int = 120, predict_length: int = 60,
                 quantiles: list = [0.1, 0.5, 0.9],
                 n_epochs: int = 10, batch_size: int = 128
                 ):
        """

        Parameters
        ----------
        input_length : int
            Input/Window length
        predict_length : int
            Prediction length
        quantiles : list
            Quantiles for loss function
        n_epochs : int
            Number of epochs
        n_epochs : int
            Number of epochs
        batch_size: int
            Batch Size

        """
        super().__init__(self, input_length, predict_length,
                         quantiles, n_epochs, batch_size)

    def create_model(self):
        """
        Create TFT model

        Returns
        -------

        """
        self.model = models.TFTModel(
            input_chunk_length=self.input_length,
            output_chunk_length=self.predict_length,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            add_encoders={
                'datetime_attribute': {'past': ['dayofweek'], 'future': ['dayofweek']},
                "position": {"past": ["relative"], "future": ["relative"]},
                'custom': {'past': [lambda idx: idx.minute + (idx.hour * 60)],
                           'future': [lambda idx: idx.minute + (idx.hour * 60)]},
                'transformer': Scaler()
            },
            likelihood=QuantileRegression(
                quantiles=self.quantiles
            ),
            random_state=self.random_state,
            log_dir=self.log_dir,
            torch_device_str=self.torch_device_str,
            save_checkpoints=self.save_checkpoints,
            log_tensorboard=self.log_tensorboard
        )

    def load_model(self, path):
        self.model = models.TFTModel().load_model(path)
        return self.model


class NBeatsModel(ForecastModel):
    def __init__(self, input_length: int = 120, predict_length: int = 60,
                 quantiles: list = [0.1, 0.5, 0.9],
                 n_epochs: int = 10, batch_size: int = 128
                 ):
        """

        Parameters
        ----------
        input_length : int
            Input/Window length
        predict_length : int
            Prediction length
        quantiles : list
            Quantiles for loss function
        n_epochs : int
            Number of epochs
        n_epochs : int
            Number of epochs
        batch_size: int
            Batch Size

        """
        super().__init__(self, input_length, predict_length,
                         quantiles, n_epochs, batch_size)

    def create_model(self):
        """
        Create NBeats model

        Returns
        -------

        """

        self.model = models.NBEATSModel(
            input_chunk_length=self.input_length,
            output_chunk_length=self.predict_length,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            add_encoders={
                'datetime_attribute': {'past': ['dayofweek']},
                'custom': {'past': [lambda idx: (idx.minute) + (idx.hour * 60)]},
                "position": {"past": ["absolute", "relative"]},
                'transformer': Scaler()
            },
            likelihood=QuantileRegression(
                quantiles=self.quantiles
            ),
            random_state=self.random_state,
            log_dir=self.log_dir,
            torch_device_str=self.torch_device_str,
            save_checkpoints=self.save_checkpoints,
            log_tensorboard=self.log_tensorboard
        )

    def load_model(self, path):
        self.model = models.NBEATSModel().load_model(path)
        return self.model
