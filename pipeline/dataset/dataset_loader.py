import pandas as pd
import numpy as np
import torch
import tsaug
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
import matplotlib.pyplot as plt


class DatasetLoader:
    """
    Load Pandas DataFrame

    """

    def __init__(self, df_path: str, time_col: str,
                 target_cols, convert_cols_to_rate: list = None, resample_freq='1Min',
                 augment=False):

        self.df = None
        self.darts_df = None
        self.series_scaled = None
        self.augmented_series = None
        self.scaler = None
        self.time_col = time_col
        self.convert_cols_to_rate = convert_cols_to_rate
        self.target_cols = target_cols
        self.resample_freq = resample_freq
        self.load_df(df_path)
        self.remove_constant_cols()
        self.create_darts_df()
        self.create_rate_cols(convert_cols_to_rate)
        self.scale_darts_series()
        if augment:
            self.series_scaled = self.augment_series()

    def load_df(self, df_path: str):
        self.df = pd.read_csv(df_path)

    def remove_constant_cols(self):
        self.df = self.df.loc[:, (self.df != self.df.iloc[0]).any()]

    def create_rate_cols(self, cols):
        if not cols:
            return
        self.darts_df[cols] = self.darts_df[cols].shift(-2) - self.darts_df[cols]
        # Fill NaNs with preceding values
        self.darts_df[cols] = self.darts_df[cols].fillna(method='ffill')

    def create_darts_df(self):
        """
        Create Pandas DataFrame for the darts library

        Returns
        -------

        """
        self.darts_df = self.df[[self.time_col] + self.target_cols].copy()
        self.darts_df[self.time_col] = pd.to_datetime(self.darts_df[self.time_col], infer_datetime_format=True)
        self.darts_df = self.darts_df.set_index(self.time_col)
        self.darts_df = self.darts_df.resample(self.resample_freq).mean()
        self.darts_df = self.darts_df.reset_index()

    def scale_darts_series(self):
        """

        Returns
        -------

        """

        series = TimeSeries.from_dataframe(self.darts_df, self.time_col, self.target_cols)
        torch.manual_seed(1)
        np.random.seed(1)
        self.scaler = Scaler()
        # TODO scale train test val separately
        self.series_scaled = self.scaler.fit_transform(series)
        self.series_scaled.plot(label="v")

    def augment_series(self, plot=False):
        X = self.series_scaled.pd_dataframe().to_numpy().swapaxes(0, 1)
        augmentations = [
            tsaug.AddNoise(scale=0.002),
            tsaug.Convolve(window="flattop", size=15),
            tsaug.Drift(max_drift=0.01, n_drift_points=20),
            tsaug.Pool(size=5),
            tsaug.Quantize(n_levels=200),
            tsaug.TimeWarp(n_speed_change=10, max_speed_ratio=1.01),
            # repeat randomly
            tsaug.Drift(max_drift=0.005, n_drift_points=20),
            tsaug.Pool(size=6),
            tsaug.Convolve(window="flattop", size=16),
            tsaug.AddNoise(scale=0.001),
            tsaug.Quantize(n_levels=250),
            tsaug.TimeWarp(n_speed_change=10, max_speed_ratio=1.005),
        ]

        self.augmented_series = []
        for x in X:
            tmp_augmented_series = x
            for aug in augmentations:
                x_aug = aug.augment(x)
                tmp_augmented_series = np.concatenate((tmp_augmented_series, x_aug), axis=0)
                if plot:
                    self.plot_aug(x, x_aug, title=str(aug))
            self.augmented_series.append(tmp_augmented_series)

        self.augmented_series = np.array(self.augmented_series).T

        index = pd.date_range(start=self.df[self.time_col].iloc[0], freq=self.resample_freq,
                              periods=self.augmented_series.shape[0])
        self.augmented_series = pd.DataFrame(data=self.augmented_series, columns=self.target_cols, index=index)
        self.augmented_series[self.time_col] = index
        self.augmented_series = TimeSeries.from_dataframe(self.augmented_series, self.time_col, self.target_cols)

        return self.augmented_series

    @staticmethod
    def plot_augment_series(x, x_aug, m=None, n=None, title=""):
        if n is None:
            m = 0
            n = x.shape[0]
        plt.figure(figsize=(8, 4), dpi=120)
        plt.plot(x_aug[m:n], label="augmented", color="b")
        plt.plot(x[m:n], label="real", color="g")
        plt.legend()
        plt.title(title)
        plt.show()

    def get_train_val_test(self, train_size=0.8, val_test_ratio=0.5):

        train, val = self.series_scaled.split_before(train_size)
        val, test = val.split_before(val_test_ratio)

        return train, val, test
