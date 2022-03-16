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
                 target_cols, convert_cols_to_rate: list = None,
                 resample_freq='1Min', shift_df_datetime: str = None,
                 augment=False):
        """

        Parameters
        ----------
        df_path :
        time_col :
        target_cols :
        convert_cols_to_rate :
        resample_freq :
        shift_df_datetime : str
            Shift dataframe index to the beginning of the given date_time

        augment :
        """

        self.df = None
        self.darts_df = None
        self.series_scaled = None
        self.augmented_series = None
        self.scaler = None
        self.shift_df_datetime = shift_df_datetime
        self.time_col = time_col
        self.convert_cols_to_rate = convert_cols_to_rate
        self.target_cols = target_cols
        self.resample_freq = resample_freq
        self.load_df(df_path)
        # self.remove_constant_cols()
        self.create_darts_df()
        # self.create_rate_cols(convert_cols_to_rate)
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

    @staticmethod
    def shift_df_to(df: pd.DataFrame, date_time: str):
        day_start = pd.Timestamp(df.index[0].strftime(date_time))
        df.index = df.index.shift(periods=1, freq=(day_start - df.index[0]))

    def shift_series_to(self, series: TimeSeries, date_time):
        df = series.pd_dataframe()
        self.shift_df_to(df, date_time)
        return TimeSeries.from_dataframe(df)

    def create_darts_df(self):
        """
        Create Pandas DataFrame for the darts library

        Returns
        -------

        """
        self.darts_df = self.df[[self.time_col] + self.target_cols].copy()
        self.darts_df[self.time_col] = pd.to_datetime(self.darts_df[self.time_col], infer_datetime_format=True)
        self.darts_df = self.darts_df.set_index(self.time_col)
        if self.shift_df_datetime:
            self.shift_df_to(self.darts_df, self.shift_df_datetime)
            # day_start = pd.Timestamp(self.darts_df.index[0].strftime(self.shift_df_datetime))
            # self.darts_df.index = self.darts_df.index.shift(periods=1, freq=(day_start - self.darts_df.index[0]))
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
        # self.series_scaled.plot(label="v")

    def augment_specific_times(self, series, date_time_ranges, augmentations=None):
        """

        Parameters
        ----------
        augmentations :
        series :
        date_time_ranges : list
            list of slices date_time_ranges to augment based on
            e.g. [slice("2022-03-01 00:01:02", "2022-03-02 00:02:05"), ...]

        Returns
        -------

        """
        if augmentations is None:
            augmentations = [tsaug.AddNoise(scale=0.001)]
        augmented_sliced_series = []
        for date_time_range in date_time_ranges:
            sdf = series.pd_dataframe()[date_time_range]
            sliced_serie = TimeSeries.from_dataframe(sdf)
            X = sliced_serie.pd_dataframe().to_numpy().swapaxes(0, 1)
            augmented_sliced_serie = []
            for x in X:
                tmp_augmented_series = []
                for aug in augmentations:
                    x_aug = aug.augment(x)
                    tmp_augmented_series = np.concatenate((tmp_augmented_series, x_aug), axis=0)
                augmented_sliced_serie.append(tmp_augmented_series)

            augmented_sliced_serie = np.array(augmented_sliced_serie).T
            # step one period of time ahead
            index = pd.date_range(start=sdf.index[-1], freq=self.resample_freq,
                                  periods=2)
            index = pd.date_range(start=index[1], freq=self.resample_freq,
                                  periods=augmented_sliced_serie.shape[0])
            augmented_sliced_serie = pd.DataFrame(data=augmented_sliced_serie, columns=self.target_cols, index=index)
            augmented_sliced_serie[self.time_col] = index
            augmented_sliced_serie = TimeSeries.from_dataframe(augmented_sliced_serie, self.time_col, self.target_cols)
            augmented_sliced_series.append(augmented_sliced_serie)

        augmented_sliced_series = sum(augmented_sliced_series) / len(augmented_sliced_series)
        augmented_sliced_series = self.shift_series_to(series=augmented_sliced_series, date_time=str(index[0]))
        return augmented_sliced_series

    def simple_add_augment(series: TimeSeries, min_, max_, step_):
        odf = series.pd_dataframe(copy=False)
        result = series.copy()
        shuffled = np.arange(min_, max_, step_)
        np.random.shuffle(shuffled)
        for i in shuffled:
            sdf = result.pd_dataframe(copy=False)
            index = pd.date_range(start=sdf.index[-1], freq="1Min",
                                  periods=2)
            index = pd.date_range(start=index[1], freq="1Min",
                                  periods=odf.shape[0])
            tmp_train = series + i
            tmp_train = tmp_train.pd_dataframe()
            tmp_train.set_index(index, inplace=True)
            tmp_train = TimeSeries.from_dataframe(tmp_train)
            result = result.append(tmp_train)
        return result

    def simple_add_augment(self, series: TimeSeries, min_, max_, step_):

        odf = series.pd_dataframe(copy=False)
        result = series.copy()
        shuffled = np.arange(min_, max_, step_)
        np.random.shuffle(shuffled)
        for i in shuffled:
            sdf = result.pd_dataframe(copy=False)
            index = pd.date_range(start=sdf.index[-1], freq=self.resample_freq,
                                  periods=2)
            index = pd.date_range(start=index[1], freq=self.resample_freq,
                                  periods=odf.shape[0])
            tmp_train = series + i
            tmp_train = tmp_train.pd_dataframe()
            tmp_train.set_index(index, inplace=True)
            tmp_train = TimeSeries.from_dataframe(tmp_train)
            result = result.append(tmp_train)
        return result

    def series_append(self, series_1: TimeSeries, series_2: TimeSeries):
        sdf1 = series_1.pd_dataframe(copy=False)
        sdf2 = series_2.pd_dataframe()
        index = pd.date_range(start=sdf1.index[-1], freq=self.resample_freq,
                              periods=2)
        index = pd.date_range(start=index[1], freq=self.resample_freq,
                              periods=sdf2.shape[0])
        sdf2.set_index(index,inplace=True)

        result = pd.concat([sdf1,sdf2])
        return TimeSeries.from_dataframe(result)
    def augment_series(self, series: TimeSeries, augmentations=None, plot=False):
        X = series.pd_dataframe().to_numpy().swapaxes(0, 1)
        series_df = series.pd_dataframe(copy=False)

        if augmentations is None:
            augmentations = [
                tsaug.AddNoise(scale=0.03),
                tsaug.AddNoise(scale=0.028),
                tsaug.AddNoise(scale=0.032),
                tsaug.AddNoise(scale=0.036),
                tsaug.AddNoise(scale=0.04),
                tsaug.AddNoise(scale=0.029),
                # tsaug.TimeWarp(n_speed_change=20, max_speed_ratio=1.11),
                tsaug.AddNoise(scale=0.033),
                tsaug.AddNoise(scale=0.027),
                tsaug.AddNoise(scale=0.031),
                tsaug.AddNoise(scale=0.035),
                tsaug.AddNoise(scale=0.039),
                tsaug.AddNoise(scale=0.038),
                # tsaug.TimeWarp(n_speed_change=20, max_speed_ratio=1.1),
                # tsaug.Drift(max_drift=0.04, n_drift_points=20),

            ]

        augmented_series = []
        for x in X:
            tmp_augmented_series = x
            for aug in augmentations:
                x_aug = aug.augment(x)
                tmp_augmented_series = np.concatenate((tmp_augmented_series, x_aug), axis=0)
                if plot:
                    self.plot_augment_series(x, x_aug, title=str(aug))
            augmented_series.append(tmp_augmented_series)

        augmented_series = np.array(augmented_series).T

        index = pd.date_range(start=series_df.index[0], freq=self.resample_freq,
                              periods=augmented_series.shape[0])
        augmented_series = pd.DataFrame(data=augmented_series, columns=self.target_cols, index=index)
        augmented_series[self.time_col] = index
        augmented_series = TimeSeries.from_dataframe(augmented_series, self.time_col, self.target_cols)

        return augmented_series

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

    def get_train_val_by_date(self, train_end: str):

        train, val = self.series_scaled.split_before(pd.Timestamp(train_end))

        return train, val


if __name__ == '__main__':
    dl = DatasetLoader('/home/bsi/adaptive-monitoring-nn/notebooks/data/cpu_rate/cpu_rate_test.csv', "Time",
                       ["cpu_rate"], resample_freq="1min", augment=False)
    train, val = dl.get_train_val_by_date("")
    # train, val, test = dl.get_train_val_test(train_size=6 / 7)
    # train.plot()
    # # val.plot()
    # # test.plot()
    #
    # print(train)
    # # print(val)
    # #
    # # print(test)
    # ss = dl.augment_specific_times(train, [slice("2022-02-23 10:37:00", "2022-02-24 10:37:00")])
    #
    # print(ss.pd_dataframe())
    # print(train.pd_dataframe())
    #
    # ss.plot(label="augemnted")
    # # plt.show()
    # #
    # ss = train.append(ss)
    # ss = dl.augment_series(ss)
    #
    # ss.plot(label="augemnted_all")
    # plt.show()
