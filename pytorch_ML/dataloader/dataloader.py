import argparse
import json
import logging
from typing import Union

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils import data

from ML import features
from ML.dataloader.datasets import MethodDataset, PrometheusDataset
from ML.features.assign import Feature


class DataLoader:
    def __init__(self, dataset_path: str, time_col: str, val_size: float,
                 test_size: float, batch_size: int, dataset_type: str
                 , features_list: list = None, window_size: int = 1,
                 scaler: Union[MinMaxScaler, StandardScaler] = None):

        """
        Init

        Parameters
        ----------

        dataset_path : str
            Path to the timeseries dataset
        time_col : str
            Time column name
        val_size: float
            size of validation data
        test_size: float
            size of test data
        batch_size : int
            batch size
        dataset_type : str
            type of the dataset
        features_list: list
            Features to assign to the dataframe
        window_size: int
            Sliding window size
        scaler: Union[MinMaxScaler, StandardScaler]
            scaler function to scale dataframe values


        """
        self.time_col = time_col
        self.val_col = None
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.dataset_path = dataset_path
        self.df = None
        self.window_size = window_size
        self.__create_df()
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        if features_list:
            self.__assign_features(features_list)
        self.scaler = scaler

    def __create_df(self):
        """
        Create Dataframe

        Returns
        -------
        None
        """
        # time column is in date time format
        if self.time_col == "Time":
            self.df = pd.read_csv(self.dataset_path, parse_dates=True)
            self.df[self.time_col] = pd.to_datetime(self.df[self.time_col])
            self.df.set_index(self.time_col, inplace=True)
        # time column is in timestamp format
        elif self.time_col == "timestamp":
            self.df = pd.read_csv(self.dataset_path)
            self.df[self.time_col] = pd.to_datetime(self.df[self.time_col], unit='s')
            self.df.set_index(self.time_col, inplace=True)
        else:
            raise ValueError("time_col is not recognizable")
        self.val_col = self.df.columns

    def __get_train_val_test_df(self) -> list:
        """
        Split df to train, validation, test dataframes

        Returns
        -------
        list[pd.Dataframe, pd.Dataframe, pd.Dataframe]
            Splitted train, validation, test dataframes

        """
        train, test = train_test_split(self.df, test_size=(self.val_size + self.test_size), shuffle=False)
        val, test = train_test_split(self.df, test_size=self.test_size, shuffle=False)
        # Avoiding pandas settingwithcopywarning using .copy()
        return train.copy(), val.copy(), test.copy()

    def __assign_features(self, features_list: list) -> None:
        """
        A private function to assign features to the dataframe

        Parameters
        ----------
        features_list : list
            list of features

        Returns
        -------
        None
        """

        for feature in features_list:
            if feature == Feature.DETAILED_DATETIME:
                self.df = features.assign.detailed_datetime(self.df)
                continue
            if feature == Feature.CYCLICAL:
                self.df = features.assign.cyclical(self.df)
                continue

    def create_dataloaders(self) -> list:
        """
        Create train val test dataloaders

        Returns
        -------
        list[data.DataLoader, data.DataLoader, data.DataLoader]
            train_loader, val_loader and test_loader

        """

        train, val, test = self.__get_train_val_test_df()
        if self.scaler:
            train = pd.DataFrame(self.scaler.fit_transform(train), columns=train.columns, index=train.index)
            val = pd.DataFrame(self.scaler.fit_transform(val), columns=val.columns, index=val.index)
            test = pd.DataFrame(self.scaler.fit_transform(test), columns=test.columns, index=test.index)
            # for col in train.columns:
            #     train[col] = self.scaler.fit_transform(train[col].values.reshape(-1, 1))
            # for col in val.columns:
            #     val[col] = self.scaler.fit_transform(val[col].values.reshape(-1, 1))
            # for col in test.columns:
            #     test[col] = self.scaler.fit_transform(test[col].values.reshape(-1, 1))
        #TODO add sliding window
        if self.dataset_type == "Method":
            train_dataset = MethodDataset(train, self.val_col)
            val_dataset = MethodDataset(val, self.val_col)
            test_dataset = MethodDataset(test, self.val_col)

        if self.dataset_type == "Prometheus":
            train_dataset = PrometheusDataset(train, self.val_col, self.window_size)
            val_dataset = PrometheusDataset(val, self.val_col, self.window_size)
            test_dataset = PrometheusDataset(test, self.val_col, self.window_size)

        # TODO add num_workers
        self.train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size,
                                            shuffle=False)
        self.val_loader = data.DataLoader(val_dataset, batch_size=self.batch_size,
                                          shuffle=False)
        self.test_loader = data.DataLoader(test_dataset, batch_size=self.batch_size,
                                           shuffle=False)

        return self.train_loader, self.val_loader, self.test_loader

    def get_num_features(self) -> int:
        """
        Getter function for returning number of features in created dataloaders

        Returns
        -------
        int
            Number of features in created dataloaders
        """

        X, _ = next(iter(self.train_loader))
        return X.shape[1]

    def get_num_class(self) -> int:
        """
        Getter function for returning number of classes/targets in created dataloaders

        Returns
        -------
        int
            Number of classes in created dataloaders
        """
        _, y = next(iter(self.train_loader))
        return y.shape[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a time series network")
    parser.add_argument('--config-file', type=str,
                        help='Path to the config file', default="../configs/prom_config.json")
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = json.load(f)

    features_list = [Feature.DETAILED_DATETIME, Feature.CYCLICAL]
    scaler = MinMaxScaler()
    dataloader = DataLoader(config['dataset_path'], config['time_column'],
                            config['value_column'],
                            config['val_size'], config['test_size'],
                            config['batch_size'], config['dataset_type'],
                            features_list, scaler)

    train_loader, val_loader, test_loader = dataloader.create_dataloaders()
    print(dataloader.get_num_class())
