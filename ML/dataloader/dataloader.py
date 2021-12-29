from typing import Union

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils import data

from ML import features
from ML.dataloader.datasets import MethodDataset
from ML.features.assign import Feature


class DataLoader:
    def __init__(self, dataset_path: str, time_col: str, val_col: str, val_size: float,
                 test_size: float, batch_size: int, dataset_type: str
                 , features_list: list = None,
                 scaler: Union[MinMaxScaler, StandardScaler] = None):

        """
        Init

        Parameters
        ----------

        dataset_path : Path to the timeseries dataset
        time_col : Time column name
        val_col : Value column name
        value_col: Value column name
        val_size: size of validation data
        test_size: size of test data
        batch_size : batch size
        dataset_type : type of the dataset
        features_list: list of features to assign to the dataframe
        scaler: scaler function to scale dataframe values


        """
        self.val_col = val_col
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.df = pd.read_csv(dataset_path, parse_dates=True)
        self.df[time_col] = pd.to_datetime(self.df[time_col])
        self.df.set_index(time_col, inplace=True)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        if features_list:
            self.__assign_features(features_list)
        self.scaler = scaler

    def __get_train_val_test_df(self):
        """
        Split df to train, validation, test dataframes
        Returns
        -------
        Splitted train, validation, test dataframes

        """
        train, test = train_test_split(self.df, test_size=(self.val_size + self.test_size), shuffle=False)
        val, test = train_test_split(self.df, test_size=self.test_size, shuffle=False)
        #Avoiding pandas settingwithcopywarning using .copy()
        return train.copy(), val.copy(), test.copy()

    def __assign_features(self, features_list: list):
        """
        A private function to assign features to the dataframe

        Parameters
        ----------
        features_list : list of features

        Returns
        -------

        """

        for feature in features_list:
            if feature == Feature.DETAILED_DATETIME:
                self.df = features.assign.detailed_datetime(self.df)
                continue
            if feature == Feature.CYCLICAL:
                self.df = features.assign.cyclical(self.df)
                continue

    def create_dataloaders(self):
        """
        Create train val test dataloaders
        Returns
        -------

        """

        train, val, test = self.__get_train_val_test_df()
        if self.scaler:
            for col in train.columns:
                train[col] = self.scaler.fit_transform(train[col].values.reshape(-1, 1))
            for col in val.columns:
                val[col] = self.scaler.fit_transform(val[col].values.reshape(-1, 1))
            for col in test.columns:
                test[col] = self.scaler.fit_transform(test[col].values.reshape(-1, 1))
            # train = pd.DataFrame(df_scaled, columns=train.columns)
            # df_scaled = self.scaler.fit_transform(val)
            # val = pd.DataFrame(df_scaled, columns=val.columns)
            # df_scaled = self.scaler.fit_transform(test)
            # test = pd.DataFrame(df_scaled, columns=test.columns)

        if self.dataset_type == "Method":
            train_dataset = MethodDataset(train, self.val_col)
            val_dataset = MethodDataset(val, self.val_col)
            test_dataset = MethodDataset(test, self.val_col)

        self.train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size,
                                            shuffle=False)
        self.val_loader = data.DataLoader(val_dataset, batch_size=self.batch_size,
                                          shuffle=False)
        self.test_loader = data.DataLoader(test_dataset, batch_size=self.batch_size,
                                           shuffle=False)

        return self.train_loader, self.val_loader, self.test_loader

    def get_num_features(self) -> int:
        """
        Number of features in created dataloaders

        Returns
        -------
        Number of features in created dataloaders
        """

        X, _ = next(iter(self.train_loader))
        return X.shape[1]
