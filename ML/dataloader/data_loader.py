from typing import Union

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torch.utils import data
from ML import features
from ML.features.assign import Feature
from torch.utils.data import Subset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class MethodDataset(Dataset):
    """
    A Dataset class for loading datasets which was created with different methods by
    dataset/dataset_creator.py in the root directory

    """

    def __init__(self, dataset_path: str, time_col: str, value_col: str
                 , features_list: list = None,
                 scaler: Union[MinMaxScaler, StandardScaler] = None):
        """
        Initialize

        Parameters
        ----------
        dataset_path : Path to the timeseries dataset
        time_col : Time column name
        value_col: Value column name
        features_list: list
                  list of features to assign to the dataframe
        scaler: scaler function to scale dataframe values
        """

        self.dataset_path = dataset_path
        self.time_col = time_col
        self.value_col = value_col
        self.df = pd.read_csv(dataset_path, parse_dates=True)
        self.df[time_col] = pd.to_datetime(self.df[time_col])
        self.df.set_index(time_col, inplace=True)
        if features_list:
            self.assign_features(features_list)
        df_scaled = scaler.fit_transform(self.df.to_numpy())
        self.df = pd.DataFrame(df_scaled, columns=self.df.columns)

    def assign_features(self, features_list: list):
        """
        Assign features to the dataframe

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

    def __len__(self) -> int:
        """
        Returns total number of rows in the dataframe

        Returns
        -------
        Total number of rows
        """

        return len(self.df)

    def __getitem__(self, index: int):
        """
        Returns the time and corresponding value  in the dataframe with given index

        Parameters
        ----------
        index : Integer index to retrieve from dataframe

        Returns
        -------
        The time and corresponding value given by the index
        """

        row = self.df.iloc[index]
        X = torch.Tensor(row.drop(index=self.value_col))
        y = torch.Tensor([row[self.value_col]])

        return X, y


class PrometheusDataset(Dataset):

    def __getitem__(self, index) -> T_co:
        pass


if __name__ == '__main__':
    features_l = [Feature.DETAILED_DATETIME, Feature.CYCLICAL]
    method_dataset = MethodDataset("../../data/ADDITION_1_2000_300_S.csv", "Time",
                                   "Value", features_l, MinMaxScaler())
    print(method_dataset.df)

    # # train_dataset = method_dataset[:50]
    # train_idx, val_idx = train_test_split(list(range(len(method_dataset))), test_size=0.25, shuffle=False)
    # datasets = {'train': Subset(method_dataset, train_idx), 'val': Subset(method_dataset, val_idx)}
    # # print(method_dataset.get_features_size())
    # # print(datasets['train'])
    #
    # train_dataset = datasets['train']
    # val_dataset = datasets['val']
    #
    # train_loader = data.DataLoader(train_dataset, batch_size=1,
    #                                shuffle=False, drop_last=True)
    # val_loader = data.DataLoader(val_dataset, batch_size=1,
    #                              shuffle=False, drop_last=True)
    #
    # X, y = next(iter(train_loader))
    # print("Features shape:", X.shape)
    # print("Target shape:", y.shape)
    #
    # for i in range(len(method_dataset)):
    #     r = method_dataset[i]
    #     print(r)
