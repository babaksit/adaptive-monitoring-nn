import pandas as pd
from torch.utils.data import Dataset
from ML import features
from ML.features.assign import Feature
import torch
import numpy as np


class MethodDataset(Dataset):
    """
    A Dataset class for loading datasets which was created with different methods by
    dataset/dataset_creator.py in the root directory

    """

    def __init__(self, dataset_path: str, time_col: str, value_col: str
                 , features_list: None):
        """
        Initialize

        Parameters
        ----------
        dataset_path : Path to the timeseries dataset
        time_col : Time column name
        value_col: Value column name
        features_list: list
                  list of features to assign to the dataframe

        """

        self.dataset_path = dataset_path
        self.time_col = time_col
        self.value_col = value_col
        self.df = pd.read_csv(dataset_path, parse_dates=True)
        self.df[time_col] = pd.to_datetime(self.df[time_col])
        self.df.set_index(time_col, inplace=True)
        if features_list:
            self.assign_features(features_list)

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
        print(row)
        X = torch.Tensor(row.drop(index="Value"))
        y = torch.Tensor([row[self.value_col]])

        return X, y


if __name__ == '__main__':

    features_l = [Feature.DETAILED_DATETIME, Feature.CYCLICAL]
    method_dataset = MethodDataset("../../data/ADDITION_1_2000_300_S.csv", "Time",
                                   "Value", features_l)

    for i in range(len(method_dataset)):
        r = method_dataset[i]
        print(r)
