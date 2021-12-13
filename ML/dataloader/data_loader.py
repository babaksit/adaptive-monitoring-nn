from abc import ABC, ABCMeta

from torch.utils.data import Dataset, DataLoader
from torch import load
import pandas as pd


class MethodDataset(Dataset):
    """
    A Dataset class for loading datasets which was created with different methods by
    dataset/dataset_creator.py in the root directory

    """

    def __init__(self, dataset_path: str, time_col: str, value_col: str):
        """
        Initialize

        Parameters
        ----------
        dataset_path : Path to the timeseries dataset
        time_col : Time column name
        value_col: Value column name

        """
        self.dataset_path = dataset_path
        self.time_col = time_col
        self.value_col = value_col
        self.df = pd.read_csv(dataset_path)

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
        X = row[self.time_col]
        y = row[self.value_col]

        return X, y


if __name__ == '__main__':
    pass