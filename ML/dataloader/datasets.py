import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from ML.features.assign import Feature


class MethodDataset(Dataset):
    """
    A Dataset class for loading datasets which was created with different methods by
    dataset/dataset_creator.py in the root directory

    """

    def __init__(self, df: pd.DataFrame, value_col: str):
        """
        Initialize

        Parameters
        ----------
        df : Input dataframe
        value_col : value column name
        """
        self.df = df
        self.value_col = value_col

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
    """
    Dataset class for prometheus
    #TODO add num_workers https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
    """

    def __init__(self, df: pd.DataFrame, value_cols: list, window_size: int =1):
        """

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        value_cols : list
            list of columns which are targets/values
        window_size : int
            Sliding window size
        """
        self.df = df
        self.value_cols = value_cols
        self.window_size = window_size

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

        row = self.df.iloc[index:self.window_size]
        X = torch.Tensor(row.drop(index=self.value_cols))
        y = torch.Tensor(row[self.value_cols])

        return X, y


if __name__ == '__main__':
    pass
