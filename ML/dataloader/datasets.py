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
