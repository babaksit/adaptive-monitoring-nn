import logging
import pandas as pd


def remove_constant_cols(df: pd.DataFrame):
    """
    Remove columns which are constant

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    """
    return df.loc[:, (df != df.iloc[0]).any()]
