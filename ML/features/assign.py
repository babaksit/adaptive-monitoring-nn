import pandas as pd
import numpy as np
import logging
from enum import Enum

__all__ = ["Feature", "detailed_datetime", "cyclical"]


class Feature(Enum):
    DETAILED_DATETIME = 1
    CYCLICAL = 2


def detailed_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign features to the the dataframe, based on the Datetime index of dataframe.
    second, minute, hour, day and dayofweek features are added.

    Parameters
    ----------
    df : Input dataframe

    Returns
    -------
    Assigned features dataframe

    """
    df_features = (
        df.assign(second=df.index.second)
          .assign(minute=df.index.minute)
          .assign(hour=df.index.hour)
          .assign(day_of_week=df.index.dayofweek)
    )

    return df_features


def cyclical(df: pd.DataFrame) -> pd.DataFrame:
    """
    #
    Adding cyclical features using cosine ans sine
    @article{petnehazi2019recurrent,
      title={Recurrent neural networks for time series forecasting},
      author={Petneh{\'a}zi, G{\'a}bor},
      journal={arXiv preprint arXiv:1901.00069},
      year={2019}
    }
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    Dataframe with assigned cyclical features as sin_{col_name} and cos_{col_name}

    """
    cols = ["second", "minute", "hour", "day", "day_of_week"]
    min_max_cols = {"second": (0, 59),
                    "minute": (0, 59),
                    "hour": (0, 23),
                    "day_of_week": (0, 6)
                    }

    for key, value in min_max_cols.items():

        min_cycle_val = value[0]
        max_cycle_val = value[1]

        if max_cycle_val <= 0:
            logging.error("max_cycle_val should be greater than 0 !")
            return None
        if max_cycle_val < min_cycle_val:
            logging.error("max_cycle_val should be greater than or equal to min_cycle_val!")
            return None
        kwargs = {
            f'sin_{key}': lambda x: np.sin(2 * np.pi *
                                           (df[key] - min_cycle_val) / max_cycle_val),
            f'cos_{key}': lambda x: np.cos(2 * np.pi *
                                           (df[key] - min_cycle_val) / max_cycle_val)
        }
        df = df.assign(**kwargs).drop(columns=[key])

    return df
