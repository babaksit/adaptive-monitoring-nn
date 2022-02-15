
from torch import nn
import pandas as pd


class Predict:
    """
    A class to predict output based on the given model
    """

    def __init__(self, model: nn.Module):
        """

        Parameters
        ----------
        model : The model which will be used to predict
        """
        self.model = model

    def predict_df(self, df: pd.DataFrame):
        """
        Predict the input dataframe
        Parameters
        ----------
        df : input dataframe

        Returns
        -------

        """

    def predict_single_entry(self, input):
        """

        Parameters
        ----------
        input :

        Returns
        -------

        """


if __name__ == '__main__':
    pass
    # df_result = format_predictions(predictions, values, X_test, scaler)
