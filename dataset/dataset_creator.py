import pandas as pd
from enum import Enum


class CreationMethod(Enum):
    """
    Enum for different methods of creating dataset

    """
    # __init__ = 'value __doc__'
    ADDITION_1 = 1, 'Each new data value is last data value plus 1'


class DatasetCreator:
    """
    Create a pandas time series dataset based the given method.

    """

    def __init__(self, length: int,
                 frequency: str,
                 save_dir=None,
                 start_time="2000",
                 creation_method=CreationMethod.ADDITION_1):
        """
        Initialize Variables

        Parameters
        ----------
        length : number of rows in the dataset
        frequency : Frequency of time series e.g. 'S' which is every second.
                    See https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
                    for a list of frequency aliases
        save_dir : Directory path for saving the dataset
        start_time: start time of time series e.g. 2021-01-01 23:23:23
        creation_method : Method of creating the dataset

        """
        self.length = length
        self.frequency = frequency
        self.save_dir = save_dir
        self.start_time = start_time
        self.creation_method = creation_method

    def create(self) -> pd.Series:
        """
        Create a pandas time series dataset based on the given variables

        Returns
        -------
        Returns created pandas time series
        """

        if self.creation_method == CreationMethod.ADDITION_1:
            idx = pd.date_range(self.start_time, periods=self.length, freq=self.frequency)
            ts = pd.Series(range(len(idx)), index=idx)
            df = pd.DataFrame({'Time': ts.index, 'Value': ts.values})
            df.set_index('Time', inplace=True)
            if self.save_dir:
                self.save_df(df)
            return ts
        else:
            raise Exception('Creation method is not defined')

    def save_df(self, df):
        """
        Save time
        Parameters
        ----------
        df : Dataframe to save

        Returns
        -------
        True If save was successful else False
        """
        save_file_name = self.creation_method.name + "_" \
                         + self.start_time + "_" + str(self.length) \
                         + "_" + self.frequency + ".csv"
        df.to_csv(save_file_name)


if __name__ == '__main__':
    dc = DatasetCreator(100, 'S', "./data")
    print(dc.create())
