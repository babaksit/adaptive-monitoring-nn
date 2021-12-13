import pandas as pd


class DataserLoader:
    """
    Class for loading different type of datasets

    """

    @staticmethod
    def load_timeseries(dataset_path, time_col:str
                        , convert_to_duration_time=False) -> pd.DataFrame:
        """
        Loads a timeseries dataset as a pandas DataFrame

        Parameters
        ----------
        dataset_path : path to the timeseries dataset
        convert_to_duration_time : if True then it will convert the timeseries
                                   column to the duration of time between current row
                                   and next row
        time_col : Time column name
        Returns
        -------

        """
        df = pd.read_csv(dataset_path, parse_dates=True)
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)

        if not convert_to_duration_time:
            return df

        df[time_col] = df.index.to_series().diff().dt.seconds.shift(-1)
        df.set_index(time_col, inplace=True)

        return df


if __name__ == '__main__':
    df = DataserLoader.load_timeseries("data/ADDITION_1_2000_300_S.csv", "Time", True)
    print(df)
