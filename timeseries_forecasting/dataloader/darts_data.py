import pandas as pd
import torch
import numpy as np
from darts import TimeSeries

# for reproducibility
from darts.dataprocessing.transformers import Scaler

torch.manual_seed(1)
np.random.seed(1)



def get_rmq_train_val_test(df_path: str, target_cols: list=None, train_size: float = 0.8, val_size: float = 0.1):
    # Read a pandas DataFrame
    df = pd.read_csv(df_path)
    df = df[:5 * 24 * 60 * 60]
    target_cols = ["rabbitmq_messages_publish_rate", "rabbitmq_exchange_messages_published_in_total"]
    df = df[["Time"] + target_cols]
    df['Time'] = pd.to_datetime(df['Time'], infer_datetime_format=True)
    df = df.set_index('Time')
    df = df.resample('1Min').mean()
    df = df.reset_index()
    # Create a TimeSeries, specifying the time and value columns
    series = TimeSeries.from_dataframe(df, 'Time', target_cols)
    scaler = Scaler()
    series_scaled = scaler.fit_transform(series)
    train, val = series_scaled.split_before(train_size)
    val, test = val.split_before(val_size/(1.0-train_size))

    train, val, test


