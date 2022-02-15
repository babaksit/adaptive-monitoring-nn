import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel
import matplotlib.pyplot as plt

target_cols = ["v1", "v2"]

rdf = pd.read_csv('test.csv')
rdf = rdf[:5*24*60*60]


rdf = rdf[["Time"]+target_cols]
rdf['Time'] = pd.to_datetime(rdf['Time'], infer_datetime_format=True)
rdf = rdf.set_index('Time')
rdf = rdf.resample('1Min').mean()
rdf = rdf.reset_index()


series = TimeSeries.from_dataframe(rdf, 'Time', target_cols)
torch.manual_seed(1)
np.random.seed(1)
scaler = Scaler()
series_scaled = scaler.fit_transform(series)
series_scaled.plot(label="v")
train, val = series_scaled.split_before(0.8)
val, test = val.split_before(0.5)


my_model = RNNModel(
    model="LSTM",
    training_length=60,
    input_chunk_length=60,
    n_epochs=50,
    work_dir="logs",
    log_tensorboard=True,
    add_encoders={
    'cyclic': { 'future': ['minute','hour','day']},
    'datetime_attribute': {'future': ['minute','hour', 'dayofweek']},
    'position': { 'future': ['relative', 'absolute']},
    'transformer': Scaler()
    },
    optimizer_kwargs={"lr": 1e-3},
)

my_model.fit(train, val_series=val, verbose=True)

pred = my_model.predict(n=60)

for i in range(pred.n_components):
    val.univariate_component(i).plot()
    pred.univariate_component(i).plot()
    plt.show()