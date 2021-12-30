import argparse
import json
from typing import Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch import nn
from torch.utils import data

from ML.dataloader.dataloader import DataLoader
from ML.features.assign import Feature
from ML.models.model_creator import create_lstm


def inverse_transform(scaler: Union[MinMaxScaler, StandardScaler],
                      df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Invert transform of a dataframe

    Parameters
    ----------
    scaler : Scaler for inversing the transform
    df : input Dataframe
    columns : columns to inverse the transform

    Returns
    -------
    Inverted Dataframe
    """
    for col in columns:
        df[col] = scaler.inverse_transform(df[col].values.reshape(-1, 1))
    return df


def format_predictions(predictions: list, values: list,
                       df_test: pd.DataFrame,
                       scaler: Union[MinMaxScaler, StandardScaler]):
    """

    Parameters
    ----------
    predictions : list of predictions
    values : list of real values
    df_test : A test dataframe
    scaler : scaler that has been used for df_test

    Returns
    -------


    """
    vals = np.concatenate(values, axis=0).ravel()
    preds = np.concatenate(predictions, axis=0).ravel()
    df_result = pd.DataFrame(data={"value": vals, "prediction": preds}, index=df_test.head(len(vals)).index)
    df_result = df_result.sort_index()
    df_result = inverse_transform(scaler, df_result, ["value", "prediction"])
    return df_result


def evaluate(model: nn.Module, test_loader: data.DataLoader,
             device: str, num_features: int):
    """
    Evaluate a DataLoader
    Parameters
    ----------
    model : The model to evaluate
    test_loader : Test DataLoader instance
    device : device e.g. cpu, cuda.
    batch_size : batch size of the input
    num_features : number of the features of the input

    Returns
    -------
    Predictions and Real Values
    """

    with torch.no_grad():
        predictions = []
        values = []
        model.eval()
        for x_test, y_test in test_loader:
            batch_size = x_test.shape[0]
            x_test = x_test.view([batch_size, -1, num_features]).to(device)
            y_test = y_test.to(device)
            yhat = model(x_test)
            predictions.append(yhat.to(device).detach().numpy())
            values.append(y_test.to(device).detach().numpy())

    return predictions, values


def calculate_metrics(df):
    return {'mae' : mean_absolute_error(df.value, df.prediction),
            'rmse' : mean_squared_error(df.value, df.prediction) ** 0.5,
            'r2' : r2_score(df.value, df.prediction)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a time series network")
    parser.add_argument('--config-file', type=str,
                        help='Path to the config file', default="../configs/config.json")
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = json.load(f)

    features_list = [Feature.DETAILED_DATETIME]
    scaler = MinMaxScaler()
    dataloader = DataLoader(config['dataset_path'], config['time_column'],
                            config['value_column'],
                            config['val_size'], config['test_size'],
                            config['batch_size'], config['dataset_type'],
                            features_list, scaler)

    train_loader, val_loader, test_loader = dataloader.create_dataloaders()
    model = create_lstm(config, dataloader.get_num_features())

    model_path = '../saved_models/lstm'

    model.load_state_dict(torch.load(model_path))

    predictions, values = evaluate(model, test_loader, config['device'], dataloader.get_num_features())

    df_result = format_predictions(predictions, values, test_loader.dataset.df, dataloader.scaler)

    print(df_result)
    result_metrics = calculate_metrics(df_result)
    print(result_metrics)