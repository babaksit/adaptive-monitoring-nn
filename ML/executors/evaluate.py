import argparse
import json
import logging
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


def format_predictions(predictions: list, values: list,
                       dataloader: DataLoader,
                       scaler: Union[MinMaxScaler, StandardScaler]) -> list:
    """

    Parameters
    ----------
    predictions : list
        list of predictions
    values : list
        list of real values
    dataloader : DataLoader
        dataloader object
    scaler :  Union[MinMaxScaler, StandardScaler]
        scaler that has been used for df_test

    Returns
    -------
    list
        predication dataframe and real values dataframe

    """

    num_class = dataloader.get_num_class()
    num_features = dataloader.get_num_features()
    df_test = dataloader.test_loader.dataset.df
    temp_feature_columns = ["feature_" + str(i) for i in range(num_features)]
    vals = np.concatenate(values, axis=0).ravel().reshape((-1, num_class))
    df_vals = pd.DataFrame(data=vals, index=df_test.index)
    for temp_feature_column in temp_feature_columns:
        df_vals[temp_feature_column] = 0.0
    df_vals = pd.DataFrame(data=scaler.inverse_transform(df_vals), index=df_test.index, columns=df_test.columns)
    df_vals.drop(df_vals.columns.difference(dataloader.val_col), axis=1, inplace=True)
    df_vals = df_vals.sort_index()

    preds = np.concatenate(predictions, axis=0).ravel().reshape((-1, num_class))
    df_preds = pd.DataFrame(data=preds, index=df_test.index)
    for temp_feature_column in temp_feature_columns:
        df_preds[temp_feature_column] = 0.0

    df_preds = pd.DataFrame(data=scaler.inverse_transform(df_preds), index=df_test.index, columns=df_test.columns)
    df_preds.drop(df_preds.columns.difference(dataloader.val_col), axis=1, inplace=True)
    df_preds = df_preds.sort_index()

    # df_result = pd.DataFrame(data=(preds-vals), index=df_test.index)
    # df_result = df_result.sort_index()
    # df_result = scaler.inverse_transform(df_result)
    return df_preds, df_vals


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


def calculate_metrics(df_predict: pd.DataFrame, df_val: pd.DataFrame):
    """
    Calculate MAE, RMSE and R2 between prediction dataframe and values dataframe

    Parameters
    ----------
    df_predict : pd.DataFrame
        prediction dataframe
    df_val : pd.DataFrame
        values dataframe

    Returns
    -------
    dict
        dictionary of MAE, RMSE and R2
    """
    return {'mae': mean_absolute_error(df_val, df_predict),
            'rmse': mean_squared_error(df_val, df_predict) ** 0.5,
            'r2': r2_score(df_val, df_predict)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a time series network")
    parser.add_argument('--config-file', type=str,
                        help='Path to the config file', default="../configs/prom_config.json")
    parser.add_argument('--model-path', type=str,
                        help='Path to the saved model')
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = json.load(f)

    features_list = [Feature.DETAILED_DATETIME, Feature.CYCLICAL]
    scaler = MinMaxScaler()
    dataloader = DataLoader(config['dataset_path'], config['time_column'],
                            config['val_size'], config['test_size'],
                            config['batch_size'], config['dataset_type'],
                            features_list, scaler)

    train_loader, val_loader, test_loader = dataloader.create_dataloaders()

    model = create_lstm(config, dataloader.get_num_class(), dataloader.get_num_features())
    device = config['device']
    model.to(device)
    model_path = args.model_path

    model.load_state_dict(torch.load(model_path))

    predictions, values = evaluate(model, test_loader, device, dataloader.get_num_features())

    df_preds, df_vals = format_predictions(predictions, values, dataloader, dataloader.scaler)

    logging.debug("df_preds: "+ df_preds)
    logging.debug("df_vals: " + df_vals)

    result_metrics = calculate_metrics(df_preds, df_vals)

    logging.debug("result_metrics: " + result_metrics)
