from typing import Dict, Any
from ML.models.lstm import LSTM


def create_lstm(config: Dict[str, Any], num_class: int, input_size: int) -> LSTM:
    """
    Create a lstm model

    Parameters
    ----------
    config : Dict[str, Any]
        JSON config file
    num_class : int
        Number of classes
    input_size : int
        Number of features

    Returns
    -------
    LSTM
        Created lstm model
    """
    model = LSTM(num_class, input_size,
                 config['lstm_hidden_size'], config['lstm_num_layers'],
                 config['lstm_dropout'])

    return model
