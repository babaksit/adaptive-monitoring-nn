from ML.models.lstm import LSTM


def create_lstm(config, input_size):

    model = LSTM(config['lstm_num_class'], input_size,
                 config['lstm_hidden_size'], config['lstm_num_layers'],
                 config['lstm_dropout'])

    return model
