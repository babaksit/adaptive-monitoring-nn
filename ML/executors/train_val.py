import argparse
import json
import torch
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils import data
import numpy as np
from torch.utils.data import Subset
from ML.dataloader.data_loader import MethodDataset
from ML.features.assign import Feature
from ML.models.lstm import LSTM
from sklearn.preprocessing import MinMaxScaler

seed = 25

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(seed)

np.random.seed(seed)
torch.manual_seed(seed)


def split_train_val_dataset(dataset: data.Dataset, split_size: float = 0.2):
    """
    Split a dataset to train and validation dataset
    Parameters
    ----------
    dataset : Input dataset
    split_size : split size

    Returns
    -------

    """
    train_idx, val_idx = train_test_split(list(range(len(dataset))),
                                          test_size=split_size, shuffle=False)
    datasets = {'train': Subset(dataset, train_idx), 'val': Subset(dataset, val_idx)}
    return datasets


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train a time series network")
    parser.add_argument('--config-file', type=str,
                        help='Path to the config file', default="../configs/config.json")
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = json.load(f)

    features = [Feature.DETAILED_DATETIME]

    if config['dataset_type'] == "Method":
        dataset = MethodDataset(config['dataset_path'], config['time_column'],
                                config['value_column'], features, MinMaxScaler())

    datasets = split_train_val_dataset(dataset)

    train_dataset = datasets['train']

    val_dataset = datasets['val']

    batch_size = config["batch_size"]
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=False)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size,
                                 shuffle=False)

    X, y = next(iter(train_loader))
    input_size = X.shape[1]

    lstm = LSTM(config['lstm_num_class'], input_size,
                config['lstm_hidden_size'], config['lstm_num_layers'],
                config['lstm_dropout'])

    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=config['learning_rate'])
    device = config["device"]
    lstm.to(device)
    # lstm.train()

    # Train the model
    for epoch in range(config['num_epochs']):
        batch_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.view([batch_size, -1, input_size]).to(device)
            optimizer.zero_grad()
            y_batch = y_batch.to(device)
            y_hat = lstm(x_batch)
            # obtain the loss function
            loss = criterion(y_batch, y_hat)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.detach().numpy())
        training_loss = np.mean(batch_losses)
        if epoch % 1 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, training_loss))
