import argparse
import datetime
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from ML.models.model_creator import create_lstm
from ML.dataloader.dataloader import DataLoader
from ML.features.assign import Feature

seed = 37

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(seed)

np.random.seed(seed)
torch.manual_seed(seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train a time series network")
    parser.add_argument('--config-file', type=str,
                        help='Path to the config file', default="../configs/prom_config.json")
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
    model = create_lstm(config, dataloader.get_num_class(), dataloader.get_num_features())
    batch_size = config['batch_size']
    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    device = config["device"]
    model.to(device)
    training_loss_ls = []
    val_loss_ls = []
    num_epochs = config['num_epochs']
    # Train the model
    for epoch in range(num_epochs):
        batch_losses = []
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.view([batch_size, -1, dataloader.get_num_features()]).to(device)
            optimizer.zero_grad()
            y_batch = y_batch.to(device)
            y_hat = model(x_batch)
            # obtain the loss function
            loss = criterion(y_batch, y_hat)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.detach().numpy())

        training_loss = np.mean(batch_losses)
        training_loss_ls.append(training_loss)

        with torch.no_grad():
            batch_val_losses = []
            model.eval()
            for x_val, y_val in val_loader:
                bs = x_val.shape[0]
                x_val = x_val.view([bs, -1, dataloader.get_num_features()]).to(device)
                y_val = y_val.to(device)
                yhat = model(x_val)
                val_loss = criterion(y_val, yhat).item()
                batch_val_losses.append(val_loss)
            validation_loss = np.mean(batch_val_losses)
            val_loss_ls.append(validation_loss)

            # if (epoch <= 10) | (epoch % 10 == 0):
            print(f"[{epoch}/{num_epochs}] Training loss: {training_loss:.4f}\t "
                  f"Validation loss: {validation_loss:.4f}")

    model_save_path = os.path.join(config["model_save_dir"],
                                   f'{model}_{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    torch.save(model.state_dict(), model_save_path)

    plt.plot(training_loss_ls, label="Training loss")
    plt.plot(val_loss_ls, label="Validation loss")
    plt.legend()
    plt.title("Losses")
    plt.show()
    plt.close()
