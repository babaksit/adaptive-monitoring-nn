import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):
    """
    LSTM class using pytorch nn.LSTM
    """

    def __init__(self, num_class: int, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        """
        Initialize

        Parameters
        ----------
        num_class : number of classes
        input_size : The number of expected features in the input data
        hidden_size : The number of features in the hidden state h
        num_layers : Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to
            form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results.
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal to

        Reference: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

        """
        super(LSTM, self).__init__()

        self.num_class = num_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # batch_first was set to true so that the input and output tensors are provided
        # as (batch, seq, feature) instead of (seq, batch, feature)

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)

        # connecting last layer with nn.Linear to number of classes
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass function.

        Parameters
        ----------
        x : Input Tensor data

        Returns
        -------
        Output tensor data
        """
        # hidden state
        h = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # internal state
        c = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # lstm with input, hidden, and internal state
        output, (hout, _) = self.lstm(x, (h, c))
        # reshaping the data for Dense layer
        output = output.view(-1, self.hidden_size)
        # Final Output
        output = self.fc(output)

        return output
