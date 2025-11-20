"""
LSTM model for sequence prediction
"""

import torch
import torch.nn as nn

from tradeAI.models.base_model import BaseModel


class LSTMModel(BaseModel):
    """LSTM model for time series prediction."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        """
        Initialize LSTM model.

        Args:
            input_size: Number of input features
            output_size: Number of output classes
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__(input_size, output_size)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Output layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, output_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM forward
        # x: (batch, seq_len, features)
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use the last output
        # If bidirectional, concatenate forward and backward hidden states
        if self.bidirectional:
            # hidden: (num_layers * 2, batch, hidden_size)
            # Take last layer's forward and backward hidden states
            forward_hidden = hidden[-2, :, :]
            backward_hidden = hidden[-1, :, :]
            last_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        else:
            # hidden: (num_layers, batch, hidden_size)
            last_hidden = hidden[-1, :, :]

        # Apply dropout
        last_hidden = self.dropout(last_hidden)

        # Fully connected layer
        output = self.fc(last_hidden)

        return output
