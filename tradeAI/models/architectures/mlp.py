"""
Multi-Layer Perceptron model
"""

from typing import List

import torch
import torch.nn as nn

from tradeAI.models.base_model import BaseModel


class MLP(BaseModel):
    """Multi-Layer Perceptron for trading prediction."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: List[int] = [128, 64, 32],
        dropout: float = 0.3,
        activation: str = "relu",
        batch_norm: bool = True,
    ):
        """
        Initialize MLP.

        Args:
            input_size: Number of input features
            output_size: Number of output classes
            hidden_layers: List of hidden layer sizes
            dropout: Dropout rate
            activation: Activation function ('relu', 'gelu', 'silu')
            batch_norm: Whether to use batch normalization
        """
        super().__init__(input_size, output_size)

        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.activation_name = activation
        self.batch_norm = batch_norm

        # Build network
        layers = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))

            # Batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))

            # Activation
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "silu":
                layers.append(nn.SiLU())
            else:
                raise ValueError(f"Unknown activation: {activation}")

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        return self.network(x)
