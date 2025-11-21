"""
Base model class for TradeAI
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn

from tradeAI.utils.logger import get_logger

logger = get_logger(__name__)


class BaseModel(nn.Module, ABC):
    """Abstract base class for all neural network models."""

    def __init__(self, input_size: int, output_size: int, **kwargs):
        """
        Initialize base model.

        Args:
            input_size: Number of input features
            output_size: Number of output classes/values
            **kwargs: Additional model-specific parameters
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kwargs = kwargs

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        pass

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        """
        Save model state dict.

        Args:
            path: Path to save model
        """
        torch.save(self.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str, device: Optional[str] = None) -> None:
        """
        Load model state dict.

        Args:
            path: Path to load model from
            device: Device to load to
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.load_state_dict(torch.load(path, map_location=device))
        logger.info(f"Model loaded from {path}")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(params={self.get_num_parameters():,})"
