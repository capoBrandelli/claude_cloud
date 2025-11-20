"""
Base class for technical indicators
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

import pandas as pd

from tradeAI.utils.logger import get_logger

logger = get_logger(__name__)


class BaseIndicator(ABC):
    """Abstract base class for technical indicators."""

    def __init__(self, name: str, **params):
        """
        Initialize indicator.

        Args:
            name: Indicator name
            **params: Indicator parameters
        """
        self.name = name
        self.params = params

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicator values.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with indicator columns
        """
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __repr__(self) -> str:
        return self.__str__()
