"""
Base data provider class for TradeAI
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List

import pandas as pd

from tradeAI.utils.logger import get_logger
from tradeAI.utils.validators import validate_symbol, validate_timeframe

logger = get_logger(__name__)


class ProviderError(Exception):
    """Raised when a data provider encounters an error."""

    pass


class BaseDataProvider(ABC):
    """
    Abstract base class for data providers.

    All data providers must implement this interface.
    """

    def __init__(
        self,
        name: str,
        rate_limit: Optional[int] = None,
        supported_asset_classes: Optional[List[str]] = None,
        supported_timeframes: Optional[List[str]] = None,
    ):
        """
        Initialize base data provider.

        Args:
            name: Provider name
            rate_limit: Rate limit (requests per minute)
            supported_asset_classes: List of supported asset classes
            supported_timeframes: List of supported timeframes
        """
        self.name = name
        self.rate_limit = rate_limit
        self.supported_asset_classes = supported_asset_classes or []
        self.supported_timeframes = supported_timeframes or []
        self.logger = get_logger(f"{__name__}.{name}")

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '1m', '5m', '1h', '1d')
            start_date: Start date for data
            end_date: End date for data
            limit: Maximum number of bars to fetch

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume

        Raises:
            ProviderError: If data fetching fails
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the provider is available (e.g., API key configured).

        Returns:
            True if provider is available
        """
        pass

    def supports_asset_class(self, asset_class: str) -> bool:
        """
        Check if provider supports an asset class.

        Args:
            asset_class: Asset class (e.g., 'stocks', 'forex', 'crypto')

        Returns:
            True if supported
        """
        if not self.supported_asset_classes:
            return True  # Support all if not specified

        return asset_class.lower() in [ac.lower() for ac in self.supported_asset_classes]

    def supports_timeframe(self, timeframe: str) -> bool:
        """
        Check if provider supports a timeframe.

        Args:
            timeframe: Timeframe string

        Returns:
            True if supported
        """
        if not self.supported_timeframes:
            return True  # Support all if not specified

        return timeframe.lower() in [tf.lower() for tf in self.supported_timeframes]

    def validate_request(
        self,
        symbol: str,
        timeframe: str,
        asset_class: Optional[str] = None,
    ) -> None:
        """
        Validate a data request.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            asset_class: Asset class (optional)

        Raises:
            ProviderError: If validation fails
        """
        # Validate symbol
        try:
            validate_symbol(symbol)
        except Exception as e:
            raise ProviderError(f"Invalid symbol: {e}")

        # Validate timeframe
        try:
            validate_timeframe(timeframe)
        except Exception as e:
            raise ProviderError(f"Invalid timeframe: {e}")

        # Check timeframe support
        if not self.supports_timeframe(timeframe):
            raise ProviderError(
                f"Provider {self.name} does not support timeframe {timeframe}. "
                f"Supported: {self.supported_timeframes}"
            )

        # Check asset class support
        if asset_class and not self.supports_asset_class(asset_class):
            raise ProviderError(
                f"Provider {self.name} does not support asset class {asset_class}. "
                f"Supported: {self.supported_asset_classes}"
            )

    def normalize_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize OHLCV DataFrame to standard format.

        Args:
            df: Raw OHLCV DataFrame

        Returns:
            Normalized DataFrame with standard column names and types
        """
        # Make a copy
        df = df.copy()

        # Normalize column names to lowercase
        df.columns = df.columns.str.lower()

        # Ensure required columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ProviderError(f"Missing required column: {col}")

        # Ensure timestamp index
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df.index = pd.to_datetime(df["timestamp"])
                df.drop("timestamp", axis=1, inplace=True)
            elif "date" in df.columns:
                df.index = pd.to_datetime(df["date"])
                df.drop("date", axis=1, inplace=True)
            else:
                raise ProviderError("No timestamp column found")

        # Ensure datetime index
        df.index = pd.to_datetime(df.index)

        # Sort by timestamp
        df.sort_index(inplace=True)

        # Ensure numeric types
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove rows with NaN values
        df.dropna(inplace=True)

        # Select only required columns in standard order
        df = df[required_cols]

        self.logger.debug(f"Normalized {len(df)} rows of OHLCV data")

        return df

    def __str__(self) -> str:
        """String representation of the provider."""
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        """Repr of the provider."""
        return self.__str__()
