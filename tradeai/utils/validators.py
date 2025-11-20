"""
Data validation utilities for TradeAI
"""

from typing import List, Optional

import pandas as pd

from tradeai.utils.logger import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Raised when data validation fails."""

    pass


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1,
    allow_nan: bool = False,
) -> None:
    """
    Validate a pandas DataFrame.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        allow_nan: Whether to allow NaN values

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"Expected DataFrame, got {type(df)}")

    if len(df) < min_rows:
        raise ValidationError(f"DataFrame has {len(df)} rows, expected at least {min_rows}")

    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValidationError(f"Missing required columns: {missing_cols}")

    if not allow_nan and df.isnull().any().any():
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0].to_dict()
        raise ValidationError(f"DataFrame contains NaN values: {null_cols}")


def validate_ohlcv(df: pd.DataFrame, strict: bool = True) -> None:
    """
    Validate OHLCV (Open, High, Low, Close, Volume) data.

    Args:
        df: OHLCV DataFrame to validate
        strict: If True, performs strict validation

    Raises:
        ValidationError: If validation fails
    """
    required_columns = ["open", "high", "low", "close", "volume"]

    # Check for required columns (case-insensitive)
    df_columns_lower = [col.lower() for col in df.columns]
    missing_cols = [col for col in required_columns if col not in df_columns_lower]

    if missing_cols:
        raise ValidationError(f"Missing required OHLCV columns: {missing_cols}")

    # Normalize column names to lowercase
    column_mapping = {col: col.lower() for col in df.columns if col.lower() in required_columns}
    df_normalized = df.rename(columns=column_mapping)

    # Validate data types
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df_normalized[col]):
            raise ValidationError(f"Column '{col}' must be numeric")

    if strict:
        # Validate high >= low
        invalid_hl = df_normalized["high"] < df_normalized["low"]
        if invalid_hl.any():
            count = invalid_hl.sum()
            raise ValidationError(f"Found {count} rows where high < low")

        # Validate high >= open, close
        invalid_h_open = df_normalized["high"] < df_normalized["open"]
        invalid_h_close = df_normalized["high"] < df_normalized["close"]
        if invalid_h_open.any() or invalid_h_close.any():
            count = (invalid_h_open | invalid_h_close).sum()
            raise ValidationError(f"Found {count} rows where high < open or high < close")

        # Validate low <= open, close
        invalid_l_open = df_normalized["low"] > df_normalized["open"]
        invalid_l_close = df_normalized["low"] > df_normalized["close"]
        if invalid_l_open.any() or invalid_l_close.any():
            count = (invalid_l_open | invalid_l_close).sum()
            raise ValidationError(f"Found {count} rows where low > open or low > close")

        # Validate volume >= 0
        invalid_volume = df_normalized["volume"] < 0
        if invalid_volume.any():
            count = invalid_volume.sum()
            raise ValidationError(f"Found {count} rows with negative volume")

        # Validate positive prices
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            invalid_prices = df_normalized[col] <= 0
            if invalid_prices.any():
                count = invalid_prices.sum()
                raise ValidationError(f"Found {count} rows with non-positive {col} prices")

    logger.debug("OHLCV validation passed")


def validate_timeframe(timeframe: str) -> bool:
    """
    Validate a timeframe string.

    Args:
        timeframe: Timeframe string (e.g., '1m', '5m', '1h', '1d')

    Returns:
        True if valid

    Raises:
        ValidationError: If timeframe is invalid
    """
    valid_timeframes = {
        # Minutes
        "1m",
        "2m",
        "3m",
        "5m",
        "15m",
        "30m",
        "45m",
        # Hours
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        # Days
        "1d",
        "3d",
        # Weeks
        "1w",
        "1wk",
        # Months
        "1M",
        "1mo",
        "3mo",
        # Also support some common variations
        "1min",
        "5min",
        "15min",
        "30min",
        "60min",
        "1hour",
        "1day",
        "daily",
        "weekly",
        "monthly",
    }

    if timeframe not in valid_timeframes:
        raise ValidationError(
            f"Invalid timeframe '{timeframe}'. Must be one of: {valid_timeframes}"
        )

    return True


def validate_symbol(symbol: str) -> bool:
    """
    Validate a trading symbol.

    Args:
        symbol: Trading symbol

    Returns:
        True if valid

    Raises:
        ValidationError: If symbol is invalid
    """
    if not symbol:
        raise ValidationError("Symbol cannot be empty")

    if not isinstance(symbol, str):
        raise ValidationError(f"Symbol must be a string, got {type(symbol)}")

    # Basic validation - alphanumeric and common separators
    if not all(c.isalnum() or c in ["_", "-", ".", "/"] for c in symbol):
        raise ValidationError(f"Symbol '{symbol}' contains invalid characters")

    return True


def validate_date_range(
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
) -> None:
    """
    Validate a date range.

    Args:
        start_date: Start date
        end_date: End date

    Raises:
        ValidationError: If date range is invalid
    """
    if start_date is not None and end_date is not None:
        if start_date >= end_date:
            raise ValidationError(f"start_date ({start_date}) must be before end_date ({end_date})")

    if start_date is not None and start_date > pd.Timestamp.now():
        raise ValidationError(f"start_date ({start_date}) cannot be in the future")


def validate_numeric_range(
    value: float,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    name: str = "value",
) -> None:
    """
    Validate that a numeric value is within a specified range.

    Args:
        value: Value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        name: Name of the value (for error messages)

    Raises:
        ValidationError: If value is out of range
    """
    if min_value is not None and value < min_value:
        raise ValidationError(f"{name} ({value}) must be >= {min_value}")

    if max_value is not None and value > max_value:
        raise ValidationError(f"{name} ({value}) must be <= {max_value}")
