"""
Helper utilities for TradeAI
"""

import pickle
from pathlib import Path
from typing import Any, Dict

from tradeai.utils.logger import get_logger

logger = get_logger(__name__)


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent.parent


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to directory

    Returns:
        Path to directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pickle(obj: Any, file_path: Path) -> None:
    """
    Save an object to a pickle file.

    Args:
        obj: Object to save
        file_path: Path to save file
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)

    with open(file_path, "wb") as f:
        pickle.dump(obj, f)

    logger.debug(f"Saved object to {file_path}")


def load_pickle(file_path: Path) -> Any:
    """
    Load an object from a pickle file.

    Args:
        file_path: Path to pickle file

    Returns:
        Loaded object
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as f:
        obj = pickle.load(f)

    logger.debug(f"Loaded object from {file_path}")
    return obj


def timeframe_to_seconds(timeframe: str) -> int:
    """
    Convert a timeframe string to seconds.

    Args:
        timeframe: Timeframe string (e.g., '1m', '5m', '1h', '1d')

    Returns:
        Number of seconds in the timeframe
    """
    # Mapping of timeframe units to seconds
    unit_seconds: Dict[str, int] = {
        "m": 60,  # minute
        "min": 60,
        "h": 3600,  # hour
        "hour": 3600,
        "d": 86400,  # day
        "day": 86400,
        "w": 604800,  # week
        "wk": 604800,
        "week": 604800,
        "M": 2592000,  # month (30 days)
        "mo": 2592000,
        "month": 2592000,
    }

    # Special cases
    if timeframe == "daily":
        return 86400
    elif timeframe == "weekly":
        return 604800
    elif timeframe == "monthly":
        return 2592000

    # Parse number and unit
    i = 0
    while i < len(timeframe) and (timeframe[i].isdigit() or timeframe[i] == "."):
        i += 1

    if i == 0:
        raise ValueError(f"Invalid timeframe format: {timeframe}")

    number = float(timeframe[:i])
    unit = timeframe[i:]

    if unit not in unit_seconds:
        raise ValueError(f"Unknown timeframe unit: {unit}")

    return int(number * unit_seconds[unit])


def timeframe_to_minutes(timeframe: str) -> int:
    """
    Convert a timeframe string to minutes.

    Args:
        timeframe: Timeframe string

    Returns:
        Number of minutes in the timeframe
    """
    return timeframe_to_seconds(timeframe) // 60


def format_number(
    value: float,
    decimals: int = 2,
    prefix: str = "",
    suffix: str = "",
) -> str:
    """
    Format a number with specified decimals and prefix/suffix.

    Args:
        value: Number to format
        decimals: Number of decimal places
        prefix: Prefix string (e.g., '$')
        suffix: Suffix string (e.g., '%')

    Returns:
        Formatted string
    """
    formatted = f"{value:,.{decimals}f}"
    return f"{prefix}{formatted}{suffix}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a value as a percentage.

    Args:
        value: Value to format (e.g., 0.05 for 5%)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return format_number(value * 100, decimals=decimals, suffix="%")


def calculate_returns(prices: Any) -> Any:
    """
    Calculate returns from prices.

    Args:
        prices: Price series or array

    Returns:
        Returns series or array
    """
    import pandas as pd
    import numpy as np

    if isinstance(prices, pd.Series):
        return prices.pct_change()
    elif isinstance(prices, np.ndarray):
        return np.diff(prices) / prices[:-1]
    else:
        raise TypeError(f"Unsupported type: {type(prices)}")


def calculate_log_returns(prices: Any) -> Any:
    """
    Calculate log returns from prices.

    Args:
        prices: Price series or array

    Returns:
        Log returns series or array
    """
    import pandas as pd
    import numpy as np

    if isinstance(prices, pd.Series):
        return np.log(prices / prices.shift(1))
    elif isinstance(prices, np.ndarray):
        return np.log(prices[1:] / prices[:-1])
    else:
        raise TypeError(f"Unsupported type: {type(prices)}")
