"""
Time series splitting utilities for TradeAI
"""

from typing import Tuple, List, Optional

import pandas as pd
import numpy as np

from tradeAI.utils.logger import get_logger

logger = get_logger(__name__)


class TimeSeriesSplitter:
    """Split time series data for training, validation, and testing."""

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        shuffle: bool = False,
    ):
        """
        Initialize splitter.

        Args:
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            shuffle: Whether to shuffle (not recommended for time series)
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Ratios must sum to 1.0")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.shuffle = shuffle

    def split(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame into train, validation, and test sets.

        Args:
            df: DataFrame to split
            target_col: Target column name (if separating features and target)

        Returns:
            Tuple of (train, val, test) DataFrames
        """
        n = len(df)

        if self.shuffle:
            logger.warning("Shuffling time series data - temporal order will be lost")
            df = df.sample(frac=1.0, random_state=42)

        # Calculate split points
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        # Split
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        logger.info(
            f"Split data: train={len(train_df)} ({len(train_df)/n*100:.1f}%), "
            f"val={len(val_df)} ({len(val_df)/n*100:.1f}%), "
            f"test={len(test_df)} ({len(test_df)/n*100:.1f}%)"
        )

        return train_df, val_df, test_df

    def split_xy(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
    ) -> Tuple[
        Tuple[pd.DataFrame, pd.Series],
        Tuple[pd.DataFrame, pd.Series],
        Tuple[pd.DataFrame, pd.Series],
    ]:
        """
        Split DataFrame into train, val, test with separated features and targets.

        Args:
            df: DataFrame to split
            target_col: Target column name
            feature_cols: Feature column names (None = all except target)

        Returns:
            ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        train_df, val_df, test_df = self.split(df)

        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]

        X_val = val_df[feature_cols]
        y_val = val_df[target_col]

        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def walk_forward_split(
        self,
        df: pd.DataFrame,
        train_size: int,
        test_size: int,
        step_size: Optional[int] = None,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create walk-forward splits for time series.

        Args:
            df: DataFrame to split
            train_size: Size of training window
            test_size: Size of test window
            step_size: Step size for rolling (None = test_size)

        Returns:
            List of (train, test) tuples
        """
        if step_size is None:
            step_size = test_size

        splits = []
        n = len(df)

        for i in range(0, n - train_size - test_size + 1, step_size):
            train_start = i
            train_end = i + train_size
            test_end = train_end + test_size

            if test_end > n:
                break

            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[train_end:test_end]

            splits.append((train_df, test_df))

        logger.info(f"Created {len(splits)} walk-forward splits")

        return splits

    def expanding_window_split(
        self,
        df: pd.DataFrame,
        min_train_size: int,
        test_size: int,
        step_size: Optional[int] = None,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create expanding window splits (growing training set).

        Args:
            df: DataFrame to split
            min_train_size: Minimum training size
            test_size: Size of test window
            step_size: Step size for expansion (None = test_size)

        Returns:
            List of (train, test) tuples
        """
        if step_size is None:
            step_size = test_size

        splits = []
        n = len(df)

        current_train_end = min_train_size

        while current_train_end + test_size <= n:
            train_df = df.iloc[:current_train_end]
            test_df = df.iloc[current_train_end:current_train_end + test_size]

            splits.append((train_df, test_df))

            current_train_end += step_size

        logger.info(f"Created {len(splits)} expanding window splits")

        return splits


def create_sequences(
    df: pd.DataFrame,
    sequence_length: int,
    target_col: Optional[str] = None,
    stride: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Create sequences for LSTM/RNN models.

    Args:
        df: DataFrame with features
        sequence_length: Length of each sequence
        target_col: Target column name
        stride: Stride for sequence creation

    Returns:
        (X, y) arrays where X is (samples, sequence_length, features)
    """
    if target_col:
        feature_cols = [col for col in df.columns if col != target_col]
        X_data = df[feature_cols].values
        y_data = df[target_col].values
    else:
        X_data = df.values
        y_data = None

    X_sequences = []
    y_sequences = [] if y_data is not None else None

    for i in range(0, len(df) - sequence_length + 1, stride):
        X_sequences.append(X_data[i:i + sequence_length])

        if y_data is not None:
            # Target is the value at the end of the sequence
            y_sequences.append(y_data[i + sequence_length - 1])

    X = np.array(X_sequences)
    y = np.array(y_sequences) if y_sequences else None

    logger.debug(f"Created {len(X)} sequences of length {sequence_length}")

    return X, y
