"""
Data cleaning utilities for TradeAI
"""

from typing import Optional, Literal

import pandas as pd
import numpy as np

from tradeAI.utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """Clean and prepare OHLCV data for analysis."""

    def __init__(
        self,
        handle_missing: Literal["ffill", "bfill", "interpolate", "drop"] = "ffill",
        remove_outliers: bool = True,
        outlier_method: Literal["iqr", "zscore"] = "iqr",
        outlier_threshold: float = 3.0,
    ):
        """
        Initialize DataCleaner.

        Args:
            handle_missing: Method to handle missing values
            remove_outliers: Whether to remove outliers
            outlier_method: Method for outlier detection
            outlier_threshold: Threshold for outlier detection
        """
        self.handle_missing = handle_missing
        self.remove_outliers = remove_outliers
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean OHLCV dataframe.

        Args:
            df: OHLCV dataframe

        Returns:
            Cleaned dataframe
        """
        df = df.copy()

        initial_rows = len(df)

        # Handle missing values
        df = self._handle_missing_values(df)

        # Remove duplicates
        df = self._remove_duplicates(df)

        # Remove outliers
        if self.remove_outliers:
            df = self._remove_outliers(df)

        # Ensure monotonic index
        df = df.sort_index()

        final_rows = len(df)
        removed = initial_rows - final_rows

        if removed > 0:
            logger.info(f"Cleaned data: removed {removed} rows ({removed/initial_rows*100:.1f}%)")

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in dataframe."""
        missing_count = df.isnull().sum().sum()

        if missing_count == 0:
            return df

        logger.debug(f"Found {missing_count} missing values")

        if self.handle_missing == "ffill":
            df = df.fillna(method="ffill")
        elif self.handle_missing == "bfill":
            df = df.fillna(method="bfill")
        elif self.handle_missing == "interpolate":
            df = df.interpolate(method="linear")
        elif self.handle_missing == "drop":
            df = df.dropna()

        # Drop any remaining NaN values
        remaining_nan = df.isnull().sum().sum()
        if remaining_nan > 0:
            logger.warning(f"Dropping {remaining_nan} remaining NaN values")
            df = df.dropna()

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate timestamps."""
        duplicates = df.index.duplicated(keep="first")
        n_duplicates = duplicates.sum()

        if n_duplicates > 0:
            logger.warning(f"Removing {n_duplicates} duplicate timestamps")
            df = df[~duplicates]

        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from price data."""
        if self.outlier_method == "iqr":
            df = self._remove_outliers_iqr(df)
        elif self.outlier_method == "zscore":
            df = self._remove_outliers_zscore(df)

        return df

    def _remove_outliers_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        price_cols = ["open", "high", "low", "close"]

        # Calculate returns instead of absolute prices
        returns = df[price_cols].pct_change()

        for col in price_cols:
            Q1 = returns[col].quantile(0.25)
            Q3 = returns[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - self.outlier_threshold * IQR
            upper_bound = Q3 + self.outlier_threshold * IQR

            outliers = (returns[col] < lower_bound) | (returns[col] > upper_bound)
            n_outliers = outliers.sum()

            if n_outliers > 0:
                logger.debug(f"Found {n_outliers} outliers in {col} using IQR")
                df = df[~outliers]

        return df

    def _remove_outliers_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using z-score method."""
        price_cols = ["open", "high", "low", "close"]

        # Calculate returns
        returns = df[price_cols].pct_change()

        for col in price_cols:
            z_scores = np.abs((returns[col] - returns[col].mean()) / returns[col].std())
            outliers = z_scores > self.outlier_threshold
            n_outliers = outliers.sum()

            if n_outliers > 0:
                logger.debug(f"Found {n_outliers} outliers in {col} using z-score")
                df = df[~outliers]

        return df

    def fill_gaps(
        self,
        df: pd.DataFrame,
        freq: Optional[str] = None,
        max_gap: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fill gaps in time series data.

        Args:
            df: OHLCV dataframe
            freq: Frequency for reindexing (e.g., '1h', '1d')
            max_gap: Maximum gap size to fill (in number of periods)

        Returns:
            Dataframe with filled gaps
        """
        if freq is None:
            # Infer frequency
            freq = pd.infer_freq(df.index)
            if freq is None:
                logger.warning("Could not infer frequency, skipping gap filling")
                return df

        logger.debug(f"Filling gaps with frequency: {freq}")

        # Create complete date range
        full_range = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=freq,
        )

        # Reindex to fill gaps
        df_filled = df.reindex(full_range)

        # Limit gap filling
        if max_gap is not None:
            # Only fill gaps smaller than max_gap
            mask = df_filled.isnull().all(axis=1)
            gap_groups = (mask != mask.shift()).cumsum()
            gap_sizes = mask.groupby(gap_groups).transform("sum")

            # Don't fill large gaps
            large_gaps = gap_sizes > max_gap
            df_filled.loc[large_gaps] = np.nan

        # Forward fill prices
        df_filled = df_filled.fillna(method="ffill")

        gaps_filled = len(df_filled) - len(df)
        if gaps_filled > 0:
            logger.info(f"Filled {gaps_filled} gaps in time series")

        return df_filled
