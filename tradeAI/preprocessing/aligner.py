"""
Multi-timeframe alignment for TradeAI
"""

from typing import Dict, List

import pandas as pd

from tradeAI.utils.logger import get_logger
from tradeAI.utils.helpers import timeframe_to_seconds

logger = get_logger(__name__)


class MultiTimeframeAligner:
    """Align data from multiple timeframes for multi-timeframe analysis."""

    def __init__(self, base_timeframe: str):
        """
        Initialize aligner.

        Args:
            base_timeframe: Base timeframe for alignment (e.g., '15m')
        """
        self.base_timeframe = base_timeframe
        self.base_seconds = timeframe_to_seconds(base_timeframe)

    def align(
        self,
        data_dict: Dict[str, pd.DataFrame],
        method: str = "ffill",
    ) -> pd.DataFrame:
        """
        Align multiple timeframes to base timeframe.

        Args:
            data_dict: Dictionary mapping timeframe to DataFrame
            method: Alignment method ('ffill', 'interpolate')

        Returns:
            Aligned DataFrame with multi-timeframe data
        """
        if self.base_timeframe not in data_dict:
            raise ValueError(f"Base timeframe {self.base_timeframe} not in data_dict")

        base_df = data_dict[self.base_timeframe].copy()
        base_index = base_df.index

        aligned_df = base_df.copy()

        # Add data from other timeframes
        for tf, df in data_dict.items():
            if tf == self.base_timeframe:
                continue

            tf_seconds = timeframe_to_seconds(tf)

            if tf_seconds < self.base_seconds:
                # Higher frequency - aggregate to base
                logger.debug(f"Aggregating {tf} to {self.base_timeframe}")
                resampled = self._aggregate_to_base(df, self.base_timeframe)
            else:
                # Lower frequency - downsample
                logger.debug(f"Downsampling {tf} to {self.base_timeframe}")
                resampled = self._downsample_to_base(df, base_index, method)

            # Add with timeframe prefix
            for col in resampled.columns:
                aligned_df[f"{tf}_{col}"] = resampled[col]

        logger.info(
            f"Aligned {len(data_dict)} timeframes to {self.base_timeframe}, "
            f"resulting in {len(aligned_df.columns)} columns"
        )

        return aligned_df

    def _aggregate_to_base(self, df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """Aggregate higher frequency data to lower frequency."""
        # Resample OHLCV data
        resampled = df.resample(target_tf).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })

        return resampled

    def _downsample_to_base(
        self,
        df: pd.DataFrame,
        target_index: pd.DatetimeIndex,
        method: str,
    ) -> pd.DataFrame:
        """Downsample lower frequency data to higher frequency."""
        # Reindex to target and forward fill
        if method == "ffill":
            resampled = df.reindex(target_index, method="ffill")
        elif method == "interpolate":
            resampled = df.reindex(target_index).interpolate(method="time")
        else:
            raise ValueError(f"Unknown method: {method}")

        return resampled

    def create_lagged_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int],
    ) -> pd.DataFrame:
        """
        Create lagged features for time series.

        Args:
            df: DataFrame
            columns: Columns to lag
            lags: List of lag periods

        Returns:
            DataFrame with lagged features
        """
        result = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            for lag in lags:
                result[f"{col}_lag{lag}"] = df[col].shift(lag)

        # Drop rows with NaN from lagging
        max_lag = max(lags)
        result = result.iloc[max_lag:]

        logger.debug(
            f"Created {len(columns) * len(lags)} lagged features, "
            f"removed {max_lag} rows"
        )

        return result
