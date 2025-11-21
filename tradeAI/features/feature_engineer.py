"""
Feature engineering for trading data
"""

from typing import List, Optional

import pandas as pd
import numpy as np

from tradeAI.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Calculate technical indicators and features for trading."""

    def __init__(self):
        """Initialize FeatureEngineer."""
        pass

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical features to DataFrame.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with added features
        """
        df = df.copy()

        # Price-based features
        df = self.add_price_features(df)

        # Moving averages
        df = self.add_moving_averages(df)

        # Momentum indicators
        df = self.add_momentum_indicators(df)

        # Volatility indicators
        df = self.add_volatility_indicators(df)

        # Volume indicators
        df = self.add_volume_indicators(df)

        logger.info(f"Added features, total columns: {len(df.columns)}")

        return df

    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features."""
        # Returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Price range
        df["range"] = df["high"] - df["low"]
        df["range_pct"] = df["range"] / df["close"]

        # Body and wicks (candlestick)
        df["body"] = df["close"] - df["open"]
        df["body_pct"] = df["body"] / df["close"]
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

        # Typical price
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3

        return df

    def add_moving_averages(
        self,
        df: pd.DataFrame,
        periods: List[int] = [5, 10, 20, 50, 200],
    ) -> pd.DataFrame:
        """Add moving averages."""
        for period in periods:
            # Simple Moving Average
            df[f"sma_{period}"] = df["close"].rolling(window=period).mean()

            # Exponential Moving Average
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

            # Distance from MA
            df[f"dist_sma_{period}"] = (df["close"] - df[f"sma_{period}"]) / df[f"sma_{period}"]

        return df

    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        # RSI
        df = self._add_rsi(df, period=14)

        # MACD
        df = self._add_macd(df)

        # Stochastic
        df = self._add_stochastic(df, period=14)

        # Rate of Change
        for period in [9, 14, 21]:
            df[f"roc_{period}"] = df["close"].pct_change(periods=period) * 100

        return df

    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        # Bollinger Bands
        df = self._add_bollinger_bands(df, period=20, std=2)

        # ATR (Average True Range)
        df = self._add_atr(df, period=14)

        # Historical Volatility
        for period in [10, 20, 30]:
            df[f"volatility_{period}"] = df["returns"].rolling(window=period).std() * np.sqrt(252)

        return df

    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators."""
        # Volume MA
        df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

        # OBV (On-Balance Volume)
        df = self._add_obv(df)

        # VWAP (Volume Weighted Average Price)
        df = self._add_vwap(df)

        return df

    def _add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index."""
        delta = df["close"].diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        return df

    def _add_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """Add MACD indicator."""
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()

        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        return df

    def _add_stochastic(
        self,
        df: pd.DataFrame,
        period: int = 14,
        smooth_k: int = 3,
    ) -> pd.DataFrame:
        """Add Stochastic Oscillator."""
        low_min = df["low"].rolling(window=period).min()
        high_max = df["high"].rolling(window=period).max()

        df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min)
        df["stoch_d"] = df["stoch_k"].rolling(window=smooth_k).mean()

        return df

    def _add_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std: float = 2.0,
    ) -> pd.DataFrame:
        """Add Bollinger Bands."""
        sma = df["close"].rolling(window=period).mean()
        rolling_std = df["close"].rolling(window=period).std()

        df["bb_middle"] = sma
        df["bb_upper"] = sma + (rolling_std * std)
        df["bb_lower"] = sma - (rolling_std * std)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        return df

    def _add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range."""
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        df[f"atr_{period}"] = true_range.rolling(window=period).mean()

        return df

    def _add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add On-Balance Volume."""
        obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
        df["obv"] = obv

        return df

    def _add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Volume Weighted Average Price."""
        df["vwap"] = (df["typical_price"] * df["volume"]).cumsum() / df["volume"].cumsum()

        return df

    def add_lagged_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int] = [1, 2, 3, 5],
    ) -> pd.DataFrame:
        """Add lagged features."""
        for col in columns:
            if col not in df.columns:
                continue

            for lag in lags:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

        return df

    def add_rolling_statistics(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int] = [5, 10, 20],
    ) -> pd.DataFrame:
        """Add rolling statistics."""
        for col in columns:
            if col not in df.columns:
                continue

            for window in windows:
                df[f"{col}_mean_{window}"] = df[col].rolling(window=window).mean()
                df[f"{col}_std_{window}"] = df[col].rolling(window=window).std()
                df[f"{col}_min_{window}"] = df[col].rolling(window=window).min()
                df[f"{col}_max_{window}"] = df[col].rolling(window=window).max()

        return df

    def select_features(
        self,
        df: pd.DataFrame,
        method: str = "variance",
        threshold: float = 0.01,
    ) -> pd.DataFrame:
        """Remove low-variance or highly correlated features."""
        if method == "variance":
            # Remove low variance features
            variances = df.var()
            keep_cols = variances[variances > threshold].index.tolist()
            df = df[keep_cols]

            logger.info(f"Removed {len(df.columns) - len(keep_cols)} low-variance features")

        elif method == "correlation":
            # Remove highly correlated features
            corr_matrix = df.corr().abs()

            # Select upper triangle of correlation matrix
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            # Find features with correlation greater than threshold
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

            df = df.drop(columns=to_drop)

            logger.info(f"Removed {len(to_drop)} highly correlated features")

        return df
