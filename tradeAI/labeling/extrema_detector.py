"""
Extrema detection for identifying reversals
"""

from typing import Literal, Tuple

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

from tradeAI.utils.logger import get_logger

logger = get_logger(__name__)


class ExtremaDetector:
    """Detect local maxima and minima in price data."""

    def __init__(
        self,
        method: Literal["zigzag", "rolling", "scipy"] = "zigzag",
        threshold: float = 0.05,
        order: int = 5,
    ):
        """
        Initialize extrema detector.

        Args:
            method: Detection method
            threshold: Threshold for zigzag (percentage)
            order: Order for scipy method (number of points on each side)
        """
        self.method = method
        self.threshold = threshold
        self.order = order

    def detect(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Detect extrema in price data.

        Args:
            df: OHLCV DataFrame

        Returns:
            Tuple of (peaks, troughs) as boolean Series
        """
        if self.method == "zigzag":
            return self._detect_zigzag(df)
        elif self.method == "rolling":
            return self._detect_rolling(df)
        elif self.method == "scipy":
            return self._detect_scipy(df)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _detect_zigzag(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Detect extrema using zigzag algorithm.

        The zigzag algorithm identifies significant price swings by filtering
        out price movements smaller than the threshold.
        """
        prices = df["close"].values
        n = len(prices)

        peaks = pd.Series(False, index=df.index)
        troughs = pd.Series(False, index=df.index)

        # Initialize
        last_extreme_idx = 0
        last_extreme_price = prices[0]
        trend = 0  # 0 = unknown, 1 = up, -1 = down

        for i in range(1, n):
            price = prices[i]
            change = (price - last_extreme_price) / last_extreme_price

            if trend == 0:
                # Establish initial trend
                if change > self.threshold:
                    trend = 1
                    troughs.iloc[last_extreme_idx] = True
                    last_extreme_idx = i
                    last_extreme_price = price
                elif change < -self.threshold:
                    trend = -1
                    peaks.iloc[last_extreme_idx] = True
                    last_extreme_idx = i
                    last_extreme_price = price

            elif trend == 1:
                # Uptrend - look for peak
                if price > last_extreme_price:
                    last_extreme_idx = i
                    last_extreme_price = price
                elif change < -self.threshold:
                    # Found peak, switch to downtrend
                    peaks.iloc[last_extreme_idx] = True
                    trend = -1
                    last_extreme_idx = i
                    last_extreme_price = price

            elif trend == -1:
                # Downtrend - look for trough
                if price < last_extreme_price:
                    last_extreme_idx = i
                    last_extreme_price = price
                elif change > self.threshold:
                    # Found trough, switch to uptrend
                    troughs.iloc[last_extreme_idx] = True
                    trend = 1
                    last_extreme_idx = i
                    last_extreme_price = price

        n_peaks = peaks.sum()
        n_troughs = troughs.sum()

        logger.info(
            f"Zigzag detected {n_peaks} peaks and {n_troughs} troughs "
            f"(threshold={self.threshold*100:.1f}%)"
        )

        return peaks, troughs

    def _detect_rolling(
        self,
        df: pd.DataFrame,
        window: int = 10,
    ) -> Tuple[pd.Series, pd.Series]:
        """Detect extrema using rolling window."""
        # Peak if price is highest in window
        peaks = df["close"] == df["close"].rolling(
            window=window, center=True
        ).max()

        # Trough if price is lowest in window
        troughs = df["close"] == df["close"].rolling(
            window=window, center=True
        ).min()

        # Remove edges
        peaks.iloc[:window//2] = False
        peaks.iloc[-window//2:] = False
        troughs.iloc[:window//2] = False
        troughs.iloc[-window//2:] = False

        logger.info(
            f"Rolling window detected {peaks.sum()} peaks and {troughs.sum()} troughs"
        )

        return peaks, troughs

    def _detect_scipy(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Detect extrema using scipy's argrelextrema."""
        prices = df["close"].values

        # Find local maxima
        peak_indices = argrelextrema(prices, np.greater, order=self.order)[0]
        peaks = pd.Series(False, index=df.index)
        peaks.iloc[peak_indices] = True

        # Find local minima
        trough_indices = argrelextrema(prices, np.less, order=self.order)[0]
        troughs = pd.Series(False, index=df.index)
        troughs.iloc[trough_indices] = True

        logger.info(
            f"Scipy detected {len(peak_indices)} peaks and {len(trough_indices)} troughs"
        )

        return peaks, troughs
