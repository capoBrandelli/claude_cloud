"""
Reversal labeling for supervised learning
"""

from typing import Literal, Optional

import pandas as pd
import numpy as np

from tradeAI.labeling.extrema_detector import ExtremaDetector
from tradeAI.utils.logger import get_logger

logger = get_logger(__name__)


class ReversalLabeler:
    """Label reversal points for supervised learning."""

    def __init__(
        self,
        method: Literal["binary", "multiclass", "regression"] = "multiclass",
        lookahead_window: int = 10,
        threshold: float = 0.02,
        num_classes: int = 5,
    ):
        """
        Initialize reversal labeler.

        Args:
            method: Labeling method
            lookahead_window: Bars to look ahead for reversal confirmation
            threshold: Minimum price change to consider a reversal
            num_classes: Number of classes for multiclass (3 or 5)
        """
        self.method = method
        self.lookahead_window = lookahead_window
        self.threshold = threshold
        self.num_classes = num_classes

    def label(self, df: pd.DataFrame) -> pd.Series:
        """
        Create labels for reversal prediction.

        Args:
            df: OHLCV DataFrame with features

        Returns:
            Series with labels
        """
        if self.method == "binary":
            return self._label_binary(df)
        elif self.method == "multiclass":
            return self._label_multiclass(df)
        elif self.method == "regression":
            return self._label_regression(df)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _label_binary(self, df: pd.DataFrame) -> pd.Series:
        """
        Binary labels: 0 = no reversal, 1 = reversal.

        A reversal is defined as a significant price change in the opposite
        direction within the lookahead window.
        """
        labels = pd.Series(0, index=df.index)

        # Detect extrema
        detector = ExtremaDetector(method="zigzag", threshold=self.threshold)
        peaks, troughs = detector.detect(df)

        # Label peaks and troughs as reversals
        labels[peaks | troughs] = 1

        n_reversals = labels.sum()
        logger.info(
            f"Binary labeling: {n_reversals} reversals ({n_reversals/len(labels)*100:.1f}%)"
        )

        return labels

    def _label_multiclass(self, df: pd.DataFrame) -> pd.Series:
        """
        Multiclass labels based on future price movement.

        Classes:
        - 0: Strong bearish (price down > 2 * threshold)
        - 1: Weak bearish (price down < threshold)
        - 2: Neutral (price change < threshold)
        - 3: Weak bullish (price up < threshold)
        - 4: Strong bullish (price up > 2 * threshold)
        """
        # Calculate future returns
        future_returns = df["close"].pct_change(periods=self.lookahead_window).shift(-self.lookahead_window)

        # Initialize labels with neutral class
        if self.num_classes == 3:
            # 0 = bearish, 1 = neutral, 2 = bullish
            labels = pd.Series(1, index=df.index)

            labels[future_returns < -self.threshold] = 0  # Bearish
            labels[future_returns > self.threshold] = 2   # Bullish

        elif self.num_classes == 5:
            # 0 = strong bearish, 1 = weak bearish, 2 = neutral, 3 = weak bullish, 4 = strong bullish
            labels = pd.Series(2, index=df.index)

            labels[future_returns < -2 * self.threshold] = 0  # Strong bearish
            labels[(future_returns >= -2 * self.threshold) & (future_returns < -self.threshold)] = 1  # Weak bearish
            labels[(future_returns > self.threshold) & (future_returns <= 2 * self.threshold)] = 3  # Weak bullish
            labels[future_returns > 2 * self.threshold] = 4  # Strong bullish

        else:
            raise ValueError(f"num_classes must be 3 or 5, got {self.num_classes}")

        # Remove labels for last bars (no future data)
        labels.iloc[-self.lookahead_window:] = -1

        # Log class distribution
        class_counts = labels[labels != -1].value_counts().sort_index()
        logger.info(f"Multiclass labeling ({self.num_classes} classes):")
        for cls, count in class_counts.items():
            logger.info(f"  Class {cls}: {count} ({count/len(labels)*100:.1f}%)")

        return labels

    def _label_regression(self, df: pd.DataFrame) -> pd.Series:
        """
        Regression labels: future return value.

        Returns the actual price change percentage over the lookahead window.
        """
        # Calculate future returns
        labels = df["close"].pct_change(periods=self.lookahead_window).shift(-self.lookahead_window)

        # Remove labels for last bars
        labels.iloc[-self.lookahead_window:] = np.nan

        logger.info(
            f"Regression labeling: mean={labels.mean():.4f}, "
            f"std={labels.std():.4f}"
        )

        return labels

    def add_reversal_probability(
        self,
        df: pd.DataFrame,
        window: int = 20,
    ) -> pd.DataFrame:
        """
        Add probability-based reversal indicators.

        Args:
            df: OHLCV DataFrame
            window: Rolling window for probability calculation

        Returns:
            DataFrame with reversal probability columns
        """
        df = df.copy()

        # Detect extrema
        detector = ExtremaDetector(method="zigzag", threshold=self.threshold)
        peaks, troughs = detector.detect(df)

        # Calculate historical reversal probability
        df["peak_prob"] = peaks.astype(float).rolling(window=window).mean()
        df["trough_prob"] = troughs.astype(float).rolling(window=window).mean()
        df["reversal_prob"] = (peaks | troughs).astype(float).rolling(window=window).mean()

        # Distance from last extrema
        peak_indices = np.where(peaks)[0]
        trough_indices = np.where(troughs)[0]

        df["bars_since_peak"] = 0
        df["bars_since_trough"] = 0

        for i in range(len(df)):
            if len(peak_indices) > 0:
                last_peak_idx = peak_indices[peak_indices < i]
                if len(last_peak_idx) > 0:
                    df.iloc[i, df.columns.get_loc("bars_since_peak")] = i - last_peak_idx[-1]

            if len(trough_indices) > 0:
                last_trough_idx = trough_indices[trough_indices < i]
                if len(last_trough_idx) > 0:
                    df.iloc[i, df.columns.get_loc("bars_since_trough")] = i - last_trough_idx[-1]

        return df
