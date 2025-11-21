"""
Direction labeling for ensemble models

This labeler predicts the direction of price movement (up/down) without
considering the current trend context. It's designed to work alongside
reversal and continuation models in an ensemble.

IMPORTANT: This labeler uses FUTURE data for labels (acceptable for supervised learning).
Features must use ONLY past data to prevent forward-looking bias.
"""

from typing import Optional, Literal
import pandas as pd
import numpy as np

from tradeAI.utils.logger import get_logger

logger = get_logger(__name__)


class DirectionLabeler:
    """
    Label directional price movement for supervised learning.

    Creates labels based on future price direction, independent of
    current trend or reversal context. This is the simplest labeling
    strategy and serves as a baseline in the ensemble.
    """

    def __init__(
        self,
        method: Literal["binary", "multiclass"] = "binary",
        lookahead: int = 10,
        threshold: float = 0.01,
        num_classes: int = 3,
        price_type: Literal["close", "high_low", "ohlc"] = "close",
    ):
        """
        Initialize direction labeler.

        Args:
            method: Labeling method ('binary' or 'multiclass')
            lookahead: Forward-looking window (uses FUTURE data - OK for labels)
            threshold: Minimum return to be considered significant (1% default)
            num_classes: Number of classes for multiclass (3 or 5)
            price_type: What price to use for direction
                - 'close': Use close-to-close returns
                - 'high_low': Use high/low for more aggressive signals
                - 'ohlc': Use average of OHLC
        """
        self.method = method
        self.lookahead = lookahead
        self.threshold = threshold
        self.num_classes = num_classes
        self.price_type = price_type

        logger.info(
            f"DirectionLabeler initialized: method={method}, "
            f"lookahead={lookahead}, threshold={threshold}"
        )

    def label(self, df: pd.DataFrame) -> pd.Series:
        """
        Create direction labels based on future price movement.

        Process:
        1. Calculate future returns using specified price type
        2. Create labels based on return magnitude and direction

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series with direction labels
        """
        if self.method == "binary":
            return self._label_binary(df)
        elif self.method == "multiclass":
            return self._label_multiclass(df)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _get_price_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Get price series based on price_type setting.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Price series to use for direction calculation
        """
        if self.price_type == "close":
            return df["close"]
        elif self.price_type == "high_low":
            # Use high for upward moves, low for downward moves
            # This is more aggressive and catches larger swings
            return (df["high"] + df["low"]) / 2
        elif self.price_type == "ohlc":
            # Average price
            return (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        else:
            raise ValueError(f"Unknown price_type: {self.price_type}")

    def _label_binary(self, df: pd.DataFrame) -> pd.Series:
        """
        Binary direction labeling.

        Labels:
        - 0: Down (price decreases)
        - 1: Up (price increases)
        - -1: Neutral (insufficient movement)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series with binary labels
        """
        labels = pd.Series(-1, index=df.index)

        # Get price series
        price = self._get_price_series(df)

        # Calculate future returns (FUTURE data - OK for labels)
        future_return = price.pct_change(self.lookahead).shift(-self.lookahead)

        # Create labels
        labels[future_return > self.threshold] = 1  # Up
        labels[future_return < -self.threshold] = 0  # Down
        # Neutral remains -1

        # Handle NaN at the end (no future data available)
        labels = labels.fillna(-1)

        # Statistics
        value_counts = labels.value_counts()
        logger.info(f"Binary direction labels created:")
        logger.info(f"  Down (0): {value_counts.get(0, 0)} ({value_counts.get(0, 0)/len(df)*100:.1f}%)")
        logger.info(f"  Up (1): {value_counts.get(1, 0)} ({value_counts.get(1, 0)/len(df)*100:.1f}%)")
        logger.info(f"  Neutral (-1): {value_counts.get(-1, 0)} ({value_counts.get(-1, 0)/len(df)*100:.1f}%)")

        return labels

    def _label_multiclass(self, df: pd.DataFrame) -> pd.Series:
        """
        Multiclass direction labeling.

        For 3 classes:
        - 0: Down
        - 1: Neutral
        - 2: Up
        - -1: No label (insufficient data)

        For 5 classes:
        - 0: Strong down
        - 1: Weak down
        - 2: Neutral
        - 3: Weak up
        - 4: Strong up
        - -1: No label (insufficient data)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series with multiclass labels
        """
        labels = pd.Series(-1, index=df.index)

        # Get price series
        price = self._get_price_series(df)

        # Calculate future returns (FUTURE data - OK for labels)
        future_return = price.pct_change(self.lookahead).shift(-self.lookahead)

        if self.num_classes == 3:
            # 3-class system: Down / Neutral / Up
            labels[future_return < -self.threshold] = 0  # Down
            labels[abs(future_return) <= self.threshold] = 1  # Neutral
            labels[future_return > self.threshold] = 2  # Up

        elif self.num_classes == 5:
            # 5-class system with finer granularity
            strong_threshold = self.threshold * 2.0
            weak_threshold = self.threshold

            # Strong down
            labels[future_return < -strong_threshold] = 0

            # Weak down
            labels[(future_return >= -strong_threshold) & (future_return < -weak_threshold)] = 1

            # Neutral
            labels[abs(future_return) < weak_threshold] = 2

            # Weak up
            labels[(future_return > weak_threshold) & (future_return <= strong_threshold)] = 3

            # Strong up
            labels[future_return > strong_threshold] = 4

        # Handle NaN at the end
        labels = labels.fillna(-1)

        # Statistics
        value_counts = labels.value_counts().sort_index()
        logger.info(f"Multiclass direction labels ({self.num_classes} classes) created:")
        for cls, count in value_counts.items():
            if cls != -1:
                logger.info(f"  Class {cls}: {count} ({count/len(df)*100:.1f}%)")
        no_label_count = value_counts.get(-1, 0)
        logger.info(f"  No label (-1): {no_label_count} ({no_label_count/len(df)*100:.1f}%)")

        return labels

    def label_with_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create labels with confidence scores.

        Confidence is based on:
        1. Magnitude of future return (larger = higher confidence)
        2. Consistency of direction in lookahead window
        3. Recent volatility (higher volatility = lower confidence)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with labels and confidence scores
        """
        result = pd.DataFrame(index=df.index)

        # Get labels
        result["direction_label"] = self.label(df)

        # Get price series
        price = self._get_price_series(df)

        # Future return (FUTURE data - OK for labels)
        future_return = price.pct_change(self.lookahead).shift(-self.lookahead)
        result["future_return"] = future_return

        # Confidence based on magnitude (normalized to 0-1)
        result["return_magnitude"] = abs(future_return)
        max_return = result["return_magnitude"].quantile(0.95)  # Cap at 95th percentile
        result["magnitude_confidence"] = (result["return_magnitude"] / max_return).clip(0, 1)

        # Directional consistency (what % of days move in the predicted direction)
        daily_returns = price.pct_change()

        def calc_consistency(i):
            """Calculate what % of next N days move in predicted direction."""
            if i >= len(daily_returns) - self.lookahead or result["direction_label"].iloc[i] == -1:
                return 0.5  # Neutral

            label = result["direction_label"].iloc[i]
            future_rets = daily_returns.iloc[i+1:i+1+self.lookahead]

            if len(future_rets) == 0:
                return 0.5

            if label == 1:  # Predict up
                consistency = (future_rets > 0).sum() / len(future_rets)
            elif label == 0:  # Predict down
                consistency = (future_rets < 0).sum() / len(future_rets)
            else:  # Neutral
                consistency = 0.5

            return consistency

        result["directional_consistency"] = [calc_consistency(i) for i in range(len(df))]

        # Volatility-based confidence (lower volatility = higher confidence)
        volatility = daily_returns.rolling(self.lookahead).std()
        max_vol = volatility.quantile(0.95)
        result["volatility_confidence"] = 1 - (volatility / max_vol).clip(0, 1)

        # Combined confidence score
        result["confidence"] = (
            result["magnitude_confidence"] * 0.4 +
            result["directional_consistency"] * 0.3 +
            result["volatility_confidence"] * 0.3
        )

        logger.info(f"Created direction labels with confidence scores")
        logger.info(f"  Mean confidence: {result['confidence'].mean():.3f}")
        logger.info(f"  High confidence (>0.7): {(result['confidence'] > 0.7).sum()} samples")

        return result

    def label_multi_timeframe(
        self,
        df: pd.DataFrame,
        lookaheads: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Create direction labels for multiple timeframes.

        This allows the ensemble to capture short-term, medium-term,
        and long-term directional predictions.

        Args:
            df: DataFrame with OHLCV data
            lookaheads: List of lookahead windows (default: [5, 10, 20])

        Returns:
            DataFrame with labels for each timeframe
        """
        if lookaheads is None:
            lookaheads = [5, 10, 20]

        result = pd.DataFrame(index=df.index)
        price = self._get_price_series(df)

        for lookahead in lookaheads:
            # Calculate future return for this timeframe
            future_return = price.pct_change(lookahead).shift(-lookahead)

            # Create label
            label_col = f"direction_{lookahead}d"
            labels = pd.Series(-1, index=df.index)

            if self.method == "binary":
                labels[future_return > self.threshold] = 1
                labels[future_return < -self.threshold] = 0
            else:
                # Use same logic as _label_multiclass
                if self.num_classes == 3:
                    labels[future_return < -self.threshold] = 0
                    labels[abs(future_return) <= self.threshold] = 1
                    labels[future_return > self.threshold] = 2

            result[label_col] = labels.fillna(-1)

        logger.info(f"Created multi-timeframe direction labels for {len(lookaheads)} timeframes")

        return result
