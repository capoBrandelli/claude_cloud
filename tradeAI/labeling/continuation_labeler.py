"""
Continuation labeling for ensemble models

This labeler identifies when an existing trend is likely to continue.
It's designed to work alongside reversal and direction models in an ensemble.

IMPORTANT: This labeler uses FUTURE data for labels (acceptable for supervised learning).
Features must use ONLY past data to prevent forward-looking bias.
"""

from typing import Optional, Literal
import pandas as pd
import numpy as np

from tradeAI.utils.logger import get_logger

logger = get_logger(__name__)


class ContinuationLabeler:
    """
    Label trend continuation patterns for supervised learning.

    Creates labels based on whether an identified trend continues or breaks.
    This complements the reversal labeler by identifying stable trending periods.
    """

    def __init__(
        self,
        method: Literal["binary", "multiclass"] = "binary",
        lookback: int = 20,
        lookahead: int = 10,
        trend_threshold: float = 0.02,
        continuation_threshold: float = 0.015,
        num_classes: int = 3,
    ):
        """
        Initialize continuation labeler.

        Args:
            method: Labeling method ('binary' or 'multiclass')
            lookback: Period to identify current trend (uses PAST data only)
            lookahead: Period to check if trend continues (uses FUTURE data - OK for labels)
            trend_threshold: Minimum return to identify a trend (2% default)
            continuation_threshold: Minimum return to confirm continuation (1.5% default)
            num_classes: Number of classes for multiclass (3 or 5)
        """
        self.method = method
        self.lookback = lookback
        self.lookahead = lookahead
        self.trend_threshold = trend_threshold
        self.continuation_threshold = continuation_threshold
        self.num_classes = num_classes

        logger.info(
            f"ContinuationLabeler initialized: method={method}, "
            f"lookback={lookback}, lookahead={lookahead}"
        )

    def label(self, df: pd.DataFrame) -> pd.Series:
        """
        Create continuation labels.

        Process:
        1. Identify current trend using PAST data (lookback window)
        2. Check if trend continues using FUTURE data (lookahead window)
        3. Create labels based on continuation strength

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series with continuation labels
        """
        if self.method == "binary":
            return self._label_binary(df)
        elif self.method == "multiclass":
            return self._label_multiclass(df)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _label_binary(self, df: pd.DataFrame) -> pd.Series:
        """
        Binary continuation labeling.

        Labels:
        - 0: Trend breaks or reverses
        - 1: Trend continues
        - -1: No clear trend (neutral)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series with binary labels
        """
        labels = pd.Series(-1, index=df.index)  # Default: no clear trend

        # Calculate past trend (PAST data only)
        past_return = df["close"].pct_change(self.lookback)

        # Calculate future return (FUTURE data - OK for labels)
        future_return = df["close"].pct_change(self.lookahead).shift(-self.lookahead)

        for i in range(self.lookback, len(df) - self.lookahead):
            past_ret = past_return.iloc[i]
            future_ret = future_return.iloc[i]

            # Skip if no clear past trend
            if abs(past_ret) < self.trend_threshold:
                labels.iloc[i] = -1
                continue

            # Check if trend continues
            if past_ret > self.trend_threshold:  # Uptrend
                if future_ret > self.continuation_threshold:
                    labels.iloc[i] = 1  # Uptrend continues
                else:
                    labels.iloc[i] = 0  # Uptrend breaks

            elif past_ret < -self.trend_threshold:  # Downtrend
                if future_ret < -self.continuation_threshold:
                    labels.iloc[i] = 1  # Downtrend continues
                else:
                    labels.iloc[i] = 0  # Downtrend breaks

        # Statistics
        value_counts = labels.value_counts()
        logger.info(f"Binary continuation labels created:")
        logger.info(f"  Break (0): {value_counts.get(0, 0)} ({value_counts.get(0, 0)/len(df)*100:.1f}%)")
        logger.info(f"  Continue (1): {value_counts.get(1, 0)} ({value_counts.get(1, 0)/len(df)*100:.1f}%)")
        logger.info(f"  No trend (-1): {value_counts.get(-1, 0)} ({value_counts.get(-1, 0)/len(df)*100:.1f}%)")

        return labels

    def _label_multiclass(self, df: pd.DataFrame) -> pd.Series:
        """
        Multiclass continuation labeling.

        For 3 classes:
        - 0: Strong break (trend reverses significantly)
        - 1: Weak break or consolidation
        - 2: Trend continues
        - -1: No clear trend

        For 5 classes:
        - 0: Strong reversal
        - 1: Weak reversal
        - 2: Consolidation/neutral
        - 3: Weak continuation
        - 4: Strong continuation
        - -1: No clear trend

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series with multiclass labels
        """
        labels = pd.Series(-1, index=df.index)  # Default: no clear trend

        # Calculate past trend (PAST data only)
        past_return = df["close"].pct_change(self.lookback)

        # Calculate future return (FUTURE data - OK for labels)
        future_return = df["close"].pct_change(self.lookahead).shift(-self.lookahead)

        if self.num_classes == 3:
            # 3-class system
            for i in range(self.lookback, len(df) - self.lookahead):
                past_ret = past_return.iloc[i]
                future_ret = future_return.iloc[i]

                # Skip if no clear past trend
                if abs(past_ret) < self.trend_threshold:
                    labels.iloc[i] = -1
                    continue

                # Uptrend
                if past_ret > self.trend_threshold:
                    if future_ret < -self.continuation_threshold:
                        labels.iloc[i] = 0  # Strong break (reverses)
                    elif abs(future_ret) < self.continuation_threshold:
                        labels.iloc[i] = 1  # Weak break (consolidates)
                    else:
                        labels.iloc[i] = 2  # Continues

                # Downtrend
                elif past_ret < -self.trend_threshold:
                    if future_ret > self.continuation_threshold:
                        labels.iloc[i] = 0  # Strong break (reverses)
                    elif abs(future_ret) < self.continuation_threshold:
                        labels.iloc[i] = 1  # Weak break (consolidates)
                    else:
                        labels.iloc[i] = 2  # Continues

        elif self.num_classes == 5:
            # 5-class system with finer granularity
            strong_threshold = self.continuation_threshold * 1.5
            weak_threshold = self.continuation_threshold * 0.5

            for i in range(self.lookback, len(df) - self.lookahead):
                past_ret = past_return.iloc[i]
                future_ret = future_return.iloc[i]

                # Skip if no clear past trend
                if abs(past_ret) < self.trend_threshold:
                    labels.iloc[i] = -1
                    continue

                # Determine continuation strength
                if past_ret > self.trend_threshold:  # Uptrend
                    if future_ret < -strong_threshold:
                        labels.iloc[i] = 0  # Strong reversal
                    elif future_ret < -weak_threshold:
                        labels.iloc[i] = 1  # Weak reversal
                    elif abs(future_ret) < weak_threshold:
                        labels.iloc[i] = 2  # Consolidation
                    elif future_ret > strong_threshold:
                        labels.iloc[i] = 4  # Strong continuation
                    else:
                        labels.iloc[i] = 3  # Weak continuation

                elif past_ret < -self.trend_threshold:  # Downtrend
                    if future_ret > strong_threshold:
                        labels.iloc[i] = 0  # Strong reversal
                    elif future_ret > weak_threshold:
                        labels.iloc[i] = 1  # Weak reversal
                    elif abs(future_ret) < weak_threshold:
                        labels.iloc[i] = 2  # Consolidation
                    elif future_ret < -strong_threshold:
                        labels.iloc[i] = 4  # Strong continuation
                    else:
                        labels.iloc[i] = 3  # Weak continuation

        # Statistics
        value_counts = labels.value_counts().sort_index()
        logger.info(f"Multiclass continuation labels ({self.num_classes} classes) created:")
        for cls, count in value_counts.items():
            if cls != -1:
                logger.info(f"  Class {cls}: {count} ({count/len(df)*100:.1f}%)")
        logger.info(f"  No trend (-1): {value_counts.get(-1, 0)} ({value_counts.get(-1, 0)/len(df)*100:.1f}%)")

        return labels

    def label_with_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create labels with additional trend strength features.

        Returns labels plus trend strength indicators that can be used
        as additional features (all calculated from PAST data only).

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with labels and trend strength features
        """
        result = pd.DataFrame(index=df.index)

        # Get labels
        result["continuation_label"] = self.label(df)

        # Add trend strength features (PAST data only)
        result["past_trend_return"] = df["close"].pct_change(self.lookback)
        result["past_trend_abs"] = result["past_trend_return"].abs()

        # Trend consistency (what % of days moved in trend direction)
        daily_returns = df["close"].pct_change()
        result["trend_consistency"] = (
            daily_returns.rolling(self.lookback)
            .apply(lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5)
        )

        # Volatility during trend (PAST data only)
        result["trend_volatility"] = daily_returns.rolling(self.lookback).std()

        logger.info(f"Created continuation labels with {len(result.columns)} features")

        return result


def identify_trend_direction(
    df: pd.DataFrame,
    lookback: int = 20,
    method: Literal["price", "ma", "adx"] = "price",
) -> pd.Series:
    """
    Identify current trend direction using PAST data only.

    This is a helper function that can be used for feature engineering
    or for more sophisticated continuation labeling.

    Args:
        df: DataFrame with OHLCV data
        lookback: Lookback period
        method: Method to identify trend
            - 'price': Based on price change
            - 'ma': Based on moving average position
            - 'adx': Based on ADX indicator (requires implementation)

    Returns:
        Series with trend direction (1: up, -1: down, 0: neutral)
    """
    if method == "price":
        # Simple price-based trend
        returns = df["close"].pct_change(lookback)
        trend = pd.Series(0, index=df.index)
        trend[returns > 0.02] = 1  # Uptrend
        trend[returns < -0.02] = -1  # Downtrend
        return trend

    elif method == "ma":
        # Moving average based
        ma_fast = df["close"].rolling(lookback // 2).mean()
        ma_slow = df["close"].rolling(lookback).mean()

        trend = pd.Series(0, index=df.index)
        trend[(df["close"] > ma_fast) & (ma_fast > ma_slow)] = 1  # Uptrend
        trend[(df["close"] < ma_fast) & (ma_fast < ma_slow)] = -1  # Downtrend
        return trend

    else:
        raise ValueError(f"Unknown trend identification method: {method}")
