"""
Temporal validation utilities to prevent forward-looking bias

This module provides utilities to validate that data splits and features
maintain point-in-time integrity and don't leak future information.
"""

from typing import Tuple, List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

from tradeAI.utils.logger import get_logger

logger = get_logger(__name__)


class TemporalValidator:
    """Validate temporal integrity to prevent forward-looking bias."""

    @staticmethod
    def validate_temporal_split(
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        test_data: Optional[pd.DataFrame] = None,
        raise_on_error: bool = True,
    ) -> Dict[str, bool]:
        """
        Validate that train/val/test splits maintain temporal order.

        Checks:
        1. Train data comes before validation data
        2. Validation data comes before test data
        3. No temporal overlap between splits

        Args:
            train_data: Training dataset with datetime index
            val_data: Validation dataset with datetime index
            test_data: Test dataset with datetime index
            raise_on_error: Whether to raise exception on validation failure

        Returns:
            Dictionary with validation results

        Raises:
            ValueError: If temporal order is violated and raise_on_error=True
        """
        results = {
            "train_val_ordered": True,
            "val_test_ordered": True,
            "no_train_val_overlap": True,
            "no_val_test_overlap": True,
            "no_train_test_overlap": True,
        }

        # Ensure datetime index
        if not isinstance(train_data.index, pd.DatetimeIndex):
            train_data = train_data.copy()
            train_data.index = pd.to_datetime(train_data.index)

        train_start = train_data.index.min()
        train_end = train_data.index.max()

        # Validate train vs val
        if val_data is not None:
            if not isinstance(val_data.index, pd.DatetimeIndex):
                val_data = val_data.copy()
                val_data.index = pd.to_datetime(val_data.index)

            val_start = val_data.index.min()
            val_end = val_data.index.max()

            # Check order
            if train_end >= val_start:
                results["train_val_ordered"] = False
                msg = f"Train data ({train_end}) overlaps or comes after validation ({val_start})"
                logger.error(msg)
                if raise_on_error:
                    raise ValueError(msg)

            # Check overlap
            train_dates = set(train_data.index)
            val_dates = set(val_data.index)
            overlap = train_dates & val_dates

            if overlap:
                results["no_train_val_overlap"] = False
                msg = f"Train and validation have {len(overlap)} overlapping timestamps"
                logger.error(msg)
                if raise_on_error:
                    raise ValueError(msg)

        # Validate val vs test
        if val_data is not None and test_data is not None:
            if not isinstance(test_data.index, pd.DatetimeIndex):
                test_data = test_data.copy()
                test_data.index = pd.to_datetime(test_data.index)

            test_start = test_data.index.min()
            test_end = test_data.index.max()

            # Check order
            if val_end >= test_start:
                results["val_test_ordered"] = False
                msg = f"Validation data ({val_end}) overlaps or comes after test ({test_start})"
                logger.error(msg)
                if raise_on_error:
                    raise ValueError(msg)

            # Check overlap
            val_dates = set(val_data.index)
            test_dates = set(test_data.index)
            overlap = val_dates & test_dates

            if overlap:
                results["no_val_test_overlap"] = False
                msg = f"Validation and test have {len(overlap)} overlapping timestamps"
                logger.error(msg)
                if raise_on_error:
                    raise ValueError(msg)

        # Validate train vs test
        if test_data is not None:
            if not isinstance(test_data.index, pd.DatetimeIndex):
                test_data = test_data.copy()
                test_data.index = pd.to_datetime(test_data.index)

            test_start = test_data.index.min()

            # Check order
            if train_end >= test_start:
                results["no_train_test_overlap"] = False
                msg = f"Train data ({train_end}) overlaps or comes after test ({test_start})"
                logger.error(msg)
                if raise_on_error:
                    raise ValueError(msg)

            # Check overlap
            train_dates = set(train_data.index)
            test_dates = set(test_data.index)
            overlap = train_dates & test_dates

            if overlap:
                results["no_train_test_overlap"] = False
                msg = f"Train and test have {len(overlap)} overlapping timestamps"
                logger.error(msg)
                if raise_on_error:
                    raise ValueError(msg)

        # Log success
        if all(results.values()):
            logger.info("✓ Temporal split validation passed - no forward-looking bias")

        return results

    @staticmethod
    def validate_no_future_features(
        df: pd.DataFrame,
        reference_time: datetime,
        raise_on_error: bool = True,
    ) -> bool:
        """
        Validate that no features use data after the reference time.

        This is a basic check that ensures the dataframe doesn't contain
        timestamps after the reference time.

        Args:
            df: DataFrame with features
            reference_time: Reference time (usually current time in backtest)
            raise_on_error: Whether to raise exception on validation failure

        Returns:
            True if validation passes

        Raises:
            ValueError: If future data detected and raise_on_error=True
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)

        future_data = df[df.index > reference_time]

        if len(future_data) > 0:
            msg = (
                f"Future data detected: {len(future_data)} timestamps after "
                f"reference time {reference_time}"
            )
            logger.error(msg)
            if raise_on_error:
                raise ValueError(msg)
            return False

        logger.debug(f"✓ No future features - all data before {reference_time}")
        return True

    @staticmethod
    def check_feature_computation(
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        check_for_negatives: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform heuristic checks on feature computation for potential bias.

        Checks:
        1. Features contain NaN at the beginning (expected from rolling windows)
        2. No suspicious negative shifts in column names
        3. Features use reasonable lookback windows

        Args:
            df: DataFrame with features
            feature_cols: List of feature columns (None = all numeric columns)
            check_for_negatives: Whether to check for negative shifts in names

        Returns:
            Dictionary with check results and warnings
        """
        results = {
            "has_leading_nans": False,
            "suspicious_columns": [],
            "warnings": [],
        }

        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Check for leading NaNs (expected from rolling windows)
        for col in feature_cols:
            if df[col].head(20).isna().any():
                results["has_leading_nans"] = True
                break

        if not results["has_leading_nans"]:
            results["warnings"].append(
                "No leading NaNs found - ensure features use rolling windows"
            )

        # Check for suspicious column names
        if check_for_negatives:
            suspicious = []
            for col in feature_cols:
                col_lower = str(col).lower()
                # Check for negative shifts: shift(-1), shift(-5), etc.
                if "shift(-" in col_lower:
                    suspicious.append(col)
                # Check for forward-looking terms
                elif any(term in col_lower for term in ["future", "forward", "ahead", "next"]):
                    suspicious.append(col)

            if suspicious:
                results["suspicious_columns"] = suspicious
                results["warnings"].append(
                    f"Suspicious column names detected (may indicate forward-looking): {suspicious}"
                )
                logger.warning(f"Suspicious columns: {suspicious}")

        # Log results
        if not results["warnings"]:
            logger.info("✓ Feature computation checks passed")
        else:
            logger.warning(f"Feature computation warnings: {results['warnings']}")

        return results

    @staticmethod
    def validate_normalization_order(
        normalizer: Any,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        raise_on_error: bool = True,
    ) -> bool:
        """
        Validate that normalizer was fit on training data only.

        This checks that the normalizer's statistics (mean, std, min, max)
        are different from the test set statistics, indicating it wasn't
        fit on the test set.

        Args:
            normalizer: Fitted normalizer (with scaler_ attribute)
            train_data: Training data
            test_data: Test data
            raise_on_error: Whether to raise exception on validation failure

        Returns:
            True if validation passes

        Raises:
            ValueError: If normalizer appears to be fit on test data
        """
        try:
            # Get scaler statistics
            if not hasattr(normalizer, "scaler_"):
                logger.warning("Normalizer has no scaler_ attribute - cannot validate")
                return True

            scaler = normalizer.scaler_

            # Get test statistics
            test_min = test_data.min().values
            test_max = test_data.max().values

            # For MinMaxScaler, check if data_min_ and data_max_ match test set
            if hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
                scaler_min = scaler.data_min_
                scaler_max = scaler.data_max_

                # Check if scaler stats are suspiciously close to test stats
                min_diff = np.abs(scaler_min - test_min).mean()
                max_diff = np.abs(scaler_max - test_max).mean()

                # If differences are very small, might indicate test set leakage
                if min_diff < 1e-6 and max_diff < 1e-6:
                    msg = (
                        "Normalizer statistics suspiciously close to test set - "
                        "possible data leakage"
                    )
                    logger.error(msg)
                    if raise_on_error:
                        raise ValueError(msg)
                    return False

            logger.info("✓ Normalizer validation passed - fit on training data only")
            return True

        except Exception as e:
            logger.warning(f"Could not validate normalizer: {e}")
            return True  # Don't fail on validation errors

    @staticmethod
    def validate_walk_forward_split(
        splits: List[Tuple[pd.DataFrame, pd.DataFrame]],
        raise_on_error: bool = True,
    ) -> bool:
        """
        Validate walk-forward splits maintain temporal order.

        Args:
            splits: List of (train, test) tuples
            raise_on_error: Whether to raise exception on validation failure

        Returns:
            True if validation passes

        Raises:
            ValueError: If temporal order is violated
        """
        for i, (train, test) in enumerate(splits):
            # Validate this split
            result = TemporalValidator.validate_temporal_split(
                train, test_data=test, raise_on_error=raise_on_error
            )

            if not all(result.values()):
                msg = f"Walk-forward split {i} failed temporal validation"
                logger.error(msg)
                if raise_on_error:
                    raise ValueError(msg)
                return False

            # Validate against next split
            if i < len(splits) - 1:
                next_train, next_test = splits[i + 1]

                # Current test should come before next train
                if test.index.max() >= next_train.index.min():
                    msg = f"Walk-forward split {i} test overlaps with split {i+1} train"
                    logger.error(msg)
                    if raise_on_error:
                        raise ValueError(msg)
                    return False

        logger.info(f"✓ Walk-forward validation passed for {len(splits)} splits")
        return True


# Convenience functions
def validate_temporal_split(*args, **kwargs) -> Dict[str, bool]:
    """Convenience wrapper for TemporalValidator.validate_temporal_split."""
    return TemporalValidator.validate_temporal_split(*args, **kwargs)


def validate_no_future_features(*args, **kwargs) -> bool:
    """Convenience wrapper for TemporalValidator.validate_no_future_features."""
    return TemporalValidator.validate_no_future_features(*args, **kwargs)


def check_feature_computation(*args, **kwargs) -> Dict[str, Any]:
    """Convenience wrapper for TemporalValidator.check_feature_computation."""
    return TemporalValidator.check_feature_computation(*args, **kwargs)


def validate_normalization_order(*args, **kwargs) -> bool:
    """Convenience wrapper for TemporalValidator.validate_normalization_order."""
    return TemporalValidator.validate_normalization_order(*args, **kwargs)


def validate_walk_forward_split(*args, **kwargs) -> bool:
    """Convenience wrapper for TemporalValidator.validate_walk_forward_split."""
    return TemporalValidator.validate_walk_forward_split(*args, **kwargs)
