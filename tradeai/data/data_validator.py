"""
Data validation and consistency checking for TradeAI
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from tradeai.utils.logger import get_logger
from tradeai.utils.validators import validate_ohlcv

logger = get_logger(__name__)


class DataValidator:
    """Validate OHLCV data and check consistency across providers."""

    def __init__(
        self,
        price_tolerance: float = 0.01,
        volume_tolerance: float = 0.05,
        min_data_points: int = 100,
    ):
        """
        Initialize DataValidator.

        Args:
            price_tolerance: Maximum allowed price difference (as fraction)
            volume_tolerance: Maximum allowed volume difference (as fraction)
            min_data_points: Minimum required data points
        """
        self.price_tolerance = price_tolerance
        self.volume_tolerance = volume_tolerance
        self.min_data_points = min_data_points

    def validate(self, df: pd.DataFrame, strict: bool = True) -> Tuple[bool, List[str]]:
        """
        Validate OHLCV data.

        Args:
            df: OHLCV DataFrame to validate
            strict: If True, performs strict validation

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check minimum data points
        if len(df) < self.min_data_points:
            issues.append(
                f"Insufficient data points: {len(df)} < {self.min_data_points}"
            )

        # Validate OHLCV structure
        try:
            validate_ohlcv(df, strict=strict)
        except Exception as e:
            issues.append(f"OHLCV validation failed: {str(e)}")

        # Check for gaps in data
        gaps = self._check_gaps(df)
        if gaps:
            issues.append(f"Found {len(gaps)} gaps in data")

        # Check for outliers
        outliers = self._check_outliers(df)
        if outliers:
            issues.append(f"Found {sum(outliers.values())} outliers")

        # Check for suspicious patterns
        suspicious = self._check_suspicious_patterns(df)
        if suspicious:
            issues.extend(suspicious)

        is_valid = len(issues) == 0

        if is_valid:
            logger.info("Data validation passed")
        else:
            logger.warning(f"Data validation found {len(issues)} issues: {issues}")

        return is_valid, issues

    def compare_providers(
        self,
        data_dict: Dict[str, pd.DataFrame],
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Compare data from multiple providers and check consistency.

        Args:
            data_dict: Dictionary mapping provider names to DataFrames

        Returns:
            Tuple of (is_consistent, comparison_results)
        """
        if len(data_dict) < 2:
            logger.warning("Need at least 2 providers for comparison")
            return True, {}

        # Filter out None/empty DataFrames
        valid_data = {k: v for k, v in data_dict.items() if v is not None and not v.empty}

        if len(valid_data) < 2:
            logger.warning("Need at least 2 valid providers for comparison")
            return True, {}

        # Find common time range
        common_index = self._get_common_index(list(valid_data.values()))

        if len(common_index) == 0:
            logger.warning("No common timestamps found across providers")
            return False, {"error": "No common timestamps"}

        # Align data to common index
        aligned_data = {}
        for provider, df in valid_data.items():
            aligned_data[provider] = df.reindex(common_index)

        # Compare prices
        price_comparison = self._compare_prices(aligned_data)

        # Compare volumes
        volume_comparison = self._compare_volumes(aligned_data)

        # Overall consistency
        is_consistent = (
            price_comparison["is_consistent"]
            and volume_comparison["is_consistent"]
        )

        results = {
            "common_data_points": len(common_index),
            "price_comparison": price_comparison,
            "volume_comparison": volume_comparison,
            "is_consistent": is_consistent,
        }

        if is_consistent:
            logger.info("Data is consistent across providers")
        else:
            logger.warning("Data inconsistencies detected across providers")

        return is_consistent, results

    def _get_common_index(self, dataframes: List[pd.DataFrame]) -> pd.DatetimeIndex:
        """Get common timestamps across all DataFrames."""
        if not dataframes:
            return pd.DatetimeIndex([])

        # Start with first dataframe's index
        common_index = dataframes[0].index

        # Intersect with other indices
        for df in dataframes[1:]:
            common_index = common_index.intersection(df.index)

        return common_index

    def _compare_prices(self, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """Compare prices across providers."""
        providers = list(data_dict.keys())
        base_provider = providers[0]
        base_df = data_dict[base_provider]

        max_diff = 0.0
        differences = []

        for provider in providers[1:]:
            df = data_dict[provider]

            # Compare close prices
            price_diff = np.abs(df["close"] - base_df["close"]) / base_df["close"]
            max_provider_diff = price_diff.max()
            mean_provider_diff = price_diff.mean()

            max_diff = max(max_diff, max_provider_diff)

            differences.append({
                "providers": f"{base_provider} vs {provider}",
                "max_diff": float(max_provider_diff),
                "mean_diff": float(mean_provider_diff),
                "exceeds_tolerance": max_provider_diff > self.price_tolerance,
            })

        is_consistent = max_diff <= self.price_tolerance

        return {
            "is_consistent": is_consistent,
            "max_difference": float(max_diff),
            "tolerance": self.price_tolerance,
            "differences": differences,
        }

    def _compare_volumes(self, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """Compare volumes across providers."""
        providers = list(data_dict.keys())
        base_provider = providers[0]
        base_df = data_dict[base_provider]

        max_diff = 0.0
        differences = []

        for provider in providers[1:]:
            df = data_dict[provider]

            # Compare volumes (avoid division by zero)
            volume_diff = np.abs(df["volume"] - base_df["volume"]) / (
                base_df["volume"] + 1e-9
            )
            max_provider_diff = volume_diff.max()
            mean_provider_diff = volume_diff.mean()

            max_diff = max(max_diff, max_provider_diff)

            differences.append({
                "providers": f"{base_provider} vs {provider}",
                "max_diff": float(max_provider_diff),
                "mean_diff": float(mean_provider_diff),
                "exceeds_tolerance": max_provider_diff > self.volume_tolerance,
            })

        is_consistent = max_diff <= self.volume_tolerance

        return {
            "is_consistent": is_consistent,
            "max_difference": float(max_diff),
            "tolerance": self.volume_tolerance,
            "differences": differences,
        }

    def _check_gaps(self, df: pd.DataFrame, max_gap_hours: int = 24) -> List:
        """Check for gaps in time series data."""
        if len(df) < 2:
            return []

        # Calculate time differences
        time_diffs = df.index.to_series().diff()

        # Find unusually large gaps
        median_diff = time_diffs.median()
        threshold = max(median_diff * 3, pd.Timedelta(hours=max_gap_hours))

        gaps = time_diffs[time_diffs > threshold]

        return list(gaps.index)

    def _check_outliers(
        self,
        df: pd.DataFrame,
        z_threshold: float = 3.0,
    ) -> Dict[str, int]:
        """Check for outliers using z-score method."""
        outliers = {}

        for col in ["open", "high", "low", "close", "volume"]:
            # Calculate returns for prices
            if col != "volume":
                values = df[col].pct_change().dropna()
            else:
                values = df[col]

            # Calculate z-scores
            z_scores = np.abs((values - values.mean()) / values.std())

            # Count outliers
            n_outliers = (z_scores > z_threshold).sum()
            if n_outliers > 0:
                outliers[col] = int(n_outliers)

        return outliers

    def _check_suspicious_patterns(self, df: pd.DataFrame) -> List[str]:
        """Check for suspicious patterns in data."""
        issues = []

        # Check for repeated values
        for col in ["open", "high", "low", "close"]:
            # Count consecutive repeated values
            repeated = (df[col] == df[col].shift(1))
            max_consecutive = repeated.groupby((repeated != repeated.shift()).cumsum()).sum().max()

            if max_consecutive > 10:
                issues.append(
                    f"{col} has {max_consecutive} consecutive repeated values"
                )

        # Check for zero volume
        zero_volume = (df["volume"] == 0).sum()
        if zero_volume > 0:
            issues.append(f"Found {zero_volume} bars with zero volume")

        # Check for impossible price movements (>50% in one bar)
        price_changes = df["close"].pct_change().abs()
        extreme_changes = (price_changes > 0.5).sum()
        if extreme_changes > 0:
            issues.append(f"Found {extreme_changes} bars with >50% price change")

        return issues
