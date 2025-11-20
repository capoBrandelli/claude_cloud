"""
Data aggregator for combining data from multiple providers
"""

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from tradeai.data.base_provider import BaseDataProvider
from tradeai.data.cache_manager import CacheManager
from tradeai.data.data_validator import DataValidator
from tradeai.utils.logger import get_logger

logger = get_logger(__name__)


class DataAggregator:
    """
    Aggregate data from multiple providers with validation and caching.
    """

    def __init__(
        self,
        providers: List[BaseDataProvider],
        cache_manager: Optional[CacheManager] = None,
        validator: Optional[DataValidator] = None,
        validate_consistency: bool = True,
    ):
        """
        Initialize DataAggregator.

        Args:
            providers: List of data providers (in priority order)
            cache_manager: Cache manager instance
            validator: Data validator instance
            validate_consistency: Whether to validate data consistency across providers
        """
        self.providers = providers
        self.cache_manager = cache_manager or CacheManager()
        self.validator = validator or DataValidator()
        self.validate_consistency = validate_consistency

        logger.info(f"DataAggregator initialized with {len(providers)} providers")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
        use_cache: bool = True,
        fallback: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data with provider fallback and caching.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            limit: Maximum number of bars
            use_cache: Whether to use cache
            fallback: Whether to fallback to other providers on failure

        Returns:
            OHLCV DataFrame

        Raises:
            Exception: If all providers fail
        """
        # Try cache first
        if use_cache and self.cache_manager.enabled:
            for provider in self.providers:
                cached_data = self.cache_manager.get(
                    provider.name,
                    symbol,
                    timeframe,
                    start_date,
                    end_date,
                )
                if cached_data is not None:
                    logger.info(f"Using cached data from {provider.name}")
                    return cached_data

        # Try each provider in order
        last_error = None

        for provider in self.providers:
            if not provider.is_available():
                logger.debug(f"Provider {provider.name} not available, skipping")
                continue

            try:
                logger.info(f"Fetching from {provider.name}...")

                df = provider.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    limit=limit,
                )

                # Validate data
                is_valid, issues = self.validator.validate(df, strict=False)

                if not is_valid:
                    logger.warning(
                        f"Data from {provider.name} has validation issues: {issues}"
                    )
                    if not fallback:
                        raise Exception(f"Data validation failed: {issues}")
                    # Try next provider
                    continue

                # Cache the data
                if use_cache:
                    self.cache_manager.set(
                        df,
                        provider.name,
                        symbol,
                        timeframe,
                        start_date,
                        end_date,
                    )

                logger.info(
                    f"Successfully fetched {len(df)} bars from {provider.name}"
                )

                return df

            except Exception as e:
                logger.warning(f"Provider {provider.name} failed: {e}")
                last_error = e

                if not fallback:
                    raise

                # Continue to next provider

        # All providers failed
        raise Exception(f"All providers failed. Last error: {last_error}")

    def fetch_from_all_providers(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data from all available providers for comparison.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary mapping provider names to DataFrames
        """
        results = {}

        for provider in self.providers:
            if not provider.is_available():
                continue

            try:
                df = provider.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                )
                results[provider.name] = df

            except Exception as e:
                logger.warning(f"Failed to fetch from {provider.name}: {e}")
                results[provider.name] = None

        return results

    def validate_and_compare(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict:
        """
        Fetch data from all providers and compare for consistency.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary with comparison results
        """
        # Fetch from all providers
        data_dict = self.fetch_from_all_providers(
            symbol,
            timeframe,
            start_date,
            end_date,
        )

        # Validate each dataset
        validation_results = {}
        for provider, df in data_dict.items():
            if df is not None:
                is_valid, issues = self.validator.validate(df, strict=False)
                validation_results[provider] = {
                    "is_valid": is_valid,
                    "issues": issues,
                    "num_bars": len(df),
                }

        # Compare across providers
        is_consistent, comparison = self.validator.compare_providers(data_dict)

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "validation": validation_results,
            "comparison": comparison,
            "is_consistent": is_consistent,
        }

    def get_best_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Get the best quality data from available providers.

        Fetches from all providers, validates, and returns the best dataset.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date

        Returns:
            Best quality OHLCV DataFrame
        """
        # Fetch from all providers
        data_dict = self.fetch_from_all_providers(
            symbol,
            timeframe,
            start_date,
            end_date,
        )

        # Score each dataset
        scores = {}

        for provider, df in data_dict.items():
            if df is None or df.empty:
                scores[provider] = -1
                continue

            # Validate
            is_valid, issues = self.validator.validate(df, strict=False)

            # Calculate score based on:
            # - Validation status
            # - Number of data points
            # - Number of issues

            score = 0

            if is_valid:
                score += 100

            # More data is better
            score += len(df) / 10

            # Fewer issues is better
            score -= len(issues) * 10

            scores[provider] = score

        # Get provider with highest score
        best_provider = max(scores, key=scores.get)

        if scores[best_provider] < 0:
            raise Exception("No valid data available from any provider")

        logger.info(
            f"Selected {best_provider} as best data source (score={scores[best_provider]:.1f})"
        )

        return data_dict[best_provider]
