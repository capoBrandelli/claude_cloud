#!/usr/bin/env python3
"""
Demo script for TradeAI data collection module

This script demonstrates how to:
1. Fetch data from multiple providers
2. Validate data quality
3. Compare data across providers
4. Use caching to reduce API calls
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tradeai.data.providers.yfinance_provider import YFinanceProvider
from tradeai.data.data_aggregator import DataAggregator
from tradeai.data.cache_manager import CacheManager
from tradeai.data.data_validator import DataValidator
from tradeai.utils.logger import setup_logging, get_logger

# Setup logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)


def demo_single_provider():
    """Demo fetching data from a single provider."""
    logger.info("=" * 60)
    logger.info("Demo 1: Fetching data from Yahoo Finance")
    logger.info("=" * 60)

    # Create provider
    provider = YFinanceProvider()

    # Fetch data
    symbol = "SPY"
    timeframe = "1h"
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()

    logger.info(f"Fetching {symbol} {timeframe} data...")

    df = provider.fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )

    logger.info(f"Fetched {len(df)} bars")
    logger.info(f"\nFirst 5 bars:\n{df.head()}")
    logger.info(f"\nLast 5 bars:\n{df.tail()}")
    logger.info(f"\nData summary:\n{df.describe()}")

    return df


def demo_data_validation(df):
    """Demo data validation."""
    logger.info("\n" + "=" * 60)
    logger.info("Demo 2: Data Validation")
    logger.info("=" * 60)

    validator = DataValidator(
        price_tolerance=0.01,
        volume_tolerance=0.05,
        min_data_points=10,
    )

    is_valid, issues = validator.validate(df, strict=True)

    if is_valid:
        logger.info("✓ Data validation passed!")
    else:
        logger.warning(f"✗ Data validation found issues: {issues}")

    return is_valid


def demo_caching():
    """Demo caching functionality."""
    logger.info("\n" + "=" * 60)
    logger.info("Demo 3: Data Caching")
    logger.info("=" * 60)

    # Create provider with cache
    provider = YFinanceProvider()
    cache = CacheManager(enabled=True, ttl_seconds=3600)

    symbol = "AAPL"
    timeframe = "1d"
    start_date = datetime.now() - timedelta(days=90)

    # First fetch - should hit API
    logger.info("First fetch (should hit API)...")
    start_time = datetime.now()

    df1 = provider.fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
    )

    # Cache the data
    cache.set(df1, provider.name, symbol, timeframe, start_date, None)

    first_duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"First fetch took {first_duration:.2f} seconds")

    # Second fetch - should hit cache
    logger.info("\nSecond fetch (should hit cache)...")
    start_time = datetime.now()

    df2 = cache.get(provider.name, symbol, timeframe, start_date, None)

    second_duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"Second fetch took {second_duration:.2f} seconds")

    if df2 is not None:
        logger.info(f"✓ Cache hit! Speedup: {first_duration / second_duration:.1f}x")
    else:
        logger.warning("✗ Cache miss")

    # Show cache stats
    stats = cache.get_cache_stats()
    logger.info(f"\nCache stats: {stats}")


def demo_aggregator():
    """Demo data aggregator with multiple providers."""
    logger.info("\n" + "=" * 60)
    logger.info("Demo 4: Data Aggregator with Fallback")
    logger.info("=" * 60)

    # Create providers (in priority order)
    providers = [
        YFinanceProvider(),
        # Add more providers here as implemented
    ]

    # Create aggregator
    aggregator = DataAggregator(
        providers=providers,
        cache_manager=CacheManager(enabled=True),
        validate_consistency=True,
    )

    # Fetch with automatic fallback and caching
    symbol = "MSFT"
    timeframe = "1h"
    start_date = datetime.now() - timedelta(days=7)

    logger.info(f"Fetching {symbol} {timeframe} with fallback...")

    df = aggregator.fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        use_cache=True,
        fallback=True,
    )

    logger.info(f"✓ Successfully fetched {len(df)} bars")

    return df


def demo_multi_symbol():
    """Demo fetching multiple symbols."""
    logger.info("\n" + "=" * 60)
    logger.info("Demo 5: Fetching Multiple Symbols")
    logger.info("=" * 60)

    provider = YFinanceProvider()

    symbols = ["SPY", "QQQ", "DIA"]
    timeframe = "1d"
    start_date = datetime.now() - timedelta(days=30)

    logger.info(f"Fetching {len(symbols)} symbols...")

    results = provider.fetch_multiple_symbols(
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date,
    )

    for symbol, df in results.items():
        if df is not None:
            logger.info(f"  ✓ {symbol}: {len(df)} bars")
        else:
            logger.warning(f"  ✗ {symbol}: Failed")


def main():
    """Run all demos."""
    logger.info("TradeAI Data Collection Module Demo")
    logger.info("=" * 60)

    try:
        # Demo 1: Single provider
        df = demo_single_provider()

        # Demo 2: Validation
        demo_data_validation(df)

        # Demo 3: Caching
        demo_caching()

        # Demo 4: Aggregator
        demo_aggregator()

        # Demo 5: Multiple symbols
        demo_multi_symbol()

        logger.info("\n" + "=" * 60)
        logger.info("All demos completed successfully! ✓")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
