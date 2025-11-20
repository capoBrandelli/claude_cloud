"""
Cache manager for TradeAI data
"""

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from tradeAI.utils.logger import get_logger
from tradeAI.utils.helpers import ensure_dir

logger = get_logger(__name__)


class CacheManager:
    """Manage caching of fetched data to avoid redundant API calls."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        ttl_seconds: int = 86400,  # 24 hours
        enabled: bool = True,
    ):
        """
        Initialize CacheManager.

        Args:
            cache_dir: Directory for cache files
            ttl_seconds: Time-to-live for cache entries (seconds)
            enabled: Whether caching is enabled
        """
        if cache_dir is None:
            cache_dir = Path("data/cache")

        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled

        if self.enabled:
            ensure_dir(self.cache_dir)
            logger.info(f"Cache enabled with TTL={ttl_seconds}s, dir={self.cache_dir}")

    def get_cache_key(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> str:
        """
        Generate a cache key for a data request.

        Args:
            provider: Provider name
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date

        Returns:
            Cache key string
        """
        # Create a unique key from request parameters
        key_data = {
            "provider": provider,
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
        }

        # Hash the key data
        key_json = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_json.encode()).hexdigest()

        return f"{provider}_{symbol}_{timeframe}_{key_hash}"

    def get_cache_path(self, cache_key: str) -> Path:
        """
        Get the file path for a cache key.

        Args:
            cache_key: Cache key

        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{cache_key}.parquet"

    def get(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Get cached data if available and not expired.

        Args:
            provider: Provider name
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date

        Returns:
            Cached DataFrame or None if not found/expired
        """
        if not self.enabled:
            return None

        cache_key = self.get_cache_key(provider, symbol, timeframe, start_date, end_date)
        cache_path = self.get_cache_path(cache_key)

        if not cache_path.exists():
            logger.debug(f"Cache miss: {cache_key}")
            return None

        # Check if cache is expired
        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if file_age.total_seconds() > self.ttl_seconds:
            logger.debug(f"Cache expired: {cache_key} (age={file_age})")
            # Delete expired cache
            cache_path.unlink()
            return None

        try:
            # Load cached data
            df = pd.read_parquet(cache_path)
            logger.info(f"Cache hit: {cache_key} ({len(df)} rows)")
            return df

        except Exception as e:
            logger.warning(f"Failed to load cache {cache_key}: {e}")
            # Delete corrupted cache
            cache_path.unlink()
            return None

    def set(
        self,
        df: pd.DataFrame,
        provider: str,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> None:
        """
        Store data in cache.

        Args:
            df: DataFrame to cache
            provider: Provider name
            symbol: Trading symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
        """
        if not self.enabled or df is None or df.empty:
            return

        cache_key = self.get_cache_key(provider, symbol, timeframe, start_date, end_date)
        cache_path = self.get_cache_path(cache_key)

        try:
            # Save to parquet format (efficient for time series data)
            df.to_parquet(cache_path)
            logger.debug(f"Cached data: {cache_key} ({len(df)} rows)")

        except Exception as e:
            logger.warning(f"Failed to cache data {cache_key}: {e}")

    def clear(self, older_than_days: Optional[int] = None) -> int:
        """
        Clear cache files.

        Args:
            older_than_days: Only clear files older than N days (None = clear all)

        Returns:
            Number of files deleted
        """
        if not self.enabled:
            return 0

        deleted = 0
        cutoff_time = None

        if older_than_days is not None:
            cutoff_time = datetime.now() - timedelta(days=older_than_days)

        for cache_file in self.cache_dir.glob("*.parquet"):
            try:
                if cutoff_time is None:
                    # Delete all
                    cache_file.unlink()
                    deleted += 1
                else:
                    # Delete only if older than cutoff
                    file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if file_time < cutoff_time:
                        cache_file.unlink()
                        deleted += 1

            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")

        logger.info(f"Cleared {deleted} cache files")
        return deleted

    def get_cache_size(self) -> int:
        """
        Get total size of cache in bytes.

        Returns:
            Cache size in bytes
        """
        if not self.enabled:
            return 0

        total_size = 0
        for cache_file in self.cache_dir.glob("*.parquet"):
            total_size += cache_file.stat().st_size

        return total_size

    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if not self.enabled:
            return {"enabled": False}

        cache_files = list(self.cache_dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "enabled": True,
            "num_files": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "ttl_seconds": self.ttl_seconds,
            "cache_dir": str(self.cache_dir),
        }
