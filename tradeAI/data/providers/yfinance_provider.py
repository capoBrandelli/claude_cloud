"""
Yahoo Finance data provider for TradeAI
"""

import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from tradeAI.data.base_provider import BaseDataProvider, ProviderError


class YFinanceProvider(BaseDataProvider):
    """Yahoo Finance data provider using yfinance library."""

    # Timeframe mapping: our format -> yfinance format
    TIMEFRAME_MAP = {
        "1m": "1m",
        "2m": "2m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "60m": "60m",
        "90m": "90m",
        "1h": "1h",
        "1d": "1d",
        "5d": "5d",
        "1wk": "1wk",
        "1mo": "1mo",
        "3mo": "3mo",
        # Aliases
        "1min": "1m",
        "5min": "5m",
        "15min": "15m",
        "30min": "30m",
        "1hour": "1h",
        "1day": "1d",
        "1w": "1wk",
        "1M": "1mo",
        "daily": "1d",
        "weekly": "1wk",
        "monthly": "1mo",
    }

    def __init__(self):
        """Initialize Yahoo Finance provider."""
        super().__init__(
            name="YFinance",
            rate_limit=2000,  # requests per hour
            supported_asset_classes=["stocks", "indices", "etfs", "forex", "commodities", "crypto"],
            supported_timeframes=list(self.TIMEFRAME_MAP.keys()),
        )

    def is_available(self) -> bool:
        """
        Check if Yahoo Finance is available.

        Returns:
            True (always available, no API key required)
        """
        return True

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.

        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'SPY', 'BTC-USD')
            timeframe: Timeframe (e.g., '1m', '5m', '1h', '1d')
            start_date: Start date for data
            end_date: End date for data
            limit: Maximum number of bars to fetch

        Returns:
            DataFrame with OHLCV data

        Raises:
            ProviderError: If data fetching fails
        """
        self.validate_request(symbol, timeframe)

        # Map timeframe
        yf_timeframe = self.TIMEFRAME_MAP.get(timeframe.lower())
        if yf_timeframe is None:
            raise ProviderError(f"Unsupported timeframe: {timeframe}")

        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()

        if start_date is None:
            # Default lookback based on timeframe
            lookback_days = self._get_default_lookback(timeframe)
            start_date = end_date - timedelta(days=lookback_days)

        try:
            self.logger.info(
                f"Fetching {symbol} {timeframe} from {start_date} to {end_date}"
            )

            # Create ticker object
            ticker = yf.Ticker(symbol)

            # Fetch data
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=yf_timeframe,
                auto_adjust=True,  # Adjust for splits and dividends
            )

            if df.empty:
                raise ProviderError(
                    f"No data returned for {symbol} {timeframe}. "
                    "Check if symbol is valid and data is available for the requested period."
                )

            # Limit number of rows if specified
            if limit is not None and len(df) > limit:
                df = df.tail(limit)

            # Normalize to standard format
            df = self.normalize_ohlcv(df)

            self.logger.info(f"Successfully fetched {len(df)} bars for {symbol}")

            return df

        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"Failed to fetch data from Yahoo Finance: {str(e)}")

    def _get_default_lookback(self, timeframe: str) -> int:
        """
        Get default lookback period in days based on timeframe.

        Args:
            timeframe: Timeframe string

        Returns:
            Number of days to look back
        """
        if "m" in timeframe and "mo" not in timeframe:
            # Minutes - last 7 days
            return 7
        elif "h" in timeframe:
            # Hours - last 60 days
            return 60
        elif "d" in timeframe:
            # Days - last 2 years
            return 730
        elif "wk" in timeframe or "w" == timeframe:
            # Weeks - last 5 years
            return 1825
        elif "mo" in timeframe or "M" in timeframe:
            # Months - last 10 years
            return 3650
        else:
            # Default - last 365 days
            return 365

    def fetch_multiple_symbols(
        self,
        symbols: list,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict:
        """
        Fetch OHLCV data for multiple symbols.

        Args:
            symbols: List of trading symbols
            timeframe: Timeframe
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}

        for symbol in symbols:
            try:
                df = self.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                )
                results[symbol] = df

                # Rate limiting - avoid hitting API limits
                time.sleep(0.5)

            except ProviderError as e:
                self.logger.warning(f"Failed to fetch {symbol}: {e}")
                results[symbol] = None

        return results
