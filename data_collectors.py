"""
Data Collection Modules for News and Futures Data
This module provides interfaces for collecting market data.
In production, replace mock data with real API calls.
"""
from typing import List, Dict
from datetime import datetime, timedelta
import random
import json


class NewsCollector:
    """
    Collects financial news data

    NOTE: This uses mock data for demonstration.
    In production, integrate with real news APIs like:
    - NewsAPI (newsapi.org)
    - Alpha Vantage News
    - Finnhub
    - Bloomberg API
    - Reuters API
    """

    def __init__(self):
        self.sources = [
            'Bloomberg', 'Reuters', 'CNBC', 'Financial Times',
            'Wall Street Journal', 'MarketWatch', 'Yahoo Finance'
        ]

        # Sample news templates by market type
        self.news_templates = {
            'forex': [
                "{symbol} rallies as {event} boosts currency sentiment",
                "{symbol} falls amid concerns over {event}",
                "Analysts upgrade {symbol} outlook on positive {event}",
                "{symbol} trading volatile after {event} announcement",
                "Central bank decision impacts {symbol} exchange rates"
            ],
            'commodity': [
                "{symbol} prices surge on supply concerns from {event}",
                "{symbol} outlook remains bullish despite {event}",
                "Demand for {symbol} increases as {event} drives growth",
                "{symbol} falls on profit-taking after {event}",
                "Weather patterns affect {symbol} production and pricing"
            ],
            'stocks': [
                "{symbol} beats earnings expectations for Q{quarter}",
                "{symbol} announces {event} in strategic move",
                "Analysts remain bullish on {symbol} after {event}",
                "{symbol} faces headwinds from {event}",
                "{symbol} stock rallies on positive {event} data"
            ],
            'etf': [
                "{symbol} sees strong inflows amid {event}",
                "{symbol} rebalances portfolio focusing on {event}",
                "Market volatility drives interest in {symbol}",
                "{symbol} outperforms benchmark on {event}",
                "Investors rotate into {symbol} for {event} exposure"
            ]
        }

        self.events = [
            'economic data', 'policy changes', 'geopolitical developments',
            'earnings reports', 'guidance updates', 'sector rotation',
            'inflation data', 'employment figures', 'GDP growth'
        ]

    def collect_news(self, market_type: str, symbol: str,
                    days_back: int = 7, count: int = 10) -> List[Dict]:
        """
        Collect news articles for a specific symbol

        Args:
            market_type: Type of market (forex, commodity, stocks, etf)
            symbol: Symbol to collect news for
            days_back: How many days back to collect news
            count: Number of articles to collect

        Returns:
            List of news article dictionaries
        """
        news_articles = []

        for i in range(count):
            # Generate random publication date within the range
            days_ago = random.randint(0, days_back)
            hours_ago = random.randint(0, 23)
            pub_date = datetime.now() - timedelta(days=days_ago, hours=hours_ago)

            # Select random template and event
            template = random.choice(self.news_templates.get(market_type.lower(),
                                                            self.news_templates['stocks']))
            event = random.choice(self.events)
            quarter = random.randint(1, 4)

            title = template.format(symbol=symbol, event=event, quarter=quarter)

            # Generate content
            content = self._generate_article_content(symbol, market_type, event)

            article = {
                'market_type': market_type,
                'symbol': symbol,
                'title': title,
                'content': content,
                'source': random.choice(self.sources),
                'publication_date': pub_date.strftime('%Y-%m-%d %H:%M:%S'),
                'url': f'https://example.com/news/{symbol.lower()}-{i}'
            }

            news_articles.append(article)

        return news_articles

    def _generate_article_content(self, symbol: str, market_type: str,
                                  event: str) -> str:
        """Generate mock article content"""
        sentiments = ['positive', 'negative', 'mixed']
        sentiment = random.choice(sentiments)

        if sentiment == 'positive':
            content = (f"Market analysts are optimistic about {symbol} following recent "
                      f"{event}. The outlook remains bullish with strong fundamentals "
                      f"supporting continued growth. Traders are positioning for further "
                      f"gains as momentum builds.")
        elif sentiment == 'negative':
            content = (f"Concerns mount for {symbol} as {event} weighs on market sentiment. "
                      f"Analysts warn of potential downside risks. Investors are taking "
                      f"a cautious stance amid uncertainty and volatile conditions.")
        else:
            content = (f"{symbol} shows mixed signals following {event}. While some "
                      f"indicators point to strength, others suggest caution. Market "
                      f"participants are divided on the near-term outlook.")

        return content


class FuturesCollector:
    """
    Collects futures market data

    NOTE: This uses mock data for demonstration.
    In production, integrate with real futures data APIs like:
    - CME Group API
    - CFTC Commitment of Traders (COT) reports
    - Interactive Brokers API
    - Alpha Vantage
    - Quandl
    """

    def __init__(self):
        pass

    def collect_futures_data(self, market_type: str, symbol: str,
                            days_back: int = 7) -> List[Dict]:
        """
        Collect futures positioning data

        Args:
            market_type: Type of market (forex, commodity, stocks, etf)
            symbol: Symbol to collect data for
            days_back: How many days back to collect data

        Returns:
            List of futures data dictionaries
        """
        futures_data = []

        # Generate daily futures data
        for i in range(days_back):
            days_ago = i
            data_date = datetime.now() - timedelta(days=days_ago)

            # Generate mock futures positioning data
            base_oi = random.randint(100000, 500000)  # Open Interest

            # Generate long/short positions with some trend
            trend = random.uniform(-0.3, 0.3)
            long_ratio = 0.5 + trend + random.uniform(-0.1, 0.1)
            long_ratio = max(0.2, min(0.8, long_ratio))  # Clamp between 20-80%

            long_positions = base_oi * long_ratio
            short_positions = base_oi * (1 - long_ratio)

            # Generate price data
            base_price = self._get_base_price(market_type, symbol)
            price_variation = random.uniform(-0.05, 0.05)
            price = base_price * (1 + price_variation)

            # Generate volume
            volume = random.randint(50000, 200000)

            futures_record = {
                'market_type': market_type,
                'symbol': symbol,
                'data_date': data_date.strftime('%Y-%m-%d'),
                'open_interest': base_oi,
                'long_positions': long_positions,
                'short_positions': short_positions,
                'net_positions': long_positions - short_positions,
                'price': round(price, 2),
                'volume': volume,
                'data_source': 'Mock Futures Exchange'
            }

            futures_data.append(futures_record)

        return futures_data

    def _get_base_price(self, market_type: str, symbol: str) -> float:
        """Get a realistic base price for the symbol"""
        price_ranges = {
            'forex': (0.8, 1.5),
            'commodity': (50, 200),
            'stocks': (20, 500),
            'etf': (50, 300)
        }

        range_min, range_max = price_ranges.get(market_type.lower(), (50, 200))
        return random.uniform(range_min, range_max)

    def collect_historical_prices(self, symbol: str, days: int = 30) -> List[Dict]:
        """
        Collect historical price data for momentum analysis

        Args:
            symbol: Symbol to collect data for
            days: Number of days of historical data

        Returns:
            List of price data dictionaries
        """
        prices = []
        base_price = random.uniform(50, 200)

        # Generate trending price data
        trend = random.uniform(-0.01, 0.01)  # Daily trend

        for i in range(days, 0, -1):
            date = datetime.now() - timedelta(days=i)

            # Add trend and random variation
            daily_change = trend + random.uniform(-0.02, 0.02)
            base_price = base_price * (1 + daily_change)

            volume = random.randint(100000, 1000000)

            price_data = {
                'date': date.strftime('%Y-%m-%d'),
                'price': round(base_price, 2),
                'volume': volume
            }

            prices.append(price_data)

        return prices


class MarketSymbolManager:
    """Manages market symbols to track"""

    def __init__(self):
        self.default_symbols = {
            'forex': [
                {'symbol': 'EUR/USD', 'name': 'Euro/US Dollar',
                 'description': 'Most traded currency pair'},
                {'symbol': 'GBP/USD', 'name': 'British Pound/US Dollar',
                 'description': 'Major currency pair'},
                {'symbol': 'USD/JPY', 'name': 'US Dollar/Japanese Yen',
                 'description': 'Major currency pair'},
                {'symbol': 'AUD/USD', 'name': 'Australian Dollar/US Dollar',
                 'description': 'Commodity currency pair'}
            ],
            'commodity': [
                {'symbol': 'GC', 'name': 'Gold Futures',
                 'description': 'Gold commodity futures'},
                {'symbol': 'CL', 'name': 'Crude Oil Futures',
                 'description': 'WTI Crude Oil futures'},
                {'symbol': 'SI', 'name': 'Silver Futures',
                 'description': 'Silver commodity futures'},
                {'symbol': 'NG', 'name': 'Natural Gas Futures',
                 'description': 'Natural gas commodity futures'}
            ],
            'stocks': [
                {'symbol': 'AAPL', 'name': 'Apple Inc.',
                 'description': 'Technology company'},
                {'symbol': 'MSFT', 'name': 'Microsoft Corporation',
                 'description': 'Technology company'},
                {'symbol': 'GOOGL', 'name': 'Alphabet Inc.',
                 'description': 'Technology company'},
                {'symbol': 'TSLA', 'name': 'Tesla Inc.',
                 'description': 'Electric vehicle manufacturer'}
            ],
            'etf': [
                {'symbol': 'SPY', 'name': 'SPDR S&P 500 ETF',
                 'description': 'S&P 500 index ETF'},
                {'symbol': 'QQQ', 'name': 'Invesco QQQ Trust',
                 'description': 'Nasdaq-100 index ETF'},
                {'symbol': 'IWM', 'name': 'iShares Russell 2000 ETF',
                 'description': 'Small-cap stocks ETF'},
                {'symbol': 'GLD', 'name': 'SPDR Gold Shares',
                 'description': 'Gold commodity ETF'}
            ]
        }

    def get_default_symbols(self, market_type: str = None) -> List[Dict]:
        """Get default symbols to track"""
        if market_type:
            return self.default_symbols.get(market_type.lower(), [])

        # Return all symbols
        all_symbols = []
        for market_type, symbols in self.default_symbols.items():
            for symbol_info in symbols:
                symbol_info['market_type'] = market_type
                all_symbols.append(symbol_info)

        return all_symbols
