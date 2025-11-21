#!/usr/bin/env python3
"""
Financial Market Sentiment Analyzer - Main Application
Collects, analyzes, and visualizes sentiment data across multiple markets
"""
import argparse
import sys
from datetime import datetime, timedelta
from typing import List, Dict

from database import Database
from sentiment_analyzer import NewsSentimentAnalyzer, FuturesSentimentAnalyzer
from comparison_engine import SentimentComparator
from data_collectors import NewsCollector, FuturesCollector, MarketSymbolManager
from dashboard import SentimentDashboard


class FinancialSentimentAnalyzer:
    """Main application coordinator"""

    def __init__(self, db_path: str = "financial_sentiment.db"):
        self.db = Database(db_path)
        self.news_analyzer = NewsSentimentAnalyzer()
        self.futures_analyzer = FuturesSentimentAnalyzer()
        self.comparator = SentimentComparator()
        self.news_collector = NewsCollector()
        self.futures_collector = FuturesCollector()
        self.symbol_manager = MarketSymbolManager()

    def initialize_symbols(self):
        """Initialize default market symbols"""
        print("Initializing market symbols...")

        symbols = self.symbol_manager.get_default_symbols()
        for symbol_info in symbols:
            self.db.add_market_symbol(
                market_type=symbol_info['market_type'],
                symbol=symbol_info['symbol'],
                name=symbol_info['name'],
                description=symbol_info.get('description', '')
            )

        print(f"✓ Initialized {len(symbols)} market symbols")

    def collect_and_analyze_news(self, market_type: str, symbol: str,
                                 days_back: int = 7, count: int = 10):
        """Collect news and perform sentiment analysis"""
        print(f"\nCollecting news for {symbol} ({market_type})...")

        # Collect news articles
        articles = self.news_collector.collect_news(market_type, symbol, days_back, count)

        for article in articles:
            # Perform sentiment analysis
            sentiment = self.news_analyzer.analyze_text(
                article['content'],
                article['title']
            )

            # Calculate relevance score
            relevance = self.news_analyzer.calculate_relevance_score(
                f"{article['title']} {article['content']}",
                symbol,
                market_type
            )

            # Extract keywords
            keywords = self.news_analyzer.extract_keywords(article['content'])

            # Store in database
            self.db.insert_news_sentiment({
                'market_type': market_type,
                'symbol': symbol,
                'title': article['title'],
                'content': article['content'],
                'source': article['source'],
                'publication_date': article['publication_date'],
                'url': article.get('url'),
                'sentiment_score': sentiment['sentiment_score'],
                'sentiment_label': sentiment['sentiment_label'],
                'relevance_score': relevance,
                'keywords': keywords
            })

        print(f"✓ Analyzed {len(articles)} news articles")

    def collect_and_analyze_futures(self, market_type: str, symbol: str, days_back: int = 7):
        """Collect futures data and perform sentiment analysis"""
        print(f"Collecting futures data for {symbol} ({market_type})...")

        # Collect futures positioning data
        futures_data = self.futures_collector.collect_futures_data(market_type, symbol, days_back)

        for data in futures_data:
            # Analyze sentiment from positioning
            sentiment = self.futures_analyzer.analyze_positions(
                data['long_positions'],
                data['short_positions'],
                data['open_interest']
            )

            # Store in database
            self.db.insert_futures_sentiment({
                'market_type': market_type,
                'symbol': symbol,
                'data_date': data['data_date'],
                'open_interest': data['open_interest'],
                'long_positions': data['long_positions'],
                'short_positions': data['short_positions'],
                'net_positions': data['net_positions'],
                'sentiment_score': sentiment['sentiment_score'],
                'sentiment_label': sentiment['sentiment_label'],
                'price': data['price'],
                'volume': data['volume'],
                'data_source': data['data_source']
            })

        print(f"✓ Analyzed {len(futures_data)} futures data points")

    def perform_comparison_analysis(self, market_type: str = None, days: int = 7):
        """Compare news and futures sentiment"""
        print(f"\nPerforming sentiment comparison analysis...")

        # Get all tracked symbols
        symbols = self.db.get_market_symbols(market_type)

        comparison_count = 0
        for symbol_info in symbols:
            symbol = symbol_info['symbol']
            m_type = symbol_info['market_type']

            # Get news and futures data
            news_data = self.db.get_news_sentiment(m_type, symbol, days)
            futures_data = self.db.get_futures_sentiment(m_type, symbol, days)

            if news_data or futures_data:
                # Perform comparison
                comparison = self.comparator.analyze_sentiment_group(news_data, futures_data)

                # Store comparison
                self.db.insert_sentiment_comparison({
                    'market_type': m_type,
                    'symbol': symbol,
                    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'news_sentiment_avg': comparison['news_sentiment_avg'],
                    'futures_sentiment_avg': comparison['futures_sentiment_avg'],
                    'alignment_score': comparison['alignment_score'],
                    'alignment_status': comparison['alignment_status'],
                    'divergence_magnitude': comparison.get('divergence_magnitude', 0),
                    'confidence_score': comparison['confidence_score'],
                    'news_count': comparison['news_count'],
                    'futures_count': comparison['futures_count'],
                    'insights': comparison['insights']
                })

                comparison_count += 1

        print(f"✓ Completed {comparison_count} sentiment comparisons")

    def display_summary(self, market_type: str = None, days: int = 7):
        """Display summary of sentiment analysis"""
        print(f"\n{'='*80}")
        print("SENTIMENT ANALYSIS SUMMARY")
        print(f"{'='*80}")

        comparisons = self.db.get_sentiment_comparisons(market_type, days=days)

        if not comparisons:
            print("No comparison data available.")
            return

        # Group by alignment status
        status_groups = {}
        for comp in comparisons:
            status = comp['alignment_status']
            if status not in status_groups:
                status_groups[status] = []
            status_groups[status].append(comp)

        # Display by status
        for status in ['Strong Alignment', 'Moderate Alignment', 'Weak Alignment',
                      'Mild Divergence', 'Strong Divergence']:
            if status in status_groups:
                print(f"\n{status} ({len(status_groups[status])} symbols)")
                print("-" * 80)

                for comp in status_groups[status]:
                    news_label = self._get_sentiment_label(comp['news_sentiment_avg'])
                    futures_label = self._get_sentiment_label(comp['futures_sentiment_avg'])

                    print(f"\n  {comp['symbol']} ({comp['market_type'].upper()})")
                    print(f"    News:    {news_label:8} ({comp['news_sentiment_avg']:+.2f}) "
                          f"[{comp['news_count']} articles]")
                    print(f"    Futures: {futures_label:8} ({comp['futures_sentiment_avg']:+.2f}) "
                          f"[{comp['futures_count']} data points]")
                    print(f"    Alignment: {comp['alignment_score']:.2f} | "
                          f"Confidence: {comp['confidence_score']:.2f}")

                    if comp.get('insights'):
                        print(f"    💡 {comp['insights']}")

        print(f"\n{'='*80}\n")

    def display_divergence_opportunities(self, min_divergence: float = 0.5):
        """Display potential trading opportunities based on divergence"""
        print(f"\n{'='*80}")
        print("DIVERGENCE OPPORTUNITIES")
        print(f"{'='*80}\n")

        comparisons = self.db.get_sentiment_comparisons()
        divergences = [c for c in comparisons
                      if 'Divergence' in c.get('alignment_status', '')
                      and c.get('divergence_magnitude', 0) >= min_divergence]

        if not divergences:
            print("No significant divergence opportunities found.")
            return

        divergences.sort(key=lambda x: x.get('divergence_magnitude', 0), reverse=True)

        for i, div in enumerate(divergences, 1):
            print(f"{i}. {div['symbol']} ({div['market_type'].upper()})")
            print(f"   Divergence Magnitude: {div['divergence_magnitude']:.2f}")
            print(f"   News Sentiment: {div['news_sentiment_avg']:+.2f}")
            print(f"   Futures Sentiment: {div['futures_sentiment_avg']:+.2f}")
            print(f"   Confidence: {div['confidence_score']:.2f}")
            print(f"   💡 {div.get('insights', 'No insights available')}\n")

        print(f"{'='*80}\n")

    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score > 0.2:
            return "Bullish"
        elif score < -0.2:
            return "Bearish"
        else:
            return "Neutral"

    def collect_all_data(self, days_back: int = 7):
        """Collect data for all tracked symbols"""
        print(f"\n{'='*80}")
        print("COLLECTING MARKET DATA")
        print(f"{'='*80}\n")

        symbols = self.db.get_market_symbols()

        for symbol_info in symbols:
            symbol = symbol_info['symbol']
            market_type = symbol_info['market_type']

            print(f"\n{symbol} ({market_type.upper()})")
            print("-" * 40)

            # Collect and analyze news
            self.collect_and_analyze_news(market_type, symbol, days_back, count=5)

            # Collect and analyze futures
            self.collect_and_analyze_futures(market_type, symbol, days_back)

        # Perform comparison analysis
        self.perform_comparison_analysis(days=days_back)

        print(f"\n{'='*80}")
        print("✓ Data collection completed")
        print(f"{'='*80}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Financial Market Sentiment Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --collect --days 7          Collect 7 days of data for all symbols
  %(prog)s --analyze                   Run sentiment comparison analysis
  %(prog)s --summary                   Display summary of analysis
  %(prog)s --dashboard                 Launch interactive dashboard
  %(prog)s --init                      Initialize database with default symbols
  %(prog)s --full                      Run full pipeline (collect, analyze, summary)
        """
    )

    parser.add_argument('--init', action='store_true',
                       help='Initialize database with default symbols')
    parser.add_argument('--collect', action='store_true',
                       help='Collect news and futures data')
    parser.add_argument('--analyze', action='store_true',
                       help='Perform sentiment comparison analysis')
    parser.add_argument('--summary', action='store_true',
                       help='Display analysis summary')
    parser.add_argument('--divergence', action='store_true',
                       help='Display divergence opportunities')
    parser.add_argument('--dashboard', action='store_true',
                       help='Launch interactive dashboard')
    parser.add_argument('--full', action='store_true',
                       help='Run full pipeline (collect, analyze, summary, dashboard)')
    parser.add_argument('--days', type=int, default=7,
                       help='Number of days to analyze (default: 7)')
    parser.add_argument('--db', type=str, default='financial_sentiment.db',
                       help='Database file path (default: financial_sentiment.db)')
    parser.add_argument('--port', type=int, default=8050,
                       help='Dashboard port (default: 8050)')

    args = parser.parse_args()

    # Create analyzer instance
    analyzer = FinancialSentimentAnalyzer(db_path=args.db)

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # Execute requested operations
    if args.init or args.full:
        analyzer.initialize_symbols()

    if args.collect or args.full:
        analyzer.collect_all_data(days_back=args.days)

    if args.analyze or args.full:
        analyzer.perform_comparison_analysis(days=args.days)

    if args.summary or args.full:
        analyzer.display_summary(days=args.days)

    if args.divergence:
        analyzer.display_divergence_opportunities()

    if args.dashboard or args.full:
        dashboard = SentimentDashboard(analyzer.db)
        dashboard.run(debug=False, port=args.port)


if __name__ == '__main__':
    main()
