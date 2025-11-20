#!/usr/bin/env python3
"""
Example usage of the Financial Sentiment Analyzer
Demonstrates programmatic usage of the main components
"""

from main import FinancialSentimentAnalyzer


def example_basic_workflow():
    """Example: Basic workflow"""
    print("="*80)
    print("EXAMPLE: Basic Workflow")
    print("="*80 + "\n")

    # Initialize analyzer
    analyzer = FinancialSentimentAnalyzer(db_path="example.db")

    # Step 1: Initialize symbols
    print("Step 1: Initializing symbols...")
    analyzer.initialize_symbols()

    # Step 2: Collect data for specific symbols
    print("\nStep 2: Collecting data...")
    analyzer.collect_and_analyze_news('stocks', 'AAPL', days_back=7, count=5)
    analyzer.collect_and_analyze_futures('stocks', 'AAPL', days_back=7)

    analyzer.collect_and_analyze_news('forex', 'EUR/USD', days_back=7, count=5)
    analyzer.collect_and_analyze_futures('forex', 'EUR/USD', days_back=7)

    # Step 3: Perform comparison
    print("\nStep 3: Performing comparison analysis...")
    analyzer.perform_comparison_analysis(days=7)

    # Step 4: Display results
    print("\nStep 4: Displaying results...")
    analyzer.display_summary(days=7)
    analyzer.display_divergence_opportunities(min_divergence=0.3)

    print("\n✓ Example completed successfully!\n")


def example_custom_analysis():
    """Example: Custom sentiment analysis"""
    print("="*80)
    print("EXAMPLE: Custom Sentiment Analysis")
    print("="*80 + "\n")

    from sentiment_analyzer import NewsSentimentAnalyzer, FuturesSentimentAnalyzer

    # News sentiment analysis
    news_analyzer = NewsSentimentAnalyzer()

    sample_news = """
    Apple Inc. announced strong quarterly earnings that beat analyst expectations.
    The company reported record revenue driven by robust iPhone sales and
    expanding services business. Market analysts remain bullish on the stock's
    outlook for the coming quarters.
    """

    news_sentiment = news_analyzer.analyze_text(sample_news, "Apple beats earnings")
    print("News Sentiment Analysis:")
    print(f"  Text: {sample_news[:100]}...")
    print(f"  Sentiment Score: {news_sentiment['sentiment_score']}")
    print(f"  Sentiment Label: {news_sentiment['sentiment_label']}")
    print(f"  Confidence: {news_sentiment['confidence']}")

    # Futures sentiment analysis
    futures_analyzer = FuturesSentimentAnalyzer()

    futures_sentiment = futures_analyzer.analyze_positions(
        long_positions=75000,
        short_positions=25000,
        open_interest=100000
    )

    print("\nFutures Sentiment Analysis:")
    print(f"  Long Positions: 75,000")
    print(f"  Short Positions: 25,000")
    print(f"  Sentiment Score: {futures_sentiment['sentiment_score']}")
    print(f"  Sentiment Label: {futures_sentiment['sentiment_label']}")
    print(f"  Long Ratio: {futures_sentiment['long_ratio']}")

    print("\n✓ Custom analysis completed!\n")


def example_comparison_engine():
    """Example: Using comparison engine directly"""
    print("="*80)
    print("EXAMPLE: Comparison Engine")
    print("="*80 + "\n")

    from comparison_engine import SentimentComparator

    comparator = SentimentComparator()

    # Example 1: Strong alignment
    print("Scenario 1: Strong Alignment")
    comparison = comparator.calculate_alignment(
        news_sentiment=0.65,
        futures_sentiment=0.72
    )
    print(f"  News: +0.65 (Bullish)")
    print(f"  Futures: +0.72 (Bullish)")
    print(f"  Alignment Score: {comparison['alignment_score']}")
    print(f"  Status: {comparison['alignment_status']}")
    print(f"  Description: {comparison['description']}")

    # Example 2: Strong divergence
    print("\nScenario 2: Strong Divergence")
    comparison = comparator.calculate_alignment(
        news_sentiment=-0.55,
        futures_sentiment=0.48
    )
    print(f"  News: -0.55 (Bearish)")
    print(f"  Futures: +0.48 (Bullish)")
    print(f"  Alignment Score: {comparison['alignment_score']}")
    print(f"  Status: {comparison['alignment_status']}")
    print(f"  Divergence Magnitude: {comparison['divergence_magnitude']}")
    print(f"  Description: {comparison['description']}")

    print("\n✓ Comparison examples completed!\n")


def example_database_queries():
    """Example: Direct database queries"""
    print("="*80)
    print("EXAMPLE: Database Queries")
    print("="*80 + "\n")

    from database import Database

    db = Database("example.db")

    # Query news sentiment
    news = db.get_news_sentiment(market_type='stocks', symbol='AAPL', days=7)
    print(f"News Articles for AAPL: {len(news)}")
    if news:
        print(f"  Latest article: {news[0]['title']}")
        print(f"  Sentiment: {news[0]['sentiment_score']} ({news[0]['sentiment_label']})")
        print(f"  Source: {news[0]['source']}")

    # Query futures sentiment
    futures = db.get_futures_sentiment(market_type='stocks', symbol='AAPL', days=7)
    print(f"\nFutures Data for AAPL: {len(futures)}")
    if futures:
        print(f"  Latest data: {futures[0]['data_date']}")
        print(f"  Sentiment: {futures[0]['sentiment_score']} ({futures[0]['sentiment_label']})")
        print(f"  Net Positions: {futures[0]['net_positions']:,.0f}")

    # Query comparisons
    comparisons = db.get_sentiment_comparisons(market_type='stocks', days=7)
    print(f"\nComparison Records: {len(comparisons)}")
    if comparisons:
        for comp in comparisons[:3]:  # Show first 3
            print(f"\n  {comp['symbol']}:")
            print(f"    Status: {comp['alignment_status']}")
            print(f"    Alignment: {comp['alignment_score']:.2f}")
            print(f"    News Sent: {comp['news_sentiment_avg']:+.2f}")
            print(f"    Futures Sent: {comp['futures_sentiment_avg']:+.2f}")

    print("\n✓ Database query examples completed!\n")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("FINANCIAL SENTIMENT ANALYZER - USAGE EXAMPLES")
    print("="*80 + "\n")

    try:
        # Run examples
        example_basic_workflow()
        example_custom_analysis()
        example_comparison_engine()
        example_database_queries()

        print("="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nNext steps:")
        print("  1. Run the full pipeline: python main.py --full")
        print("  2. Launch the dashboard: python main.py --dashboard")
        print("  3. Read the README.md for detailed documentation")
        print()

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
