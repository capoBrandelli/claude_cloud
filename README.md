# Financial Market Sentiment Analyzer

A comprehensive Python application for collecting, analyzing, and visualizing financial market sentiment data across Forex, Commodities, Stocks, and ETF markets. The system compares sentiment from news sources with futures market positioning to identify alignment and divergence opportunities.

## Features

### Core Capabilities
- 📰 **News Sentiment Analysis**: Analyzes financial news articles using NLP-based sentiment scoring
- 📊 **Futures Sentiment Analysis**: Evaluates market sentiment from futures positioning data (long/short ratios)
- 🔄 **Sentiment Comparison Engine**: Identifies alignment and divergence between news and futures sentiment
- 📈 **Interactive Dashboard**: Real-time visualization with Plotly Dash
- 🎯 **Multi-Market Coverage**: Forex, Commodities, Stocks, and ETFs
- 💾 **SQLite Database**: Stores historical sentiment data with full provenance tracking

### Sentiment Comparison Insights
- **Strong Alignment**: News and futures sentiment strongly agree (high confidence signals)
- **Moderate/Weak Alignment**: Partial agreement with varying confidence levels
- **Mild Divergence**: Same direction but different magnitudes
- **Strong Divergence**: Conflicting signals - potential trading opportunities

### Tracked Metrics
- Sentiment scores (-1 to +1 scale)
- Alignment scores (0 to 1 scale)
- Relevance scores for news articles
- Confidence scores for analysis
- Source attribution and publication dates
- Keywords and trend analysis

## Project Structure

```
financial-sentiment-analyzer/
├── main.py                    # Main application entry point
├── database.py                # SQLite database schema and operations
├── sentiment_analyzer.py      # News and futures sentiment analysis
├── comparison_engine.py       # Sentiment comparison logic
├── data_collectors.py         # Data collection modules
├── dashboard.py               # Interactive Plotly Dash dashboard
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download the repository:
```bash
git clone <repository-url>
cd claude_cloud
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize the database:
```bash
python main.py --init
```

## Usage

### Command Line Interface

The application provides a comprehensive CLI with multiple operation modes:

#### 1. Full Pipeline (Recommended for first run)
```bash
python main.py --full --days 7
```
This will:
- Initialize the database with default symbols
- Collect news and futures data for the past 7 days
- Perform sentiment comparison analysis
- Display a summary
- Launch the interactive dashboard

#### 2. Individual Operations

**Initialize database with default symbols:**
```bash
python main.py --init
```

**Collect data only:**
```bash
python main.py --collect --days 7
```

**Run comparison analysis:**
```bash
python main.py --analyze --days 7
```

**Display summary:**
```bash
python main.py --summary --days 7
```

**Show divergence opportunities:**
```bash
python main.py --divergence
```

**Launch dashboard only:**
```bash
python main.py --dashboard --port 8050
```

### Interactive Dashboard

The dashboard provides:

1. **Summary Cards**: Quick overview of data metrics
2. **Sentiment Comparison Chart**: Bar chart comparing news vs futures sentiment by symbol
3. **Alignment Heatmap**: Visual representation of alignment scores
4. **Time Series Charts**: Sentiment trends over time for both news and futures
5. **Divergence Opportunities Table**: Highlights symbols with strong divergence
6. **Detailed Analysis Table**: Comprehensive comparison data with insights

Access the dashboard at `http://localhost:8050` (or your specified port)

### Dashboard Filters

- **Market Type**: Filter by Forex, Commodities, Stocks, ETFs, or All Markets
- **Time Period**: View data for 1, 3, 7, 14, or 30 days
- **Refresh Button**: Reload data without restarting the server

## Default Tracked Symbols

### Forex
- EUR/USD - Euro/US Dollar
- GBP/USD - British Pound/US Dollar
- USD/JPY - US Dollar/Japanese Yen
- AUD/USD - Australian Dollar/US Dollar

### Commodities
- GC - Gold Futures
- CL - Crude Oil Futures
- SI - Silver Futures
- NG - Natural Gas Futures

### Stocks
- AAPL - Apple Inc.
- MSFT - Microsoft Corporation
- GOOGL - Alphabet Inc.
- TSLA - Tesla Inc.

### ETFs
- SPY - SPDR S&P 500 ETF
- QQQ - Invesco QQQ Trust
- IWM - iShares Russell 2000 ETF
- GLD - SPDR Gold Shares

## Database Schema

### Tables

1. **news_sentiment**: Stores news articles with sentiment analysis
   - Source, title, content, publication date
   - Sentiment score, label, relevance score
   - Keywords and metadata

2. **futures_sentiment**: Stores futures positioning data
   - Open interest, long/short positions
   - Net positions, price, volume
   - Sentiment score and label

3. **sentiment_comparison**: Stores comparison analysis results
   - Average sentiment scores for news and futures
   - Alignment score and status
   - Divergence magnitude, confidence score
   - Generated insights

4. **market_symbols**: Reference table for tracked symbols
   - Market type, symbol, name, description

## Sentiment Analysis Methodology

### News Sentiment
- **Lexicon-based approach**: Uses financial-specific positive/negative word lists
- **Context awareness**: Handles negations and intensifiers
- **Scoring**: Normalized to -1 (bearish) to +1 (bullish) scale
- **Relevance**: Calculates how relevant the news is to the specific symbol

### Futures Sentiment
- **Position analysis**: Evaluates long/short ratio from futures data
- **Net positioning**: Considers net long/short positions
- **Confidence scoring**: Based on open interest and position size
- **Price momentum**: Optional analysis of price trends

### Comparison Engine
- **Alignment calculation**: Measures agreement between news and futures sentiment
- **Divergence detection**: Identifies conflicting signals
- **Confidence weighting**: Based on data quantity and consistency
- **Insight generation**: Produces human-readable analysis

## Extending the Application

### Adding Real Data Sources

The current implementation uses mock data for demonstration. To integrate real APIs:

1. **News APIs**: Replace `NewsCollector` methods with calls to:
   - NewsAPI (newsapi.org)
   - Alpha Vantage News
   - Finnhub
   - Bloomberg Terminal API

2. **Futures Data**: Replace `FuturesCollector` methods with calls to:
   - CME Group API
   - CFTC Commitment of Traders reports
   - Interactive Brokers API
   - Quandl/Nasdaq Data Link

3. **Advanced NLP**: Integrate transformer-based models:
   ```python
   # Uncomment in requirements.txt:
   # transformers>=4.30.0
   # torch>=2.0.0

   from transformers import pipeline
   sentiment_pipeline = pipeline("sentiment-analysis",
                                 model="ProsusAI/finbert")
   ```

### Adding New Symbols

```python
from database import Database

db = Database()
db.add_market_symbol(
    market_type='stocks',
    symbol='NVDA',
    name='NVIDIA Corporation',
    description='Graphics processing units manufacturer'
)
```

### Custom Analysis

The modular design allows easy customization:

```python
from sentiment_analyzer import NewsSentimentAnalyzer
from comparison_engine import SentimentComparator

# Custom sentiment analysis
analyzer = NewsSentimentAnalyzer()
result = analyzer.analyze_text("Your custom news text here")

# Custom comparison logic
comparator = SentimentComparator()
comparison = comparator.calculate_alignment(news_score, futures_score)
```

## Example Output

```
================================================================================
SENTIMENT ANALYSIS SUMMARY
================================================================================

Strong Alignment (4 symbols)
--------------------------------------------------------------------------------

  AAPL (STOCKS)
    News:    Bullish  (+0.45) [5 articles]
    Futures: Bullish  (+0.52) [7 data points]
    Alignment: 0.93 | Confidence: 0.78
    💡 News sentiment: Bullish (+0.45) | Futures sentiment: Bullish (+0.52) |
       ✓ Market signals aligned - both indicate bullish sentiment

Strong Divergence (2 symbols)
--------------------------------------------------------------------------------

  EUR/USD (FOREX)
    News:    Bearish  (-0.38) [5 articles]
    Futures: Bullish  (+0.42) [7 data points]
    Alignment: 0.20 | Confidence: 0.65
    💡 ⚠ Strong divergence detected - news shows bearish while futures show
       bullish | This could indicate market uncertainty or pending reversal
```

## Performance Considerations

- **Database**: SQLite is suitable for development and small-scale deployments
- **Scaling**: For production, consider PostgreSQL or MySQL
- **API Rate Limits**: Implement caching and rate limiting when using real APIs
- **Dashboard**: Use `debug=False` in production for better performance

## Troubleshooting

### Dashboard won't start
- Check if port 8050 is available: `lsof -i :8050`
- Use a different port: `python main.py --dashboard --port 8051`

### No data displayed
- Ensure you've run data collection: `python main.py --collect`
- Check database file exists: `ls -la financial_sentiment.db`

### Import errors
- Verify all dependencies installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (needs 3.8+)

## Future Enhancements

- [ ] Real-time data streaming
- [ ] Machine learning-based sentiment models (FinBERT, etc.)
- [ ] Email/SMS alerts for divergence opportunities
- [ ] Backtesting framework
- [ ] API endpoint for programmatic access
- [ ] Social media sentiment integration
- [ ] Advanced technical indicators
- [ ] Multi-language support for international news

## License

This project is provided as-is for educational and research purposes.

## Disclaimer

This application is for informational and educational purposes only. It is not financial advice. Trading financial instruments carries risk. Always conduct your own research and consult with qualified financial advisors before making investment decisions.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Support

For questions or issues, please open an issue on the repository.
