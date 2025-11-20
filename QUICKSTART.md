# Quick Start Guide

Get the Financial Sentiment Analyzer up and running in 5 minutes!

## Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Initialize the database
python main.py --init
```

## Run Your First Analysis

### Option 1: Full Pipeline (Recommended)

Run everything with one command:

```bash
python main.py --full --days 7
```

This will:
1. ✓ Initialize database with symbols
2. ✓ Collect news and futures data
3. ✓ Analyze sentiment
4. ✓ Compare news vs futures
5. ✓ Display summary
6. ✓ Launch interactive dashboard at http://localhost:8050

### Option 2: Step by Step

```bash
# Step 1: Collect data
python main.py --collect --days 7

# Step 2: Run analysis
python main.py --analyze

# Step 3: View summary
python main.py --summary

# Step 4: Launch dashboard
python main.py --dashboard
```

## What You'll See

### Terminal Output

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
    💡 Market signals aligned - both indicate bullish sentiment

Strong Divergence (2 symbols)
--------------------------------------------------------------------------------

  EUR/USD (FOREX)
    News:    Bearish  (-0.38) [5 articles]
    Futures: Bullish  (+0.42) [7 data points]
    Alignment: 0.20 | Confidence: 0.65
    ⚠ Strong divergence - potential opportunity
```

### Dashboard Features

Access at **http://localhost:8050**

- 📊 **Summary Cards**: Key metrics at a glance
- 📈 **Comparison Charts**: News vs Futures sentiment
- 🔥 **Alignment Heatmap**: Visual sentiment alignment
- 📉 **Time Series**: Sentiment trends over time
- 💡 **Divergence Opportunities**: Trading signals
- 📋 **Detailed Analysis**: Complete data tables

## Customization

### Change Time Period

```bash
python main.py --full --days 14  # 14 days of data
```

### Change Dashboard Port

```bash
python main.py --dashboard --port 8080
```

### View Divergence Opportunities Only

```bash
python main.py --divergence
```

## Tracked Markets

By default, the analyzer tracks:

- **Forex**: EUR/USD, GBP/USD, USD/JPY, AUD/USD
- **Commodities**: Gold (GC), Oil (CL), Silver (SI), Natural Gas (NG)
- **Stocks**: AAPL, MSFT, GOOGL, TSLA
- **ETFs**: SPY, QQQ, IWM, GLD

## Understanding the Output

### Sentiment Scores

- **+1.0**: Maximum bullish sentiment
- **0.0**: Neutral sentiment
- **-1.0**: Maximum bearish sentiment

### Alignment Status

- **Strong Alignment** (0.8-1.0): High confidence - signals agree
- **Moderate Alignment** (0.6-0.8): Signals generally agree
- **Weak Alignment** (0.4-0.6): Slight agreement
- **Mild Divergence** (0.2-0.4): Same direction, different magnitude
- **Strong Divergence** (0.0-0.2): Conflicting signals - potential opportunity

### What to Look For

✅ **Strong Alignment + High Confidence** = Reliable signal
⚠️ **Strong Divergence + High Magnitude** = Potential trading opportunity
ℹ️ **Low Confidence** = Need more data

## Next Steps

1. **Explore the Dashboard**: Interactive charts and filters
2. **Review Divergences**: Identify potential opportunities
3. **Customize Data Sources**: See README.md for real API integration
4. **Run Examples**: `python example_usage.py` for programmatic usage

## Troubleshooting

**Dashboard won't start?**
```bash
# Try a different port
python main.py --dashboard --port 8051
```

**No data showing?**
```bash
# Make sure you collected data first
python main.py --collect
```

**Import errors?**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## Get Help

- Full documentation: See `README.md`
- Example code: Run `python example_usage.py`
- Issues: Check error messages for guidance

## Pro Tips

💡 Run `--collect` daily to build historical data
💡 Use `--days 30` for longer-term trend analysis
💡 Check divergences regularly for trading opportunities
💡 Higher confidence scores = more reliable signals

---

**Ready to start?** Run `python main.py --full` now!
