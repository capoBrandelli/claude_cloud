# TradeAI: Neural Network-Based Trading System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular Python application designed to train neural networks for identifying trade reversals across multiple financial instruments and timeframes.

## Features

- **Multi-Provider Data Collection**: Fetch OHLCV data from multiple sources (Yahoo Finance, Alpha Vantage, Binance, Polygon, Twelve Data)
- **Data Validation**: Automatic data consistency checking across providers
- **Intelligent Caching**: Reduce API calls with smart caching
- **Multi-Timeframe Analysis**: Analyze data across multiple timeframes simultaneously
- **Advanced Feature Engineering**: 50+ technical indicators and custom features
- **Neural Network Models**: Multiple architectures (MLP, LSTM, GRU, CNN, Transformer, Ensemble)
- **Hyperparameter Optimization**: Automatic tuning with Optuna
- **Comprehensive Backtesting**: Realistic backtesting with transaction costs and risk management
- **Experiment Tracking**: Integration with MLflow for tracking experiments
- **Modular Design**: Clean separation of concerns for easy extension

## Project Status

### ✅ Completed Modules

1. **Core Utilities**
   - Logger (Loguru-based)
   - Configuration Loader (YAML + environment variables)
   - Validators (OHLCV, timeframe, symbol)
   - Helper functions

2. **Data Collection**
   - Base data provider interface
   - YFinance provider implementation
   - Data validator with consistency checking
   - Cache manager (Parquet-based)
   - Data aggregator with multi-provider support

### 🚧 In Progress

3. **Data Preprocessing** (Next)
4. **Feature Engineering**
5. **Labeling Module**
6. **Neural Network Models**
7. **Training Pipeline**
8. **Evaluation & Visualization**
9. **Backtesting Engine**

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd claude_cloud

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_tradeai.txt

# Install in development mode
pip install -e .

# Copy environment configuration
cp .env.example .env
# Edit .env and add your API keys
```

## Quick Start

### 1. Configure API Keys

Edit `.env` and add your API keys:

```bash
ALPHA_VANTAGE_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
```

### 2. Fetch Data

```python
from tradeai.data.providers.yfinance_provider import YFinanceProvider
from tradeai.data.data_aggregator import DataAggregator
from datetime import datetime, timedelta

# Create provider
provider = YFinanceProvider()

# Fetch data
df = provider.fetch_ohlcv(
    symbol="SPY",
    timeframe="1h",
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now(),
)

print(df.head())
print(f"Fetched {len(df)} bars")
```

### 3. Use Data Aggregator with Multiple Providers

```python
from tradeai.data.providers.yfinance_provider import YFinanceProvider
from tradeai.data.data_aggregator import DataAggregator
from tradeai.data.cache_manager import CacheManager

# Create providers
providers = [
    YFinanceProvider(),
    # Add more providers as implemented
]

# Create aggregator with caching
aggregator = DataAggregator(
    providers=providers,
    cache_manager=CacheManager(enabled=True),
)

# Fetch with automatic fallback
df = aggregator.fetch_ohlcv(
    symbol="AAPL",
    timeframe="1d",
    start_date=datetime.now() - timedelta(days=365),
)

# Validate and compare across providers
results = aggregator.validate_and_compare(
    symbol="AAPL",
    timeframe="1d",
    start_date=datetime.now() - timedelta(days=30),
)

print(f"Data is consistent: {results['is_consistent']}")
```

## Architecture

The application follows a modular architecture with clear separation of concerns:

```
tradeai/
├── data/              # Data collection and management
├── preprocessing/     # Data cleaning and normalization
├── features/          # Feature engineering
├── labeling/          # Reversal labeling
├── models/            # Neural network models
├── training/          # Training pipeline
├── evaluation/        # Model evaluation
├── backtesting/       # Trading simulation
├── visualization/     # Charts and dashboards
└── utils/             # Utilities
```

See [TRADEAI_ARCHITECTURE_PLAN.md](TRADEAI_ARCHITECTURE_PLAN.md) for detailed architecture documentation.

## Configuration

All configuration is managed through YAML files in the `config/` directory:

- `default.yaml`: General application settings
- `data_providers.yaml`: Data provider configuration
- `features.yaml`: Feature engineering settings
- `models.yaml`: Model architectures and training
- `backtesting.yaml`: Backtesting parameters

## Supported Asset Classes

- **Stock Indices**: SPY, QQQ, DIA
- **Forex**: EURUSD, GBPUSD, USDJPY
- **Commodities**: GLD, USO, SLV
- **Cryptocurrencies**: BTCUSDT, ETHUSDT
- **Individual Stocks**: AAPL, MSFT, GOOGL, and more

## Supported Timeframes

- Minutes: 1m, 5m, 15m, 30m
- Hours: 1h, 4h
- Days: 1d
- Weeks: 1w
- Months: 1M

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black tradeai/
isort tradeai/
flake8 tradeai/
```

### Type Checking

```bash
mypy tradeai/
```

## Roadmap

- [x] Core utilities
- [x] Data collection module
- [ ] Data preprocessing
- [ ] Feature engineering (technical indicators)
- [ ] Labeling module (extrema detection)
- [ ] Neural network models
- [ ] Training pipeline
- [ ] Evaluation and visualization
- [ ] Backtesting engine
- [ ] Web dashboard
- [ ] Real-time prediction API
- [ ] Paper trading integration

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{tradeai2025,
  title={TradeAI: Neural Network-Based Trading System},
  author={TradeAI Team},
  year={2025},
  url={https://github.com/yourusername/tradeai}
}
```

## Disclaimer

This software is for educational and research purposes only. Trading financial instruments carries risk. Past performance does not guarantee future results. Always do your own research and consult with financial professionals before making investment decisions.

## Support

For questions, issues, or feature requests, please open an issue on GitHub.

---

**Made with ❤️ for the trading and AI community**
