# TradeAI: Neural Network-Based Trading System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular Python application designed to train neural networks for identifying trade reversals across multiple financial instruments and timeframes.

## Features

- **Multi-Provider Data Collection**: Fetch OHLCV data from multiple sources (Yahoo Finance, Alpha Vantage, Binance, Polygon, Twelve Data)
- **Data Validation**: Automatic data consistency checking across providers
- **Intelligent Caching**: Reduce API calls with smart caching
- **Multi-Timeframe Analysis**: Analyze data across multiple timeframes simultaneously
- **Advanced Feature Engineering**: 40+ technical indicators and custom features
- **Neural Network Models**: MLP and LSTM architectures with ensemble support
- **Three-Model Ensemble**: Reversal, continuation, and direction models for robust predictions
- **Interactive Visualization**: HTML reports with Plotly candlestick charts
- **Forward-Looking Bias Prevention**: Systematic temporal validation
- **Signal Combination**: Rule-based, weighted, and meta-model strategies
- **Modular Design**: Clean separation of concerns for easy extension

## Project Status

### ✅ Production Ready

TradeAI is now a **complete production-ready system** with all core modules implemented and tested:

1. **Core Utilities** ✅
   - Logger (Loguru-based)
   - Configuration Loader (YAML + environment variables)
   - Validators (OHLCV, timeframe, symbol)
   - Temporal validation (bias prevention)

2. **Data Collection** ✅
   - Multi-provider support (Yahoo Finance, Alpha Vantage, Binance, etc.)
   - Intelligent caching with TTL
   - Data validator with consistency checking
   - Data aggregator with automatic fallback

3. **Data Preprocessing** ✅
   - Data cleaning (missing values, outliers)
   - Normalization (MinMax, Z-score, Robust)
   - Multi-timeframe alignment
   - Train/val/test splitting

4. **Feature Engineering** ✅
   - 40+ technical indicators
   - Price features, moving averages, momentum
   - Volatility and volume indicators

5. **Labeling Module** ✅
   - Reversal labeling (extrema detection)
   - Continuation labeling (trend persistence)
   - Direction labeling (baseline)

6. **Neural Network Models** ✅
   - MLP and LSTM architectures
   - Three-model ensemble system

7. **Training Pipeline** ✅
   - Complete training system with early stopping
   - Checkpointing and model saving

8. **Visualization** ✅
   - Interactive HTML reports with Plotly
   - Candlestick charts with labels and signals
   - Performance metrics tables

9. **Signal Generation** ✅
   - Rule-based combination
   - Weighted combination
   - Meta-model stacking

### 🚧 Future Enhancements

- Walk-forward backtesting engine
- Hyperparameter optimization (Optuna)
- Additional model architectures (GRU, CNN, Transformer)
- Real-time prediction API

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

**The fastest way to get started:**

```bash
# Install dependencies
pip install -r requirements_tradeai.txt

# Train three-model ensemble (recommended)
python scripts/train_ensemble_models.py

# Or train single model (basic)
python scripts/train_first_model.py

# View example visualizations
python scripts/demo_visualization.py
```

**See [QUICKSTART.md](QUICKSTART.md) for detailed instructions.**

### Python API Usage

```python
from tradeAI.data.providers.yfinance_provider import YFinanceProvider
from tradeAI.preprocessing.cleaner import DataCleaner
from tradeAI.features.feature_engineer import FeatureEngineer
from tradeAI.labeling.reversal_labeler import ReversalLabeler
from tradeAI.models.architectures.mlp import MLP
from tradeAI.training.trainer import Trainer
from datetime import datetime, timedelta

# 1. Fetch data
provider = YFinanceProvider()
df = provider.fetch_ohlcv(
    symbol="SPY",
    timeframe="1h",
    start_date=datetime.now() - timedelta(days=365)
)

# 2. Preprocess
cleaner = DataCleaner()
df = cleaner.clean(df)

# 3. Add features
engineer = FeatureEngineer()
df = engineer.add_all_features(df)
df = df.dropna()

# 4. Create labels
labeler = ReversalLabeler(method="multiclass", num_classes=5)
df["label"] = labeler.label(df)

# 5. Split and normalize (CRITICAL: split BEFORE normalizing!)
from tradeAI.preprocessing.normalizer import Normalizer

X = df.drop("label", axis=1).values
y = df["label"].values

n = len(X)
X_train_raw = X[:int(n*0.7)]
X_test_raw = X[int(n*0.85):]

normalizer = Normalizer(method="minmax")
X_train = normalizer.fit_transform(pd.DataFrame(X_train_raw)).values
X_test = normalizer.transform(pd.DataFrame(X_test_raw)).values

# 6. Train
model = MLP(input_size=X_train.shape[1], output_size=5)
trainer = Trainer(model, epochs=50)
trainer.fit(X_train, y_train, task="classification")

# 7. Evaluate
metrics = trainer.evaluate(X_test, y_test, task="classification")
print(f"Test Accuracy: {metrics['test_accuracy']:.3f}")
```

## Architecture

The application follows a modular architecture with clear separation of concerns:

```
tradeAI/
├── data/              # Data collection and management ✅
├── preprocessing/     # Data cleaning and normalization ✅
├── features/          # Feature engineering ✅
├── labeling/          # Labeling strategies ✅
├── models/            # Neural network models ✅
├── training/          # Training pipeline ✅
├── visualization/     # Interactive HTML reports ✅
└── utils/             # Utilities ✅
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

## Visualization

Training scripts automatically generate **interactive HTML reports** using Plotly:

```bash
# Generate example reports
python scripts/demo_visualization.py

# View reports
open results/examples/*.html
```

**Features:**
- Interactive candlestick charts (zoom, pan, hover)
- Color-coded labels and signals
- Prediction markers (correct/incorrect)
- Confidence and probability visualizations
- Performance metrics tables

See [tradeAI/README.md](tradeAI/README.md#-visualization) for detailed visualization documentation.

---

## Roadmap

### ✅ Completed
- [x] Core utilities
- [x] Data collection module
- [x] Data preprocessing
- [x] Feature engineering (40+ indicators)
- [x] Labeling module (reversal, continuation, direction)
- [x] Neural network models (MLP, LSTM)
- [x] Training pipeline
- [x] Three-model ensemble
- [x] Signal combination strategies
- [x] Interactive visualization
- [x] Temporal validation framework

### 🚧 Future Enhancements
- [ ] Walk-forward backtesting engine
- [ ] Hyperparameter optimization (Optuna)
- [ ] Additional architectures (GRU, CNN, Transformer)
- [ ] Real-time prediction API
- [ ] Production deployment guide

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
