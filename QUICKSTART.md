# TradeAI Quick Start Guide

Get TradeAI up and running in 5 minutes!

## What is TradeAI?

TradeAI is a **production-ready** neural network-based trading system that trains ensemble models to identify market reversals, trend continuations, and directional bias across multiple financial instruments.

**Key Features:**
- ✅ No forward-looking bias (systematic validation)
- ✅ Three-model ensemble for robust predictions
- ✅ 40+ technical indicators
- ✅ Multiple signal combination strategies
- ✅ Production-grade data handling

---

## Installation

```bash
# 1. Navigate to project directory
cd /home/user/claude_cloud

# 2. Install dependencies
pip install -r requirements_tradeai.txt

# 3. Install in development mode (optional)
pip install -e .
```

---

## Train Your First Model (3 Options)

### Option 1: Single Model (Basic) ⚡

Train a single MLP or LSTM model:

```bash
python scripts/train_first_model.py
```

**This will:**
- Fetch 1 year of SPY hourly data
- Clean and preprocess
- Generate 40+ technical features
- Label reversals (5 classes)
- Train MLP and LSTM models
- Show performance metrics

**Time:** ~5 minutes on CPU

---

### Option 2: Three-Model Ensemble (Recommended) 🎯

Train three independent models for robust predictions:

```bash
python scripts/train_ensemble_models.py
```

**This trains:**
1. **Reversal Model** - Identifies turning points
2. **Continuation Model** - Predicts trend persistence
3. **Direction Model** - Provides directional baseline

**Output:**
- Three trained models saved to `models/ensemble/`
- Performance metrics for each model
- Ensemble average accuracy
- Usage examples

**Time:** ~15 minutes on CPU

---

### Option 3: Custom Training (Python API) 💻

```python
from datetime import datetime, timedelta
from tradeAI.data.providers.yfinance_provider import YFinanceProvider
from tradeAI.preprocessing.cleaner import DataCleaner
from tradeAI.features.feature_engineer import FeatureEngineer
from tradeAI.labeling.reversal_labeler import ReversalLabeler
from tradeAI.models.architectures.mlp import MLP
from tradeAI.training.trainer import Trainer

# 1. Collect data
provider = YFinanceProvider()
df = provider.fetch_ohlcv(
    symbol="SPY",
    timeframe="1h",
    start_date=datetime.now() - timedelta(days=365)
)

# 2. Preprocess
cleaner = DataCleaner()
df = cleaner.clean(df)

# 3. Add features (40+ indicators)
engineer = FeatureEngineer()
df = engineer.add_all_features(df)
df = df.dropna()

# 4. Create labels
labeler = ReversalLabeler(method="multiclass", num_classes=5)
df["label"] = labeler.label(df)
df = df[df["label"] != -1]

# 5. Split data (CRITICAL: Split BEFORE normalizing!)
from tradeAI.preprocessing.normalizer import Normalizer
import pandas as pd

X = df.drop("label", axis=1).values
y = df["label"].values

# Split first
n = len(X)
X_train_raw = X[:int(n*0.7)]
X_test_raw = X[int(n*0.85):]
y_train = y[:int(n*0.7)]
y_test = y[int(n*0.85):]

# Normalize (fit on training only!)
normalizer = Normalizer(method="minmax")
X_train = normalizer.fit_transform(pd.DataFrame(X_train_raw)).values
X_test = normalizer.transform(pd.DataFrame(X_test_raw)).values

# 6. Train model
model = MLP(input_size=X_train.shape[1], output_size=5)
trainer = Trainer(model, epochs=50)
trainer.fit(X_train, y_train, task="classification")

# 7. Evaluate
metrics = trainer.evaluate(X_test, y_test, task="classification")
print(f"Test Accuracy: {metrics['test_accuracy']:.3f}")
```

---

## Generate Trading Signals

After training the ensemble:

```python
from tradeAI.models.ensemble import RuleBasedCombiner
import numpy as np

# Load your three trained models
# model_reversal, model_continuation, model_direction

# Get latest market data and calculate features
latest_features = get_latest_features()  # Your data pipeline

# Get predictions from all three models
p_reversal = model_reversal.predict(latest_features)
p_continuation = model_continuation.predict(latest_features)
p_direction = model_direction.predict(latest_features)

# Combine predictions
combiner = RuleBasedCombiner(num_classes=5)
result = combiner.combine(p_reversal, p_continuation, p_direction)

# Use signals
signal = result['signal'][0]        # -1 (bearish), 0 (neutral), 1 (bullish)
confidence = result['confidence'][0] # 0.0 - 1.0
agreement = result['agreement'][0]   # 0.0 - 1.0

if signal == 1 and confidence > 0.7 and agreement > 0.8:
    print("🟢 Strong BULLISH signal - High confidence!")
elif signal == -1 and confidence > 0.7 and agreement > 0.8:
    print("🔴 Strong BEARISH signal - High confidence!")
else:
    print("⚪ Weak signal or model disagreement - Stay cautious")
```

---

## Expected Performance

### Model Accuracy

| Model | Expected Accuracy | Training Time (CPU) |
|-------|-------------------|---------------------|
| MLP (Single) | 50-65% | ~2 minutes |
| LSTM (Single) | 55-70% | ~5 minutes |
| Ensemble (3 models) | 55-70% (robust) | ~15 minutes |

**Note:** Reversal prediction is inherently challenging. Accuracy >60% can be profitable with proper risk management!

### Signal Quality

- **High Agreement (>0.8) + High Confidence (>0.7)** = Strong, reliable signals
- **Low Agreement (<0.5)** = Model disagreement, stay cautious
- **Moderate Confidence (0.5-0.7)** = Use smaller position sizes

---

## What You Can Do Now

### 1. Experiment with Different Assets

```bash
# Edit train_ensemble_models.py, change:
symbol = "AAPL"  # or "QQQ", "BTCUSD", etc.
```

### 2. Try Different Timeframes

```bash
timeframe = "4h"  # or "15m", "1d", etc.
```

### 3. Tune Hyperparameters

```python
model = MLP(
    input_size=input_size,
    output_size=5,
    hidden_layers=[256, 128, 64, 32],  # Deeper network
    dropout=0.4,                        # More regularization
    activation="gelu",                  # Different activation
)

trainer = Trainer(
    model,
    learning_rate=0.0005,  # Lower learning rate
    batch_size=128,        # Larger batches
    epochs=100,            # More epochs
)
```

### 4. Try Different Combination Strategies

```python
# Rule-based (default)
combiner = RuleBasedCombiner(num_classes=5)

# Weighted average
from tradeAI.models.ensemble import WeightedCombiner
combiner = WeightedCombiner(
    weights={'reversal': 0.4, 'continuation': 0.3, 'direction': 0.3}
)

# Meta-model (stacking)
from tradeAI.models.ensemble import MetaModelCombiner
combiner = MetaModelCombiner(num_classes=5)
combiner.fit(val_probs_reversal, val_probs_continuation, val_probs_direction, val_labels)
```

---

## Understanding the Output

### When You Train

```
================================================================================
ENSEMBLE TRAINING SUMMARY
================================================================================

Model 1 - Reversal:
  Parameters: 123,456
  Test Accuracy: 0.623
  Test Loss: 0.892

Model 2 - Continuation:
  Parameters: 123,456
  Test Accuracy: 0.611
  Test Loss: 0.905

Model 3 - Direction:
  Parameters: 123,456
  Test Accuracy: 0.597
  Test Loss: 0.918

Ensemble:
  Average Test Accuracy: 0.610
  Total Parameters: 370,368
```

### When You Generate Signals

```python
result = {
    'signal': np.array([1, -1, 0, ...]),      # Trading signal
    'confidence': np.array([0.82, 0.65, ...]), # Confidence score
    'agreement': np.array([0.91, 0.53, ...])   # Model agreement
}
```

**Interpretation:**
- `signal = 1`: Bullish
- `signal = -1`: Bearish
- `signal = 0`: Neutral
- `confidence`: How confident (0.0 - 1.0)
- `agreement`: How much models agree (0.0 - 1.0)

---

## Key Features

### ✅ No Forward-Looking Bias

- **Critical:** Data is split BEFORE normalization
- Features use ONLY past data
- Labels can use future data (for supervised learning)
- Systematic validation at every step

### ✅ Three-Model Ensemble

- **Reversal Model**: Identifies turning points (peaks/troughs)
- **Continuation Model**: Predicts trend persistence
- **Direction Model**: Provides baseline directional bias
- Combined predictions = robust signals

### ✅ Multiple Combination Strategies

1. **Rule-Based**: Trading logic (if reversal high AND continuation low → strong signal)
2. **Weighted**: Learned weights on validation data
3. **Meta-Model**: Stacking with second-level model

### ✅ Production-Ready

- Comprehensive logging
- Error handling
- Type hints throughout
- Modular design
- Well-documented code

---

## Troubleshooting

### Missing Dependencies?

```bash
pip install torch numpy pandas scikit-learn scipy tqdm yfinance loguru pyyaml
```

### GPU/CUDA Issues?

Models automatically detect GPU. For CPU-only:

```python
trainer = Trainer(model, device="cpu")
```

### Memory Issues?

Reduce batch size:

```python
trainer = Trainer(model, batch_size=32)  # or 16
```

### Data Download Fails?

Yahoo Finance sometimes rate-limits. Try:
- Reducing date range
- Adding delays between requests
- Using cached data

---

## Next Steps

1. **✅ Train Models**: `python scripts/train_ensemble_models.py`
2. **Evaluate Performance**: Check confusion matrices, metrics
3. **Backtest Strategies**: Test on historical data
4. **Deploy for Real-Time**: Integrate with live data feed
5. **Monitor Performance**: Track actual vs predicted

---

## Documentation

- **TRADEAI_ARCHITECTURE_PLAN.md** - Complete system design
- **FORWARD_LOOKING_BIAS_PREVENTION.md** - Critical: How we prevent data leakage
- **MULTI_MODEL_ENSEMBLE_STRATEGY.md** - Ensemble design and rationale
- **SESSION_SUMMARY.md** - Latest implementation session details
- **README_TRADEAI.md** - Full documentation

---

## Pro Tips

💡 **Always split data BEFORE normalizing** - Critical to prevent data leakage
💡 **High model agreement** = High confidence signals
💡 **Low agreement** = Stay cautious, use smaller positions
💡 **Accuracy >60%** on reversal prediction is excellent
💡 **Combine with risk management** for profitable trading

---

**Ready to start?**

```bash
# Train the ensemble!
python scripts/train_ensemble_models.py
```

The models are just **one command away**! 🚀
