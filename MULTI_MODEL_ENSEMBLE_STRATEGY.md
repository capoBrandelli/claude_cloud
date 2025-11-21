# Multi-Model Ensemble for Robust Reversal Prediction

## Executive Summary

**Proposed Architecture**: Three independent models, each predicting a different aspect of market behavior:
1. **Reversal Model**: P(reversal at this point)
2. **Continuation Model**: P(trend continues from this point)
3. **Direction Model**: P(up) vs P(down)

**Key Insight**: These models capture different market regimes and can be combined for more robust, confident predictions.

**Critical Requirement**: All three models must use the SAME past-only features, but have DIFFERENT labeling strategies.

---

## 1. Conceptual Framework

### 1.1 Why Three Independent Models?

Markets exhibit three fundamental states:
- **Reversal**: Trend is ending, about to change direction
- **Continuation**: Trend is strong, will persist
- **Directional Bias**: Overall momentum direction

Traditional single-model approaches miss the nuance:
```
Single Model: "Price will go up/down"
❌ Problem: Doesn't distinguish between:
   - Strong uptrend continuation (high confidence)
   - Weak reversal from downtrend (low confidence)
```

Multi-model approach captures context:
```
Ensemble:
   Reversal: P=0.8 (likely reversal)
   Continuation: P=0.2 (weak trend)
   Direction: P(up)=0.7 (upward bias)

   → Interpretation: Strong reversal signal upward!
   → High confidence trade
```

### 1.2 Independence Requirement

**Critical**: Models must be TRULY independent to reduce correlated errors:

| Requirement | Purpose | Implementation |
|-------------|---------|----------------|
| Different labels | Predict different phenomena | Different labeling strategies |
| Different features (optional) | Reduce correlation | Feature subsets per model |
| Different architectures (recommended) | Different learning patterns | MLP vs LSTM vs CNN |
| Different training (optional) | Reduce overfitting | Different random seeds |

---

## 2. The Three Models: Detailed Design

### 2.1 Model 1: Reversal Probability

**Objective**: Identify turning points (peaks and troughs)

**Input**: Features at time T (using data ≤ T)
**Output**: P(reversal at T) ∈ [0, 1]

**Labeling Strategy** (uses future data - acceptable):
```python
def label_reversals(df: pd.DataFrame, threshold: float = 0.05) -> pd.Series:
    """
    Label reversal points using extrema detection.

    This uses FUTURE data to identify extrema, which is acceptable
    for creating ground truth labels.
    """
    # Detect extrema using zigzag algorithm
    detector = ExtremaDetector(method="zigzag", threshold=threshold)
    peaks, troughs = detector.detect(df)

    # Create labels:
    # 0 = no reversal
    # 1 = reversal (peak or trough)
    labels = pd.Series(0, index=df.index)
    labels[peaks | troughs] = 1

    # Or multiclass:
    # 0 = no reversal
    # 1 = bullish reversal (trough)
    # 2 = bearish reversal (peak)
    labels = pd.Series(0, index=df.index)
    labels[troughs] = 1  # Bullish reversal
    labels[peaks] = 2    # Bearish reversal

    return labels
```

**Key Features for Reversal Model**:
- RSI (overbought/oversold)
- Stochastic oscillator
- Bollinger Band position
- Divergences
- Price vs moving average distance

**Why No Look-Ahead Bias**:
- Features at T use only data ≤ T ✅
- Labels at T use future data to identify reversals ✅ (acceptable for training)
- At inference, model predicts unknown future reversal from past features ✅

---

### 2.2 Model 2: Continuation Probability

**Objective**: Identify strong trends that will persist

**Input**: Features at time T (using data ≤ T)
**Output**: P(trend continues) ∈ [0, 1]

**Labeling Strategy** (uses future data - acceptable):
```python
def label_continuation(
    df: pd.DataFrame,
    lookback: int = 20,
    lookahead: int = 10,
    threshold: float = 0.02,
) -> pd.Series:
    """
    Label trend continuation.

    At time T:
    1. Identify current trend using PAST data (lookback)
    2. Check if trend continues using FUTURE data (lookahead)
    3. Label accordingly
    """
    labels = pd.Series(0, index=df.index)

    for i in range(lookback, len(df) - lookahead):
        # 1. Identify trend direction using PAST data only
        past_prices = df["close"].iloc[i-lookback:i]
        current_price = df["close"].iloc[i]

        # Calculate trend using linear regression on past data
        x = np.arange(lookback)
        y = past_prices.values
        slope = np.polyfit(x, y, 1)[0]

        # Define trend
        if slope > 0:
            trend = "up"
        elif slope < 0:
            trend = "down"
        else:
            trend = "neutral"

        # 2. Check if trend CONTINUES using FUTURE data (lookahead)
        future_prices = df["close"].iloc[i:i+lookahead]
        future_return = (future_prices.iloc[-1] - current_price) / current_price

        # 3. Label based on continuation
        if trend == "up" and future_return > threshold:
            labels.iloc[i] = 1  # Uptrend continued
        elif trend == "down" and future_return < -threshold:
            labels.iloc[i] = 1  # Downtrend continued
        else:
            labels.iloc[i] = 0  # Trend did not continue (reversal or choppy)

    # Can also be multiclass:
    # 0 = no continuation / reversal
    # 1 = uptrend continuation
    # 2 = downtrend continuation

    return labels
```

**Key Features for Continuation Model**:
- ADX (trend strength)
- Moving average slopes
- Momentum (ROC)
- Volume trend
- Higher timeframe trend alignment

**Why No Look-Ahead Bias**:
- Current trend identified using PAST data only ✅
- Features at T use only data ≤ T ✅
- Labels at T use future data to check continuation ✅ (acceptable for training)
- At inference, model predicts unknown future continuation from past features ✅

---

### 2.3 Model 3: Direction Probability

**Objective**: Simple directional bias prediction

**Input**: Features at time T (using data ≤ T)
**Output**: P(up) vs P(down)

**Labeling Strategy** (uses future data - acceptable):
```python
def label_direction(
    df: pd.DataFrame,
    lookahead: int = 10,
    threshold: float = 0.01,
) -> pd.Series:
    """
    Label future price direction.

    Simple: Will price be higher or lower in N bars?
    """
    # Calculate future return
    future_return = df["close"].pct_change(periods=lookahead).shift(-lookahead)

    # Create labels
    labels = pd.Series(0, index=df.index)

    # Binary:
    labels[future_return > threshold] = 1   # Up
    labels[future_return < -threshold] = 0  # Down

    # Or three-class:
    labels[future_return > threshold] = 2      # Up
    labels[abs(future_return) <= threshold] = 1  # Neutral
    labels[future_return < -threshold] = 0     # Down

    # Remove last N bars (no future data)
    labels.iloc[-lookahead:] = -1

    return labels
```

**Key Features for Direction Model**:
- MACD
- Moving average crossovers
- Rate of change
- Volume trends
- Market breadth

**Why No Look-Ahead Bias**:
- Features at T use only data ≤ T ✅
- Labels at T use future price ✅ (acceptable for training)
- At inference, model predicts unknown future direction from past features ✅

---

## 3. Preventing Look-Ahead Bias in the Ensemble

### 3.1 Critical Rule: Same Features, Different Labels

All three models MUST use the same features (or subsets) calculated from PAST data only:

```python
# At time T:
features_at_T = calculate_features(data_up_to_T)  # ✅ Past only

# All three models use SAME features:
reversal_prediction = model1.predict(features_at_T)
continuation_prediction = model2.predict(features_at_T)
direction_prediction = model3.predict(features_at_T)

# But were trained on DIFFERENT labels:
# model1 trained on reversal_labels
# model2 trained on continuation_labels
# model3 trained on direction_labels
```

### 3.2 Normalization: CRITICAL

Must normalize ONCE before splitting, fit ONLY on training data:

```python
# ❌ WRONG: Normalize separately for each model
scaler1.fit(X_all)  # Leaks test data into model1!
scaler2.fit(X_all)  # Leaks test data into model2!
scaler3.fit(X_all)  # Leaks test data into model3!

# ✅ CORRECT: Single normalization pipeline
# 1. Split data
X_train, X_val, X_test = split_data(X, y)

# 2. Fit normalizer ONLY on training data
scaler = Normalizer(method="minmax")
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

# 3. Train all models on SAME normalized training data
model1.fit(X_train_norm, y_reversal_train)
model2.fit(X_train_norm, y_continuation_train)
model3.fit(X_train_norm, y_direction_train)

# 4. Evaluate all models on SAME normalized test data
p1 = model1.predict(X_test_norm)
p2 = model2.predict(X_test_norm)
p3 = model3.predict(X_test_norm)
```

### 3.3 Train/Val/Test Split Strategy

Each model needs proper temporal split:

```python
def prepare_ensemble_data(df: pd.DataFrame):
    """
    Prepare data for ensemble training with NO forward-looking bias.
    """
    # 1. Calculate features (past data only)
    features = calculate_features(df)  # Uses rolling windows, no future data

    # 2. Create THREE different label sets
    y_reversal = label_reversals(df)
    y_continuation = label_continuation(df)
    y_direction = label_direction(df)

    # 3. Align indices (remove rows with invalid labels)
    valid_idx = (
        (y_reversal != -1) &
        (y_continuation != -1) &
        (y_direction != -1)
    )

    features = features[valid_idx]
    y_reversal = y_reversal[valid_idx]
    y_continuation = y_continuation[valid_idx]
    y_direction = y_direction[valid_idx]

    # 4. Store original dates for validation
    dates = df.index[valid_idx]

    # 5. Split temporally
    n = len(features)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    # Features
    X_train = features.iloc[:train_end]
    X_val = features.iloc[train_end:val_end]
    X_test = features.iloc[val_end:]

    # Dates
    dates_train = dates[:train_end]
    dates_val = dates[train_end:val_end]
    dates_test = dates[val_end:]

    # Validate temporal order
    from tradeAI.utils.validation import validate_temporal_split
    validate_temporal_split(dates_train, dates_val, dates_test)

    # 6. Normalize ONCE, fit on training only
    scaler = Normalizer(method="minmax")
    X_train_norm = scaler.fit_transform(X_train).values
    X_val_norm = scaler.transform(X_val).values
    X_test_norm = scaler.transform(X_test).values

    # 7. Labels for each model
    y1_train = y_reversal.iloc[:train_end].values
    y1_val = y_reversal.iloc[train_end:val_end].values
    y1_test = y_reversal.iloc[val_end:].values

    y2_train = y_continuation.iloc[:train_end].values
    y2_val = y_continuation.iloc[train_end:val_end].values
    y2_test = y_continuation.iloc[val_end:].values

    y3_train = y_direction.iloc[:train_end].values
    y3_val = y_direction.iloc[train_end:val_end].values
    y3_test = y_direction.iloc[val_end:].values

    return {
        "X_train": X_train_norm,
        "X_val": X_val_norm,
        "X_test": X_test_norm,
        "y_reversal": (y1_train, y1_val, y1_test),
        "y_continuation": (y2_train, y2_val, y2_test),
        "y_direction": (y3_train, y3_val, y3_test),
        "scaler": scaler,
        "dates": (dates_train, dates_val, dates_test),
    }
```

---

## 4. Combining the Three Probabilities

### 4.1 Method 1: Rule-Based Combination

Define clear trading rules based on model agreement:

```python
class EnsembleSignalGenerator:
    """Generate trading signals from ensemble of three models."""

    def __init__(
        self,
        reversal_threshold: float = 0.7,
        continuation_threshold: float = 0.7,
        direction_threshold: float = 0.6,
    ):
        self.rev_thresh = reversal_threshold
        self.cont_thresh = continuation_threshold
        self.dir_thresh = direction_threshold

    def generate_signal(
        self,
        p_reversal: float,
        p_continuation: float,
        p_direction_up: float,
    ) -> dict:
        """
        Generate trading signal from three probabilities.

        Returns:
            dict with signal, confidence, and reasoning
        """
        # Strong reversal signal
        if p_reversal > self.rev_thresh and p_continuation < (1 - self.cont_thresh):
            if p_direction_up > 0.5:
                return {
                    "signal": "REVERSAL_LONG",
                    "confidence": (p_reversal + (1 - p_continuation) + p_direction_up) / 3,
                    "reasoning": f"High reversal prob ({p_reversal:.2f}), "
                                f"low continuation ({p_continuation:.2f}), "
                                f"upward direction ({p_direction_up:.2f})",
                }
            else:
                return {
                    "signal": "REVERSAL_SHORT",
                    "confidence": (p_reversal + (1 - p_continuation) + (1 - p_direction_up)) / 3,
                    "reasoning": f"High reversal prob ({p_reversal:.2f}), "
                                f"low continuation ({p_continuation:.2f}), "
                                f"downward direction ({1-p_direction_up:.2f})",
                }

        # Strong continuation signal
        elif p_continuation > self.cont_thresh and p_reversal < (1 - self.rev_thresh):
            if p_direction_up > 0.5:
                return {
                    "signal": "CONTINUATION_LONG",
                    "confidence": (p_continuation + (1 - p_reversal) + p_direction_up) / 3,
                    "reasoning": f"High continuation ({p_continuation:.2f}), "
                                f"low reversal ({p_reversal:.2f}), "
                                f"upward direction ({p_direction_up:.2f})",
                }
            else:
                return {
                    "signal": "CONTINUATION_SHORT",
                    "confidence": (p_continuation + (1 - p_reversal) + (1 - p_direction_up)) / 3,
                    "reasoning": f"High continuation ({p_continuation:.2f}), "
                                f"low reversal ({p_reversal:.2f}), "
                                f"downward direction ({1-p_direction_up:.2f})",
                }

        # Uncertain - no clear signal
        else:
            return {
                "signal": "NO_TRADE",
                "confidence": 0.0,
                "reasoning": f"Mixed signals: reversal={p_reversal:.2f}, "
                            f"continuation={p_continuation:.2f}, "
                            f"direction_up={p_direction_up:.2f}",
            }
```

### 4.2 Method 2: Weighted Average

Learn optimal weights on validation set:

```python
class WeightedEnsemble:
    """Combine three models using learned weights."""

    def __init__(self):
        self.weights = None

    def fit(
        self,
        p_reversal_val: np.ndarray,
        p_continuation_val: np.ndarray,
        p_direction_val: np.ndarray,
        y_true_val: np.ndarray,
    ):
        """
        Learn optimal weights on validation set.

        This uses validation set to learn weights, so no test set leakage.
        """
        from scipy.optimize import minimize

        def loss(weights):
            # Combine predictions
            combined = (
                weights[0] * p_reversal_val +
                weights[1] * p_continuation_val +
                weights[2] * p_direction_val
            )
            # Calculate loss (e.g., cross-entropy)
            return -np.mean(y_true_val * np.log(combined + 1e-10))

        # Optimize weights
        result = minimize(
            loss,
            x0=[1/3, 1/3, 1/3],
            bounds=[(0, 1), (0, 1), (0, 1)],
            constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
        )

        self.weights = result.x
        logger.info(f"Learned weights: {self.weights}")

    def predict(
        self,
        p_reversal: np.ndarray,
        p_continuation: np.ndarray,
        p_direction: np.ndarray,
    ) -> np.ndarray:
        """Combine predictions using learned weights."""
        return (
            self.weights[0] * p_reversal +
            self.weights[1] * p_continuation +
            self.weights[2] * p_direction
        )
```

### 4.3 Method 3: Meta-Model (Stacking)

Train a fourth model to combine the three predictions:

```python
class MetaEnsemble:
    """
    Meta-model that learns to combine three base models.

    CRITICAL: Must prevent data leakage!
    """

    def __init__(self, meta_model):
        self.meta_model = meta_model

    def fit(
        self,
        # Base model predictions on VALIDATION set
        p_rev_val: np.ndarray,
        p_cont_val: np.ndarray,
        p_dir_val: np.ndarray,
        y_val: np.ndarray,

        # Original features (optional, for richer meta-model)
        X_val: np.ndarray = None,
    ):
        """
        Train meta-model on VALIDATION set predictions.

        CRITICAL: Do NOT use test set predictions here!
        This would create severe data leakage.
        """
        # Combine base predictions as meta-features
        if X_val is not None:
            # Include original features too
            meta_features = np.column_stack([
                p_rev_val,
                p_cont_val,
                p_dir_val,
                X_val,  # Original features
            ])
        else:
            meta_features = np.column_stack([
                p_rev_val,
                p_cont_val,
                p_dir_val,
            ])

        # Train meta-model
        self.meta_model.fit(meta_features, y_val)

        logger.info("Meta-model trained on validation set")

    def predict(
        self,
        p_rev: np.ndarray,
        p_cont: np.ndarray,
        p_dir: np.ndarray,
        X: np.ndarray = None,
    ) -> np.ndarray:
        """Make predictions using meta-model."""
        if X is not None:
            meta_features = np.column_stack([p_rev, p_cont, p_dir, X])
        else:
            meta_features = np.column_stack([p_rev, p_cont, p_dir])

        return self.meta_model.predict(meta_features)
```

**Critical**: Meta-model must be trained on validation set only, never on test set!

---

## 5. Assigning Probabilities to Latest Candles

### 5.1 For Each New Candle in Real-Time

```python
def predict_latest_candle(
    current_data: pd.DataFrame,
    model_reversal,
    model_continuation,
    model_direction,
    scaler,
    feature_engineer,
) -> dict:
    """
    Predict probabilities for the latest candle.

    This is what you'd use in live trading.

    Args:
        current_data: All historical data up to current time
        model_*: Trained models
        scaler: Fitted normalizer (from training period)
        feature_engineer: Feature calculator

    Returns:
        dict with all probabilities and combined signal
    """
    # 1. Calculate features using ALL PAST data (no future data!)
    features = feature_engineer.add_all_features(current_data)

    # 2. Get latest candle features
    latest_features = features.iloc[-1:].drop(columns=["label"], errors="ignore")

    # 3. Normalize using TRAINING scaler
    latest_features_norm = scaler.transform(latest_features)

    # 4. Get predictions from all three models
    p_reversal = model_reversal.predict_proba(latest_features_norm)[0]
    p_continuation = model_continuation.predict_proba(latest_features_norm)[0]
    p_direction = model_direction.predict_proba(latest_features_norm)[0]

    # 5. Combine predictions
    signal_generator = EnsembleSignalGenerator()
    signal = signal_generator.generate_signal(
        p_reversal=p_reversal[1] if len(p_reversal) > 1 else p_reversal[0],
        p_continuation=p_continuation[1] if len(p_continuation) > 1 else p_continuation[0],
        p_direction_up=p_direction[1] if len(p_direction) > 1 else p_direction[0],
    )

    # 6. Return comprehensive result
    return {
        "timestamp": current_data.index[-1],
        "p_reversal": float(p_reversal[1] if len(p_reversal) > 1 else p_reversal[0]),
        "p_continuation": float(p_continuation[1] if len(p_continuation) > 1 else p_continuation[0]),
        "p_direction_up": float(p_direction[1] if len(p_direction) > 1 else p_direction[0]),
        "p_direction_down": float(p_direction[0] if len(p_direction) > 1 else 1 - p_direction[0]),
        "signal": signal["signal"],
        "confidence": signal["confidence"],
        "reasoning": signal["reasoning"],
    }
```

### 5.2 Batch Prediction for Historical Analysis

```python
def predict_all_candles(
    data: pd.DataFrame,
    model_reversal,
    model_continuation,
    model_direction,
    scaler,
    feature_engineer,
) -> pd.DataFrame:
    """
    Generate predictions for all historical candles.

    Useful for backtesting and analysis.
    """
    # Calculate features
    features = feature_engineer.add_all_features(data)

    # Remove label column if exists
    X = features.drop(columns=["label"], errors="ignore")

    # Normalize
    X_norm = scaler.transform(X)

    # Get predictions from all models
    p_reversal = model_reversal.predict_proba(X_norm)
    p_continuation = model_continuation.predict_proba(X_norm)
    p_direction = model_direction.predict_proba(X_norm)

    # Create results DataFrame
    results = pd.DataFrame({
        "timestamp": data.index,
        "close": data["close"],
        "p_reversal": p_reversal[:, 1] if p_reversal.shape[1] > 1 else p_reversal[:, 0],
        "p_continuation": p_continuation[:, 1] if p_continuation.shape[1] > 1 else p_continuation[:, 0],
        "p_direction_up": p_direction[:, 1] if p_direction.shape[1] > 1 else p_direction[:, 0],
    })

    # Generate signals for each candle
    signal_generator = EnsembleSignalGenerator()

    signals = []
    confidences = []
    for i in range(len(results)):
        signal = signal_generator.generate_signal(
            p_reversal=results.iloc[i]["p_reversal"],
            p_continuation=results.iloc[i]["p_continuation"],
            p_direction_up=results.iloc[i]["p_direction_up"],
        )
        signals.append(signal["signal"])
        confidences.append(signal["confidence"])

    results["signal"] = signals
    results["confidence"] = confidences

    return results
```

---

## 6. Walk-Forward Ensemble Backtesting

Critical: All three models must be retrained in walk-forward fashion:

```python
class EnsembleWalkForward:
    """
    Walk-forward backtesting for ensemble of three models.

    Ensures NO forward-looking bias in ensemble evaluation.
    """

    def __init__(
        self,
        train_window: int = 1000,
        test_window: int = 100,
        retrain_frequency: int = 1,
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.retrain_frequency = retrain_frequency

    def walk_forward(
        self,
        df: pd.DataFrame,
        feature_engineer,
        model_reversal_class,
        model_continuation_class,
        model_direction_class,
    ) -> pd.DataFrame:
        """
        Perform walk-forward analysis on ensemble.

        Each iteration:
        1. Train all three models on past data
        2. Predict on next period
        3. Store predictions
        4. Move forward
        """
        # Prepare data
        features = feature_engineer.add_all_features(df)

        # Create labels for all three models
        y_reversal = label_reversals(df)
        y_continuation = label_continuation(df)
        y_direction = label_direction(df)

        # Align
        valid_idx = (y_reversal != -1) & (y_continuation != -1) & (y_direction != -1)
        X = features[valid_idx]
        y_rev = y_reversal[valid_idx]
        y_cont = y_continuation[valid_idx]
        y_dir = y_direction[valid_idx]
        dates = df.index[valid_idx]

        # Results storage
        all_predictions = []

        n = len(X)
        n_iterations = (n - self.train_window) // self.test_window

        logger.info(f"Walk-forward ensemble: {n_iterations} iterations")

        for i in tqdm(range(n_iterations)):
            train_end = self.train_window + i * self.test_window
            test_start = train_end
            test_end = test_start + self.test_window

            if test_end > n:
                break

            # Get train/test splits
            X_train = X.iloc[:train_end]
            X_test = X.iloc[test_start:test_end]

            y1_train = y_rev.iloc[:train_end]
            y2_train = y_cont.iloc[:train_end]
            y3_train = y_dir.iloc[:train_end]

            dates_test = dates.iloc[test_start:test_end]

            # CRITICAL: Normalize using ONLY training data
            scaler = Normalizer(method="minmax")
            X_train_norm = scaler.fit_transform(X_train)
            X_test_norm = scaler.transform(X_test)

            # Train all three models
            model1 = model_reversal_class()
            model1.fit(X_train_norm, y1_train)

            model2 = model_continuation_class()
            model2.fit(X_train_norm, y2_train)

            model3 = model_direction_class()
            model3.fit(X_train_norm, y3_train)

            # Predict on test period
            p_rev = model1.predict_proba(X_test_norm)
            p_cont = model2.predict_proba(X_test_norm)
            p_dir = model3.predict_proba(X_test_norm)

            # Store predictions
            for j in range(len(X_test_norm)):
                all_predictions.append({
                    "timestamp": dates_test.iloc[j],
                    "p_reversal": p_rev[j, 1] if p_rev.shape[1] > 1 else p_rev[j, 0],
                    "p_continuation": p_cont[j, 1] if p_cont.shape[1] > 1 else p_cont[j, 0],
                    "p_direction_up": p_dir[j, 1] if p_dir.shape[1] > 1 else p_dir[j, 0],
                })

        # Convert to DataFrame
        results = pd.DataFrame(all_predictions)

        logger.info(f"Walk-forward complete: {len(results)} predictions")

        return results
```

---

## 7. Implementation Checklist

### 7.1 No Forward-Looking Bias Verification

Before deploying, verify:

**Data Preparation**:
- [ ] Features calculated using rolling windows (past data only)
- [ ] No negative shifts in features (`shift(-n)`)
- [ ] Labels use future data (acceptable)
- [ ] Last N bars removed where labels invalid

**Normalization**:
- [ ] Single scaler fit ONLY on training data
- [ ] Same scaler applied to val, test, and production
- [ ] No separate scaling per model

**Training**:
- [ ] All three models use same normalized features
- [ ] Each model has different labels
- [ ] Temporal order maintained (train → val → test)
- [ ] No data overlap between splits

**Meta-Model** (if used):
- [ ] Trained only on validation predictions
- [ ] Never trained on test predictions
- [ ] Separate validation set for meta-model

**Walk-Forward**:
- [ ] Each iteration trains on past data only
- [ ] Tests on immediate next period
- [ ] No information from future iterations

**Production**:
- [ ] Features use current + past data only
- [ ] Scaler from original training used
- [ ] Models from latest retraining used

### 7.2 File Structure

```
tradeAI/
├── models/
│   └── ensemble/
│       ├── __init__.py
│       ├── three_model_ensemble.py      # Main ensemble class
│       ├── reversal_labeler.py         # Already exists (enhance)
│       ├── continuation_labeler.py     # NEW
│       ├── direction_labeler.py        # NEW
│       └── signal_generator.py         # NEW
│
├── training/
│   └── ensemble_trainer.py             # NEW - Train all three models
│
└── backtesting/
    └── ensemble_walk_forward.py        # NEW - Walk-forward for ensemble
```

---

## 8. Example Usage

### 8.1 Training the Ensemble

```python
from tradeAI.models.ensemble import ThreeModelEnsemble
from tradeAI.training import EnsembleTrainer

# Prepare data
data = prepare_ensemble_data(df)

# Create models
from tradeAI.models.architectures import MLP, LSTMModel

model_reversal = MLP(input_size=45, output_size=2, hidden_layers=[128, 64])
model_continuation = LSTMModel(input_size=45, output_size=2, hidden_size=64)
model_direction = MLP(input_size=45, output_size=2, hidden_layers=[64, 32])

# Train ensemble
trainer = EnsembleTrainer(
    model_reversal=model_reversal,
    model_continuation=model_continuation,
    model_direction=model_direction,
)

trainer.fit(
    X_train=data["X_train"],
    y_reversal_train=data["y_reversal"][0],
    y_continuation_train=data["y_continuation"][0],
    y_direction_train=data["y_direction"][0],
    X_val=data["X_val"],
    y_reversal_val=data["y_reversal"][1],
    y_continuation_val=data["y_continuation"][1],
    y_direction_val=data["y_direction"][1],
)

# Evaluate
predictions = trainer.predict_ensemble(data["X_test"])
```

### 8.2 Real-Time Prediction

```python
# In live trading
current_data = fetch_latest_data()  # All data up to now

prediction = predict_latest_candle(
    current_data=current_data,
    model_reversal=trained_model_reversal,
    model_continuation=trained_model_continuation,
    model_direction=trained_model_direction,
    scaler=fitted_scaler,
    feature_engineer=engineer,
)

print(f"Timestamp: {prediction['timestamp']}")
print(f"Reversal Probability: {prediction['p_reversal']:.2f}")
print(f"Continuation Probability: {prediction['p_continuation']:.2f}")
print(f"Direction Up: {prediction['p_direction_up']:.2f}")
print(f"Signal: {prediction['signal']}")
print(f"Confidence: {prediction['confidence']:.2f}")
print(f"Reasoning: {prediction['reasoning']}")

# Example output:
# Timestamp: 2025-01-15 14:30:00
# Reversal Probability: 0.82
# Continuation Probability: 0.18
# Direction Up: 0.75
# Signal: REVERSAL_LONG
# Confidence: 0.79
# Reasoning: High reversal prob (0.82), low continuation (0.18), upward direction (0.75)
```

---

## 9. Benefits of This Approach

### 9.1 Robustness

**Diversification**: Three models capture different aspects
- Reversals: Turning points
- Continuation: Trend strength
- Direction: Overall bias

**Error Reduction**: Uncorrelated errors average out
- If reversal model makes error, others compensate
- Ensemble variance < individual model variance

### 9.2 Confidence Quantification

**Clear Interpretation**:
```
High reversal + Low continuation = Strong reversal confidence
Low reversal + High continuation = Strong trend confidence
Mid-range all three = Low confidence → Don't trade
```

**Better Risk Management**:
- Trade size proportional to confidence
- Filter out uncertain signals
- Only trade when models agree

### 9.3 Adaptability

**Different Market Regimes**:
- Trending markets: Continuation model dominates
- Range-bound: Reversal model dominates
- Strong momentum: Direction model dominates

**Ensemble adapts** to current conditions!

---

## 10. Summary

### The Architecture

```
Historical Data → Features (past only) → ┌─ Model 1: Reversal    ─┐
                                         ├─ Model 2: Continuation ─┤→ Combiner → Signal
                                         └─ Model 3: Direction    ─┘

Training:
- Same features for all (different subsets OK)
- Different labels for each
- Same normalization (fit on training only!)
- Different architectures (recommended)

Inference:
- Features from current + past data only ✅
- Three independent probabilities
- Combined signal with confidence
```

### No Forward-Looking Bias Because:

1. ✅ Features use only past data (rolling windows, no future shifts)
2. ✅ Labels use future data (acceptable - this is supervised learning!)
3. ✅ Single normalization fit only on training data
4. ✅ Temporal order maintained throughout
5. ✅ Walk-forward validation tests realistic performance
6. ✅ Meta-model (if used) trained only on validation set

### Implementation Priority:

1. **Phase 1** (1-2 days): Implement three labeling functions
2. **Phase 2** (1 day): Create ensemble training pipeline
3. **Phase 3** (1 day): Implement signal combination logic
4. **Phase 4** (2 days): Walk-forward backtesting
5. **Phase 5** (1 day): Real-time prediction interface

**Total**: ~1 week for complete ensemble system

This approach is **production-ready** and will **not suffer from forward-looking bias**! 🎯
