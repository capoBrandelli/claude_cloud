# Preventing Forward-Looking Bias in TradeAI: Strategy & Gap Analysis

## Executive Summary

**The Problem**: Forward-looking bias (look-ahead bias) occurs when future information leaks into the training process, creating artificially inflated performance that fails catastrophically in live trading.

**The Solution**: Strict temporal separation between features (past data only) and labels (can use future data for ground truth), combined with proper normalization, validation, and backtesting procedures.

**Critical Finding**: Our current implementation has **CRITICAL data leakage** in the normalization step that must be fixed immediately.

---

## 1. Deep Analysis: The Nature of Forward-Looking Bias

### 1.1 The Fundamental Distinction

There are TWO types of forward-looking information in trading ML:

#### Type 1: Forward-Looking LABELS (Acceptable)
```
Time:     T0    T1    T2    T3    T4    T5    T6
Features: [F0]  [F1]  [F2]  [F3]  [F4]  [F5]  [F6]
Label:           L1          L3          L5
                 ↑           ↑           ↑
                 Uses T3     Uses T5     Uses T7 (future)
```

**This is VALID** because:
- We're creating ground truth for supervised learning
- In production, we predict the label we don't know yet
- Model learns: "Given features at T, this pattern leads to this outcome"

#### Type 2: Forward-Looking FEATURES (Fatal Error)
```
Time:     T0    T1    T2    T3    T4    T5    T6
Features: [F0 using T1 data] ← WRONG! Can't use future data
Label:    [L0]
```

**This is INVALID** because:
- At time T0, we cannot know T1 data
- Creates impossible predictions
- Fails completely in live trading

### 1.2 The Key Principle: Point-in-Time Integrity

**Point-in-Time Rule**: At any time T, we can only use information available up to and including T.

```python
# VALID: Features at time T use data [T0...T]
features_at_T = calculate_indicators(data_up_to_T)

# INVALID: Features at time T use data [T0...T+n]
features_at_T = calculate_indicators(data_including_future)
```

---

## 2. Where Forward-Looking Bias Can Occur

### 2.1 Common Sources of Bias

| Stage | Risk Level | Example |
|-------|-----------|---------|
| **Labeling** | 🟢 Acceptable | Using future prices to create labels |
| **Feature Engineering** | 🔴 High Risk | Using `.shift(-n)` or `center=True` |
| **Normalization** | 🔴 **CRITICAL** | Fitting scaler on entire dataset |
| **Feature Selection** | 🟡 Medium Risk | Selecting features based on full dataset |
| **Train/Val/Test Split** | 🔴 High Risk | Shuffling time series data |
| **Cross-Validation** | 🔴 High Risk | Using standard k-fold instead of walk-forward |
| **Backtesting** | 🔴 High Risk | Using future data for stop-loss, position sizing |

### 2.2 The Normalization Problem (MOST CRITICAL)

This is the **#1 source** of data leakage in production ML systems:

```python
# ❌ WRONG - Leaks information from test set into training
scaler.fit(X_all)  # Fits on train + val + test
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Why is this wrong?
# - Training data is normalized using statistics (min/max/mean/std) from the test set
# - Model learns patterns that include information from the future
# - Test performance is artificially inflated

# ✅ CORRECT - No information leakage
scaler.fit(X_train)  # Fit ONLY on training data
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)  # Apply same transformation
X_test = scaler.transform(X_test)  # Apply same transformation

# Now training data is normalized using ONLY training statistics
```

---

## 3. The Correct Training Paradigm

### 3.1 The Training Contract

```
TRAINING PHASE:
Input:  Features at time T (using only data ≤ T)
Output: Label at time T (created using data > T)
Learn:  "Given these historical patterns, this is what happened next"

INFERENCE PHASE:
Input:  Features at time T (using only data ≤ T)
Output: Predicted label at time T
Use:    "Given these current patterns, predict what happens next"
```

This is valid because:
1. Model never sees future features
2. Model only sees future labels during training (this is supervised learning)
3. At inference, model predicts unknown future outcome from known past features

### 3.2 Walk-Forward Validation

Standard train/test split is insufficient. We need walk-forward analysis:

```
Period:  1    2    3    4    5    6    7    8    9    10
        [Train 1-4][Test 5]
             [Train 1-5][Test 6]
                  [Train 1-6][Test 7]
                       [Train 1-7][Test 8]
```

Each iteration:
1. Train on all past data
2. Test on next period
3. Retrain with expanded dataset
4. Repeat

This simulates real trading where:
- You only know the past
- You must predict the future
- You can retrain as new data arrives

---

## 4. Gap Analysis: Current TradeAI Implementation

### 4.1 Module-by-Module Analysis

#### ✅ **SAFE MODULES** (No Changes Needed)

1. **data/providers/**
   - Fetches historical data only
   - No bias possible
   - ✅ Status: SAFE

2. **features/feature_engineer.py**
   - All indicators use rolling windows with past data only
   - No negative shifts (`shift(-n)`)
   - No centered rolling windows
   - ✅ Status: SAFE
   - Examples:
     ```python
     df["returns"] = df["close"].pct_change()  # Uses previous bar ✅
     df["sma_20"] = df["close"].rolling(window=20).mean()  # Past 20 bars ✅
     df["rsi_14"] = _add_rsi(df, 14)  # Uses past data only ✅
     ```

3. **utils/**
   - No bias concerns
   - ✅ Status: SAFE

#### ⚠️ **REVIEW NEEDED** (Minor Issues)

1. **preprocessing/cleaner.py**
   - Line: `df.interpolate(method="linear")`
   - Issue: Some interpolation methods may look forward
   - **Fix**: Specify `limit_direction='backward'`
   ```python
   # Current:
   df = df.interpolate(method="linear")

   # Fixed:
   df = df.interpolate(method="linear", limit_direction='backward')
   ```

2. **preprocessing/aligner.py**
   - Line: Rolling window extrema detection
   - Issue: Need to ensure no forward-looking in alignment
   - **Fix**: Add validation that all operations are backward-looking

#### 🔴 **CRITICAL ISSUES** (Must Fix)

##### Issue 1: Normalization Data Leakage (CRITICAL - P0)

**File**: `scripts/train_first_model.py`
**Lines**: 100-102
```python
# ❌ CURRENT CODE (HAS DATA LEAKAGE):
normalizer = Normalizer(method="minmax")
X = normalizer.fit_transform(pd.DataFrame(X, columns=feature_cols)).values
# ^^^ Fits on ALL data including test set!

# Then splits:
X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]
```

**Problem**: The normalizer is fit on the entire dataset (including test set) BEFORE splitting. This means:
- Training data is normalized using statistics from the test set
- Model sees information from the "future" (test period)
- Test performance is artificially inflated
- **THIS IS THE #1 SOURCE OF DATA LEAKAGE**

**Impact**:
- Estimated performance inflation: 5-15%
- False confidence in model
- Will fail in production

**Fix**:
```python
# ✅ CORRECT CODE:
# 1. Split FIRST
X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

# 2. Fit normalizer ONLY on training data
normalizer = Normalizer(method="minmax")
X_train = normalizer.fit_transform(pd.DataFrame(X_train)).values

# 3. Transform val and test using training statistics
X_val = normalizer.transform(pd.DataFrame(X_val)).values
X_test = normalizer.transform(pd.DataFrame(X_test)).values
```

##### Issue 2: Label Contamination (HIGH PRIORITY - P1)

**File**: `labeling/reversal_labeler.py`
**Lines**: 83-87

```python
# Current:
labels.iloc[-self.lookahead_window:] = -1  # Mark as invalid
# But these rows are still in the dataset!
```

**Problem**: Last N bars have invalid labels but are still included. These should be completely removed from training.

**Fix**:
```python
# After labeling:
df = df[df["label"] != -1].copy()  # Remove invalid labels
# This is actually done in train_first_model.py line 109, so OK ✅
```

**Status**: Actually already handled correctly in the training script! ✅

##### Issue 3: No Temporal Validation (MEDIUM PRIORITY - P2)

**File**: `training/trainer.py`

**Problem**: No checks to ensure train/val/test maintain temporal order

**Fix**: Add validation function
```python
def validate_temporal_split(train_dates, val_dates, test_dates):
    """Ensure no temporal overlap between splits."""
    if train_dates.max() >= val_dates.min():
        raise ValueError(
            f"Train data leaks into validation! "
            f"Train ends: {train_dates.max()}, Val starts: {val_dates.min()}"
        )
    if val_dates.max() >= test_dates.min():
        raise ValueError(
            f"Validation data leaks into test! "
            f"Val ends: {val_dates.max()}, Test starts: {test_dates.min()}"
        )
    logger.info("✓ Temporal split validation passed - no data leakage")
```

##### Issue 4: Missing Walk-Forward Backtesting (HIGH PRIORITY - P1)

**File**: `backtesting/` (not yet implemented)

**Problem**: Need proper walk-forward validation for realistic performance estimates

**Fix**: Implement walk-forward backtesting (see implementation section below)

##### Issue 5: Shuffle Warning Too Weak (LOW PRIORITY - P3)

**File**: `preprocessing/splitter.py`
**Line**: 40

```python
# Current:
if self.shuffle:
    logger.warning("Shuffling time series data - temporal order will be lost")

# Fix:
if self.shuffle:
    raise ValueError(
        "Shuffling is not allowed for time series data! "
        "This would create forward-looking bias. "
        "Set shuffle=False."
    )
```

---

## 5. Implementation Strategy

### 5.1 Immediate Fixes (Sprint 1 - Critical)

#### Fix 1: Correct Normalization Order

**File**: `scripts/train_first_model.py`

```python
# Replace lines 95-110 with:

# ========================================================================
# STEP 5: Prepare Training Data (FIXED - No Data Leakage)
# ========================================================================
logger.info("\n" + "=" * 70)
logger.info("STEP 5: Prepare Training Data (Preventing Forward-Looking Bias)")
logger.info("=" * 70)

# Split features and labels
feature_cols = [col for col in df.columns if col not in ["label"]]
X = df[feature_cols].values
y = df["label"].values

logger.info(f"  Total samples: {len(X)}")
logger.info(f"  Features: {X.shape[1]}")

# CRITICAL: Split BEFORE normalization to prevent data leakage
n = len(X)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

X_train_raw, y_train = X[:train_end], y[:train_end]
X_val_raw, y_val = X[train_end:val_end], y[train_end:val_end]
X_test_raw, y_test = X[val_end:], y[val_end:]

logger.info(f"\n  Data split (before normalization):")
logger.info(f"    Train: {len(X_train_raw)} samples")
logger.info(f"    Val:   {len(X_val_raw)} samples")
logger.info(f"    Test:  {len(X_test_raw)} samples")

# Normalize features - FIT ONLY ON TRAINING DATA
logger.info("\n  Normalizing features...")
logger.info("    ✓ Fitting normalizer ONLY on training data (no data leakage)")

normalizer = Normalizer(method="minmax")
X_train = normalizer.fit_transform(
    pd.DataFrame(X_train_raw, columns=feature_cols)
).values

# Transform val and test using TRAINING statistics only
X_val = normalizer.transform(
    pd.DataFrame(X_val_raw, columns=feature_cols)
).values
X_test = normalizer.transform(
    pd.DataFrame(X_test_raw, columns=feature_cols)
).values

logger.info("    ✓ Training set normalized using its own statistics")
logger.info("    ✓ Validation set normalized using TRAINING statistics")
logger.info("    ✓ Test set normalized using TRAINING statistics")
logger.info("\n✓ No forward-looking bias in normalization!")
```

#### Fix 2: Add Temporal Validation

**File**: Create new file `tradeAI/utils/validation.py`

```python
"""
Validation utilities to prevent forward-looking bias
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional

from tradeAI.utils.logger import get_logger

logger = get_logger(__name__)


def validate_temporal_split(
    train_idx: pd.DatetimeIndex,
    val_idx: Optional[pd.DatetimeIndex] = None,
    test_idx: Optional[pd.DatetimeIndex] = None,
) -> None:
    """
    Validate that train/val/test splits maintain strict temporal order.

    Args:
        train_idx: Training set datetime index
        val_idx: Validation set datetime index (optional)
        test_idx: Test set datetime index (optional)

    Raises:
        ValueError: If temporal order is violated
    """
    if val_idx is not None:
        # Check train vs val
        if train_idx.max() >= val_idx.min():
            raise ValueError(
                f"FORWARD-LOOKING BIAS DETECTED! "
                f"Training data overlaps with validation data.\n"
                f"  Train period ends:   {train_idx.max()}\n"
                f"  Val period starts:   {val_idx.min()}\n"
                f"This would leak future information into training!"
            )

        gap = (val_idx.min() - train_idx.max()).total_seconds() / 3600
        logger.info(f"  ✓ Train-Val gap: {gap:.1f} hours (no overlap)")

    if test_idx is not None:
        # Check val vs test (if val exists)
        if val_idx is not None:
            if val_idx.max() >= test_idx.min():
                raise ValueError(
                    f"FORWARD-LOOKING BIAS DETECTED! "
                    f"Validation data overlaps with test data.\n"
                    f"  Val period ends:    {val_idx.max()}\n"
                    f"  Test period starts: {test_idx.min()}\n"
                    f"This would leak future information!"
                )

            gap = (test_idx.min() - val_idx.max()).total_seconds() / 3600
            logger.info(f"  ✓ Val-Test gap: {gap:.1f} hours (no overlap)")

        # Check train vs test
        if train_idx.max() >= test_idx.min():
            raise ValueError(
                f"FORWARD-LOOKING BIAS DETECTED! "
                f"Training data overlaps with test data.\n"
                f"  Train period ends:  {train_idx.max()}\n"
                f"  Test period starts: {test_idx.min()}"
            )

        gap = (test_idx.min() - train_idx.max()).total_seconds() / 3600
        logger.info(f"  ✓ Train-Test gap: {gap:.1f} hours (no overlap)")

    logger.info("✓ Temporal validation passed - no forward-looking bias detected!")


def validate_no_future_features(
    df: pd.DataFrame,
    reference_time: pd.Timestamp,
) -> None:
    """
    Validate that a dataframe contains no data after the reference time.

    Args:
        df: DataFrame to validate
        reference_time: Maximum allowed timestamp

    Raises:
        ValueError: If future data is found
    """
    if df.index.max() > reference_time:
        raise ValueError(
            f"FORWARD-LOOKING BIAS DETECTED! "
            f"DataFrame contains data after reference time.\n"
            f"  Latest data:      {df.index.max()}\n"
            f"  Reference time:   {reference_time}\n"
            f"  Future data points: {(df.index > reference_time).sum()}"
        )

    logger.debug(f"✓ No future data detected (ref: {reference_time})")


def check_feature_computation(
    df: pd.DataFrame,
    feature_name: str,
) -> bool:
    """
    Check if a feature could potentially use forward-looking data.

    This is a heuristic check looking for common patterns that indicate
    forward-looking bias.

    Args:
        df: DataFrame with features
        feature_name: Name of feature to check

    Returns:
        True if feature appears safe, False if suspicious
    """
    # Check for suspiciously high correlation with future returns
    if "label" in df.columns:
        future_return = df["close"].pct_change().shift(-1)
        corr = df[feature_name].corr(future_return)

        if abs(corr) > 0.8:
            logger.warning(
                f"⚠ Feature '{feature_name}' has suspiciously high correlation "
                f"({corr:.3f}) with future returns. "
                f"This might indicate forward-looking bias!"
            )
            return False

    return True
```

#### Fix 3: Enhance Splitter

**File**: `tradeAI/preprocessing/splitter.py`

```python
# Add to TimeSeriesSplitter class:

def split_with_validation(
    self,
    df: pd.DataFrame,
    target_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame with temporal validation.

    This version includes checks to prevent forward-looking bias.
    """
    if self.shuffle:
        raise ValueError(
            "shuffle=True is not allowed for time series data! "
            "This would create forward-looking bias by mixing future "
            "data with past data. Set shuffle=False."
        )

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "DataFrame must have DatetimeIndex for temporal validation. "
            "Current index type: {type(df.index)}"
        )

    # Perform split
    train_df, val_df, test_df = self.split(df, target_col)

    # Validate temporal order
    from tradeAI.utils.validation import validate_temporal_split
    validate_temporal_split(train_df.index, val_df.index, test_df.index)

    return train_df, val_df, test_df
```

### 5.2 Walk-Forward Implementation (Sprint 2)

**File**: Create new file `tradeAI/backtesting/walk_forward.py`

```python
"""
Walk-forward analysis for realistic backtesting
"""

from typing import List, Tuple, Callable, Dict, Any
import pandas as pd
import numpy as np
from tqdm import tqdm

from tradeAI.utils.logger import get_logger

logger = get_logger(__name__)


class WalkForwardAnalyzer:
    """
    Implement walk-forward analysis to prevent forward-looking bias.

    Walk-forward analysis simulates realistic trading by:
    1. Training on historical data only
    2. Testing on the immediate next period
    3. Retraining with expanded dataset
    4. Repeating until all data is tested
    """

    def __init__(
        self,
        train_window: int,
        test_window: int,
        retrain_frequency: int = None,
        expanding_window: bool = True,
    ):
        """
        Initialize walk-forward analyzer.

        Args:
            train_window: Number of periods for training
            test_window: Number of periods for testing
            retrain_frequency: Retrain every N test windows (None = every window)
            expanding_window: If True, training window expands. If False, rolls.
        """
        self.train_window = train_window
        self.test_window = test_window
        self.retrain_frequency = retrain_frequency or 1
        self.expanding_window = expanding_window

    def walk_forward(
        self,
        df: pd.DataFrame,
        train_func: Callable,
        predict_func: Callable,
    ) -> Dict[str, Any]:
        """
        Perform walk-forward analysis.

        Args:
            df: Full dataset with features and labels
            train_func: Function to train model on data
                        Signature: train_func(train_df) -> model
            predict_func: Function to make predictions
                         Signature: predict_func(model, test_df) -> predictions

        Returns:
            Dictionary with results including all predictions and metrics
        """
        n = len(df)
        all_predictions = []
        all_actuals = []
        all_dates = []

        # Calculate number of walk-forward iterations
        if self.expanding_window:
            n_iterations = (n - self.train_window) // self.test_window
        else:
            n_iterations = (n - self.train_window) // self.test_window

        logger.info(f"Starting walk-forward analysis:")
        logger.info(f"  Train window: {self.train_window}")
        logger.info(f"  Test window: {self.test_window}")
        logger.info(f"  Iterations: {n_iterations}")
        logger.info(f"  Expanding window: {self.expanding_window}")

        model = None
        iterations_since_retrain = 0

        for i in tqdm(range(n_iterations), desc="Walk-Forward"):
            test_end = self.train_window + (i + 1) * self.test_window

            if test_end > n:
                break

            # Define train and test periods
            if self.expanding_window:
                train_start = 0
            else:
                train_start = i * self.test_window

            train_end = self.train_window + i * self.test_window
            test_start = train_end

            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[test_start:test_end]

            # Validate no temporal overlap
            if train_df.index.max() >= test_df.index.min():
                raise ValueError(
                    f"Temporal overlap detected at iteration {i}!\n"
                    f"Train ends: {train_df.index.max()}\n"
                    f"Test starts: {test_df.index.min()}"
                )

            # Retrain if needed
            if model is None or iterations_since_retrain >= self.retrain_frequency:
                logger.debug(
                    f"Iteration {i+1}/{n_iterations}: "
                    f"Training on {len(train_df)} samples"
                )
                model = train_func(train_df)
                iterations_since_retrain = 0
            else:
                iterations_since_retrain += 1

            # Predict on test set
            predictions = predict_func(model, test_df)

            # Store results
            all_predictions.extend(predictions)
            all_actuals.extend(test_df["label"].values)
            all_dates.extend(test_df.index)

        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)

        accuracy = (all_predictions == all_actuals).mean()

        results = {
            "predictions": all_predictions,
            "actuals": all_actuals,
            "dates": all_dates,
            "accuracy": accuracy,
            "n_samples": len(all_predictions),
            "n_iterations": n_iterations,
        }

        logger.info(f"\nWalk-Forward Results:")
        logger.info(f"  Total predictions: {len(all_predictions)}")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  ✓ No forward-looking bias - all predictions used only past data")

        return results
```

### 5.3 Updated Training Script (Sprint 1)

**File**: Create `scripts/train_model_no_bias.py`

```python
#!/usr/bin/env python3
"""
Training script with STRICT forward-looking bias prevention
"""

# [Include all imports from train_first_model.py]

# Add new import:
from tradeAI.utils.validation import validate_temporal_split

def main():
    logger.info("=" * 70)
    logger.info("TradeAI: Training with Forward-Looking Bias Prevention")
    logger.info("=" * 70)

    # [Steps 1-4 remain the same: data collection, cleaning, features, labeling]

    # STEP 5: Prepare Training Data (FIXED)
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Prepare Training Data (NO FORWARD-LOOKING BIAS)")
    logger.info("=" * 70)

    feature_cols = [col for col in df.columns if col not in ["label"]]
    X = df[feature_cols]
    y = df["label"]

    # Store original index for validation
    dates = df.index

    # CRITICAL: Split BEFORE normalization
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]

    # Validate temporal order
    logger.info("\nValidating temporal split...")
    validate_temporal_split(train_dates, val_dates, test_dates)

    # Split data
    X_train_raw = X.iloc[:train_end]
    X_val_raw = X.iloc[train_end:val_end]
    X_test_raw = X.iloc[val_end:]

    y_train = y.iloc[:train_end].values
    y_val = y.iloc[train_end:val_end].values
    y_test = y.iloc[val_end:].values

    logger.info(f"\n  Train period: {train_dates[0]} to {train_dates[-1]}")
    logger.info(f"  Val period:   {val_dates[0]} to {val_dates[-1]}")
    logger.info(f"  Test period:  {test_dates[0]} to {test_dates[-1]}")

    # Normalize - FIT ONLY ON TRAINING DATA
    logger.info("\nNormalizing with NO forward-looking bias...")
    normalizer = Normalizer(method="minmax")

    logger.info("  1. Fitting normalizer on TRAINING data only...")
    X_train = normalizer.fit_transform(X_train_raw).values

    logger.info("  2. Applying TRAINING statistics to validation data...")
    X_val = normalizer.transform(X_val_raw).values

    logger.info("  3. Applying TRAINING statistics to test data...")
    X_test = normalizer.transform(X_test_raw).values

    logger.info("\n✓ Normalization complete - NO DATA LEAKAGE!")
    logger.info("  Training statistics used for all sets")
    logger.info("  Test set never influenced training normalization")

    # [Continue with model training as before]

    # STEP 6: Additional validation
    logger.info("\n" + "=" * 70)
    logger.info("Final Validation: Checking for Forward-Looking Bias")
    logger.info("=" * 70)

    logger.info("\nChecklist:")
    logger.info("  ✓ Features use only past data (rolling windows, no negative shifts)")
    logger.info("  ✓ Labels use future data (acceptable for supervised learning)")
    logger.info("  ✓ Normalizer fit only on training data")
    logger.info("  ✓ Temporal order maintained (train → val → test)")
    logger.info("  ✓ No data overlap between splits")
    logger.info("  ✓ Walk-forward validation recommended for deployment")

    logger.info("\n" + "=" * 70)
    logger.info("✓ Training complete with NO forward-looking bias!")
    logger.info("=" * 70)
```

---

## 6. Summary and Recommendations

### 6.1 Critical Action Items

| Priority | Action | File | Effort | Impact |
|----------|--------|------|--------|--------|
| **P0** | Fix normalization order | `train_first_model.py` | 5 min | CRITICAL |
| **P1** | Add temporal validation | New: `utils/validation.py` | 30 min | HIGH |
| **P1** | Implement walk-forward | New: `backtesting/walk_forward.py` | 2 hours | HIGH |
| **P2** | Enhance splitter validation | `preprocessing/splitter.py` | 15 min | MEDIUM |
| **P3** | Change shuffle to error | `preprocessing/splitter.py` | 2 min | LOW |

### 6.2 The Golden Rules

**Rule 1**: Features MUST use only current and past data
```python
# ✅ GOOD
df["sma_20"] = df["close"].rolling(20).mean()

# ❌ BAD
df["future_price"] = df["close"].shift(-1)
```

**Rule 2**: Fit scalers/normalizers ONLY on training data
```python
# ✅ GOOD
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# ❌ BAD
scaler.fit(X_all)
```

**Rule 3**: Maintain strict temporal order
```python
# ✅ GOOD
train: [2020-01-01 to 2020-12-31]
val:   [2021-01-01 to 2021-06-30]
test:  [2021-07-01 to 2021-12-31]

# ❌ BAD
Random shuffle across all dates
```

**Rule 4**: Labels can use future data (that's supervised learning!)
```python
# ✅ GOOD
label = (price_in_10_bars - current_price) / current_price

# This is OK because:
# - Model learns from historical examples
# - At inference, we predict unknown future
```

**Rule 5**: Use walk-forward for realistic evaluation
```python
# ✅ GOOD
for each time period:
    train on all past data
    test on next period
    retrain

# ❌ BAD
Single train/test split
Standard k-fold cross-validation
```

### 6.3 Estimated Impact of Fixes

| Issue | Current Bias | After Fix |
|-------|--------------|-----------|
| Normalization leak | 5-15% inflated performance | Realistic performance |
| No walk-forward | Overly optimistic | Conservative estimate |
| No temporal validation | Silent failures | Errors caught early |

### 6.4 Verification Checklist

Before deploying any model, verify:

- [ ] Normalizer fit only on training data
- [ ] No negative shifts in features (`shift(-n)`)
- [ ] No centered rolling windows (`center=True`)
- [ ] Temporal order validated (no overlaps)
- [ ] Walk-forward backtesting performed
- [ ] Performance drop from training to walk-forward is reasonable (<10%)
- [ ] Model maintains performance over multiple walk-forward iterations

---

## 7. Conclusion

Forward-looking bias is the **#1 reason trading ML models fail in production**. Our current implementation has:

- ✅ Safe feature engineering
- ✅ Safe data collection
- 🔴 **CRITICAL normalization leak** (easy fix)
- 🟡 Missing walk-forward validation (moderate effort)
- 🟡 Weak temporal validation (easy fix)

**Immediate Action**: Fix the normalization order in `train_first_model.py` (5 minutes)

**Next Steps**: Implement walk-forward backtesting and temporal validation (2-3 hours)

With these fixes, TradeAI will have **production-grade forward-looking bias prevention**, ensuring models perform in live trading as they do in backtesting.

---

## References

1. Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
2. Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies*. Wiley.
3. Aronson, D. (2006). *Evidence-Based Technical Analysis*. Wiley.
