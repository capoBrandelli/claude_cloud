#!/usr/bin/env python3
"""
Train three-model ensemble for robust trading predictions

This script demonstrates the complete ensemble strategy:
1. Reversal model - Identifies turning points
2. Continuation model - Predicts trend persistence
3. Direction model - Provides directional baseline

All three models:
- Use the SAME features (calculated from past data only)
- Use DIFFERENT labels (calculated from future data - acceptable for training)
- Share a SINGLE normalization pipeline (fit on training data only)
- Are trained independently to capture different market dynamics

This ensures NO forward-looking bias while creating a robust ensemble.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

# TradeAI modules
from tradeAI.data.providers.yfinance_provider import YFinanceProvider
from tradeAI.preprocessing.cleaner import DataCleaner
from tradeAI.preprocessing.normalizer import Normalizer
from tradeAI.preprocessing.splitter import TimeSeriesSplitter
from tradeAI.features.feature_engineer import FeatureEngineer
from tradeAI.labeling.reversal_labeler import ReversalLabeler
from tradeAI.labeling.continuation_labeler import ContinuationLabeler
from tradeAI.labeling.direction_labeler import DirectionLabeler
from tradeAI.models.architectures.mlp import MLP
from tradeAI.training.trainer import Trainer
from tradeAI.utils.logger import setup_logging, get_logger
from tradeAI.utils.helpers import ensure_dir
from tradeAI.utils.validation import TemporalValidator
from tradeAI.visualization import TradeAIReportGenerator
from tradeAI.models.ensemble import RuleBasedCombiner

# Setup logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)


def main():
    """Run the complete ensemble training pipeline."""
    logger.info("=" * 80)
    logger.info("TradeAI: Three-Model Ensemble Training Pipeline")
    logger.info("=" * 80)

    # Initialize report generator
    report_gen = TradeAIReportGenerator(output_dir="results")
    logger.info("")
    logger.info("Training Strategy:")
    logger.info("  Model 1: Reversal Probability  - Identifies turning points")
    logger.info("  Model 2: Continuation Probability - Predicts trend persistence")
    logger.info("  Model 3: Direction Probability - Provides directional baseline")
    logger.info("")
    logger.info("Key Principles:")
    logger.info("  ✓ Same features for all models (past data only)")
    logger.info("  ✓ Different labels for each model (future data - OK for training)")
    logger.info("  ✓ Single normalization (fit on training only)")
    logger.info("  ✓ No forward-looking bias")
    logger.info("=" * 80)

    # ========================================================================
    # STEP 1: Data Collection
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Data Collection")
    logger.info("=" * 80)

    provider = YFinanceProvider()

    symbol = "SPY"
    timeframe = "1h"
    start_date = datetime.now() - timedelta(days=365)  # 1 year
    end_date = datetime.now()

    logger.info(f"Fetching {symbol} {timeframe} data from {start_date.date()} to {end_date.date()}")

    df = provider.fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )

    logger.info(f"✓ Fetched {len(df)} bars")
    logger.info(f"  Date range: {df.index[0]} to {df.index[-1]}")

    # ========================================================================
    # STEP 2: Data Preprocessing
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Data Preprocessing")
    logger.info("=" * 80)

    cleaner = DataCleaner(
        handle_missing="ffill",
        remove_outliers=True,
        outlier_method="iqr",
        outlier_threshold=3.0,
    )

    df = cleaner.clean(df)
    logger.info(f"✓ Data cleaned: {len(df)} bars remaining")

    # ========================================================================
    # STEP 3: Feature Engineering (PAST DATA ONLY)
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Feature Engineering (Past Data Only)")
    logger.info("=" * 80)

    engineer = FeatureEngineer()
    df = engineer.add_all_features(df)
    logger.info(f"✓ Features added: {len(df.columns)} total columns")

    # Remove NaN
    df = df.dropna()
    logger.info(f"  After removing NaN: {len(df)} bars")

    # Validate features (check for forward-looking bias)
    logger.info("\nValidating feature computation...")
    feature_check = TemporalValidator.check_feature_computation(
        df,
        feature_cols=[col for col in df.columns if col not in ["open", "high", "low", "close", "volume"]],
    )
    if feature_check["warnings"]:
        logger.warning(f"Feature validation warnings: {feature_check['warnings']}")

    # ========================================================================
    # STEP 4: Create Three Different Label Sets (FUTURE DATA - OK)
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Create Three Different Label Sets (Future Data - OK)")
    logger.info("=" * 80)

    # Label 1: Reversal (identifies turning points)
    logger.info("\nCreating reversal labels...")
    reversal_labeler = ReversalLabeler(
        method="multiclass",
        lookahead_window=10,
        threshold=0.02,
        num_classes=5,
    )
    df["label_reversal"] = reversal_labeler.label(df)

    # Label 2: Continuation (identifies trend persistence)
    logger.info("\nCreating continuation labels...")
    continuation_labeler = ContinuationLabeler(
        method="multiclass",
        lookback=20,
        lookahead=10,
        trend_threshold=0.02,
        continuation_threshold=0.015,
        num_classes=5,
    )
    df["label_continuation"] = continuation_labeler.label(df)

    # Label 3: Direction (simple directional prediction)
    logger.info("\nCreating direction labels...")
    direction_labeler = DirectionLabeler(
        method="multiclass",
        lookahead=10,
        threshold=0.01,
        num_classes=5,
    )
    df["label_direction"] = direction_labeler.label(df)

    # Remove samples without valid labels
    logger.info("\nFiltering samples with valid labels...")
    df_clean = df[
        (df["label_reversal"] != -1) &
        (df["label_continuation"] != -1) &
        (df["label_direction"] != -1)
    ].copy()

    logger.info(f"✓ Samples with all valid labels: {len(df_clean)}")
    logger.info(f"  Filtered out: {len(df) - len(df_clean)} samples")

    # ========================================================================
    # STEP 5: Prepare Training Data (SINGLE NORMALIZATION)
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Prepare Training Data (Single Normalization)")
    logger.info("=" * 80)

    # Extract features (same for all three models)
    feature_cols = [col for col in df_clean.columns if not col.startswith("label")]
    X = df_clean[feature_cols].values

    # Extract three label sets
    y_reversal = df_clean["label_reversal"].values
    y_continuation = df_clean["label_continuation"].values
    y_direction = df_clean["label_direction"].values

    logger.info(f"  Features shape: {X.shape}")
    logger.info(f"  Reversal labels: {y_reversal.shape}")
    logger.info(f"  Continuation labels: {y_continuation.shape}")
    logger.info(f"  Direction labels: {y_direction.shape}")

    # CRITICAL: Split FIRST, then normalize
    logger.info("\nSplitting data temporally...")
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    # Split features
    X_train_raw = X[:train_end]
    X_val_raw = X[train_end:val_end]
    X_test_raw = X[val_end:]

    # Split all three label sets
    y_reversal_train, y_reversal_val, y_reversal_test = (
        y_reversal[:train_end],
        y_reversal[train_end:val_end],
        y_reversal[val_end:],
    )
    y_continuation_train, y_continuation_val, y_continuation_test = (
        y_continuation[:train_end],
        y_continuation[train_end:val_end],
        y_continuation[val_end:],
    )
    y_direction_train, y_direction_val, y_direction_test = (
        y_direction[:train_end],
        y_direction[train_end:val_end],
        y_direction[val_end:],
    )

    logger.info(f"✓ Data split:")
    logger.info(f"  Train: {len(X_train_raw)} samples")
    logger.info(f"  Val:   {len(X_val_raw)} samples")
    logger.info(f"  Test:  {len(X_test_raw)} samples")

    # Validate temporal split
    logger.info("\nValidating temporal split...")
    train_df = df_clean.iloc[:train_end]
    val_df = df_clean.iloc[train_end:val_end]
    test_df = df_clean.iloc[val_end:]

    TemporalValidator.validate_temporal_split(
        train_data=train_df,
        val_data=val_df,
        test_data=test_df,
        raise_on_error=True,
    )

    # SINGLE normalization for all models (fit on training only)
    logger.info("\nNormalizing features (SINGLE normalization for all models)...")
    normalizer = Normalizer(method="minmax")

    # Fit ONLY on training data
    X_train = normalizer.fit_transform(pd.DataFrame(X_train_raw, columns=feature_cols)).values
    X_val = normalizer.transform(pd.DataFrame(X_val_raw, columns=feature_cols)).values
    X_test = normalizer.transform(pd.DataFrame(X_test_raw, columns=feature_cols)).values

    logger.info("✓ Features normalized using ONLY training statistics")
    logger.info("  (Same normalized features will be used for all three models)")

    # ========================================================================
    # STEP 6: Train Model 1 - Reversal Probability
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Training Model 1 - Reversal Probability")
    logger.info("=" * 80)

    input_size = X_train.shape[1]
    output_size = 5

    model_reversal = MLP(
        input_size=input_size,
        output_size=output_size,
        hidden_layers=[128, 64, 32],
        dropout=0.3,
        activation="relu",
        batch_norm=True,
    )

    logger.info(f"Model: {model_reversal}")

    trainer_reversal = Trainer(
        model=model_reversal,
        learning_rate=0.001,
        batch_size=64,
        epochs=50,
        early_stopping_patience=10,
        checkpoint_dir="models/ensemble/reversal",
    )

    logger.info("\nTraining reversal model...")
    history_reversal = trainer_reversal.fit(
        X_train=X_train,
        y_train=y_reversal_train,
        X_val=X_val,
        y_val=y_reversal_val,
        task="classification",
    )

    # Evaluate
    logger.info("\nEvaluating reversal model on test set...")
    metrics_reversal = trainer_reversal.evaluate(
        X_test, y_reversal_test, task="classification"
    )

    logger.info("\n✓ Reversal model training complete!")
    logger.info(f"  Test accuracy: {metrics_reversal['test_accuracy']:.3f}")
    logger.info(f"  Test loss: {metrics_reversal['test_loss']:.4f}")

    # ========================================================================
    # STEP 7: Train Model 2 - Continuation Probability
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: Training Model 2 - Continuation Probability")
    logger.info("=" * 80)

    model_continuation = MLP(
        input_size=input_size,
        output_size=output_size,
        hidden_layers=[128, 64, 32],
        dropout=0.3,
        activation="relu",
        batch_norm=True,
    )

    logger.info(f"Model: {model_continuation}")

    trainer_continuation = Trainer(
        model=model_continuation,
        learning_rate=0.001,
        batch_size=64,
        epochs=50,
        early_stopping_patience=10,
        checkpoint_dir="models/ensemble/continuation",
    )

    logger.info("\nTraining continuation model...")
    history_continuation = trainer_continuation.fit(
        X_train=X_train,
        y_train=y_continuation_train,
        X_val=X_val,
        y_val=y_continuation_val,
        task="classification",
    )

    # Evaluate
    logger.info("\nEvaluating continuation model on test set...")
    metrics_continuation = trainer_continuation.evaluate(
        X_test, y_continuation_test, task="classification"
    )

    logger.info("\n✓ Continuation model training complete!")
    logger.info(f"  Test accuracy: {metrics_continuation['test_accuracy']:.3f}")
    logger.info(f"  Test loss: {metrics_continuation['test_loss']:.4f}")

    # ========================================================================
    # STEP 8: Train Model 3 - Direction Probability
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 8: Training Model 3 - Direction Probability")
    logger.info("=" * 80)

    model_direction = MLP(
        input_size=input_size,
        output_size=output_size,
        hidden_layers=[128, 64, 32],
        dropout=0.3,
        activation="relu",
        batch_norm=True,
    )

    logger.info(f"Model: {model_direction}")

    trainer_direction = Trainer(
        model=model_direction,
        learning_rate=0.001,
        batch_size=64,
        epochs=50,
        early_stopping_patience=10,
        checkpoint_dir="models/ensemble/direction",
    )

    logger.info("\nTraining direction model...")
    history_direction = trainer_direction.fit(
        X_train=X_train,
        y_train=y_direction_train,
        X_val=X_val,
        y_val=y_direction_val,
        task="classification",
    )

    # Evaluate
    logger.info("\nEvaluating direction model on test set...")
    metrics_direction = trainer_direction.evaluate(
        X_test, y_direction_test, task="classification"
    )

    logger.info("\n✓ Direction model training complete!")
    logger.info(f"  Test accuracy: {metrics_direction['test_accuracy']:.3f}")
    logger.info(f"  Test loss: {metrics_direction['test_loss']:.4f}")

    # ========================================================================
    # STEP 8.5: Generate Ensemble Visualization Report
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 9: Generating Ensemble Visualization Reports")
    logger.info("=" * 80)

    # Get test predictions from all three models
    p_reversal_test = trainer_reversal.predict(X_test)
    p_continuation_test = trainer_continuation.predict(X_test)
    p_direction_test = trainer_direction.predict(X_test)

    # Combine signals using rule-based combiner
    combiner = RuleBasedCombiner(num_classes=output_size)
    ensemble_result = combiner.combine(
        p_reversal_test,
        p_continuation_test,
        p_direction_test
    )

    combined_signals = ensemble_result['signal']
    confidence = ensemble_result['confidence']
    agreement = ensemble_result['agreement']

    # Generate ensemble report
    ensemble_report_path = report_gen.generate_ensemble_report(
        df=test_df,
        reversal_probs=p_reversal_test,
        continuation_probs=p_continuation_test,
        direction_probs=p_direction_test,
        combined_signals=combined_signals,
        confidence=confidence,
        agreement=agreement,
        symbol=symbol,
        timeframe=timeframe,
        metadata={
            "num_samples": len(test_df),
            "total_params": (
                model_reversal.get_num_parameters() +
                model_continuation.get_num_parameters() +
                model_direction.get_num_parameters()
            ),
            "reversal_accuracy": metrics_reversal['test_accuracy'],
            "continuation_accuracy": metrics_continuation['test_accuracy'],
            "direction_accuracy": metrics_direction['test_accuracy'],
        }
    )
    logger.info(f"\n✓ Ensemble report generated: {ensemble_report_path}")

    # Also generate individual model reports
    logger.info("\nGenerating individual model reports...")

    # Reversal training report
    rev_train_report = report_gen.generate_training_report(
        df=train_df,
        labels=y_reversal_train,
        label_type="reversal",
        model_name="Reversal",
        symbol=symbol,
        timeframe=timeframe,
        metadata={"num_samples": len(train_df), "model_params": model_reversal.get_num_parameters()}
    )

    # Continuation training report
    cont_train_report = report_gen.generate_training_report(
        df=train_df,
        labels=y_continuation_train,
        label_type="continuation",
        model_name="Continuation",
        symbol=symbol,
        timeframe=timeframe,
        metadata={"num_samples": len(train_df), "model_params": model_continuation.get_num_parameters()}
    )

    # Direction training report
    dir_train_report = report_gen.generate_training_report(
        df=train_df,
        labels=y_direction_train,
        label_type="direction",
        model_name="Direction",
        symbol=symbol,
        timeframe=timeframe,
        metadata={"num_samples": len(train_df), "model_params": model_direction.get_num_parameters()}
    )

    logger.info("✓ All visualization reports generated successfully!")

    # ========================================================================
    # STEP 10: Ensemble Summary
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("ENSEMBLE TRAINING SUMMARY")
    logger.info("=" * 80)

    logger.info(f"\nData:")
    logger.info(f"  Symbol: {symbol}")
    logger.info(f"  Timeframe: {timeframe}")
    logger.info(f"  Total samples: {len(df_clean)}")
    logger.info(f"  Features: {input_size}")
    logger.info(f"  Classes: {output_size}")

    logger.info(f"\nModel 1 - Reversal:")
    logger.info(f"  Parameters: {model_reversal.get_num_parameters():,}")
    logger.info(f"  Test Accuracy: {metrics_reversal['test_accuracy']:.3f}")
    logger.info(f"  Test Loss: {metrics_reversal['test_loss']:.4f}")

    logger.info(f"\nModel 2 - Continuation:")
    logger.info(f"  Parameters: {model_continuation.get_num_parameters():,}")
    logger.info(f"  Test Accuracy: {metrics_continuation['test_accuracy']:.3f}")
    logger.info(f"  Test Loss: {metrics_continuation['test_loss']:.4f}")

    logger.info(f"\nModel 3 - Direction:")
    logger.info(f"  Parameters: {model_direction.get_num_parameters():,}")
    logger.info(f"  Test Accuracy: {metrics_direction['test_accuracy']:.3f}")
    logger.info(f"  Test Loss: {metrics_direction['test_loss']:.4f}")

    # Calculate ensemble average accuracy
    avg_accuracy = (
        metrics_reversal['test_accuracy'] +
        metrics_continuation['test_accuracy'] +
        metrics_direction['test_accuracy']
    ) / 3

    logger.info(f"\nEnsemble:")
    logger.info(f"  Average Test Accuracy: {avg_accuracy:.3f}")
    logger.info(f"  Total Parameters: {(model_reversal.get_num_parameters() + model_continuation.get_num_parameters() + model_direction.get_num_parameters()):,}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ Ensemble training completed successfully!")
    logger.info("=" * 80)

    logger.info("\nGenerated Reports (saved to results/):")
    logger.info(f"  1. Ensemble report: {ensemble_report_path}")
    logger.info(f"  2. Reversal training report: {rev_train_report}")
    logger.info(f"  3. Continuation training report: {cont_train_report}")
    logger.info(f"  4. Direction training report: {dir_train_report}")

    logger.info("\nNext steps:")
    logger.info("  1. Open HTML reports in browser to analyze ensemble results")
    logger.info("  2. Analyze agreement/disagreement patterns in reports")
    logger.info("  3. Test ensemble on new data")
    logger.info("  4. Implement walk-forward backtesting")
    logger.info("  5. Deploy for real-time predictions")

    logger.info("\nUsage example:")
    logger.info("  # Get predictions from all three models")
    logger.info("  p_reversal = model_reversal.predict(latest_features)")
    logger.info("  p_continuation = model_continuation.predict(latest_features)")
    logger.info("  p_direction = model_direction.predict(latest_features)")
    logger.info("")
    logger.info("  # Combine for robust signal")
    logger.info("  if p_reversal[1] > 0.7 and p_continuation[0] > 0.6:")
    logger.info("      # High probability reversal + trend break = strong reversal signal")
    logger.info("  elif p_continuation[4] > 0.7 and p_direction[4] > 0.6:")
    logger.info("      # High continuation + bullish direction = strong trend continuation")

    return 0


if __name__ == "__main__":
    sys.exit(main())
