#!/usr/bin/env python3
"""
End-to-end script to train the first TradeAI model

This script demonstrates the complete pipeline:
1. Data collection
2. Preprocessing
3. Feature engineering
4. Labeling
5. Model training
6. Evaluation
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
from tradeAI.models.architectures.mlp import MLP
from tradeAI.models.architectures.lstm import LSTMModel
from tradeAI.training.trainer import Trainer
from tradeAI.utils.logger import setup_logging, get_logger
from tradeAI.utils.helpers import ensure_dir
from tradeAI.utils.validation import TemporalValidator
from tradeAI.visualization import TradeAIReportGenerator

# Setup logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)


def main():
    """Run the complete training pipeline."""
    logger.info("=" * 70)
    logger.info("TradeAI: End-to-End Model Training Pipeline")
    logger.info("=" * 70)

    # Initialize report generator
    report_gen = TradeAIReportGenerator(output_dir="results")

    # ========================================================================
    # STEP 1: Data Collection
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Data Collection")
    logger.info("=" * 70)

    provider = YFinanceProvider()

    symbol = "SPY"
    timeframe = "1h"
    start_date = datetime.now() - timedelta(days=365)  # 1 year of data
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
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Data Preprocessing")
    logger.info("=" * 70)

    # Clean data
    cleaner = DataCleaner(
        handle_missing="ffill",
        remove_outliers=True,
        outlier_method="iqr",
        outlier_threshold=3.0,
    )

    df = cleaner.clean(df)
    logger.info(f"✓ Data cleaned: {len(df)} bars remaining")

    # ========================================================================
    # STEP 3: Feature Engineering
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Feature Engineering")
    logger.info("=" * 70)

    engineer = FeatureEngineer()

    df = engineer.add_all_features(df)
    logger.info(f"✓ Features added: {len(df.columns)} total columns")

    # Remove NaN values from indicators
    df = df.dropna()
    logger.info(f"  After removing NaN: {len(df)} bars")

    # Validate feature computation (check for forward-looking bias)
    logger.info("\nValidating feature computation...")
    feature_check = TemporalValidator.check_feature_computation(
        df,
        feature_cols=[col for col in df.columns if col not in ["open", "high", "low", "close", "volume"]],
    )
    if feature_check["warnings"]:
        logger.warning(f"Feature validation warnings: {feature_check['warnings']}")

    # ========================================================================
    # STEP 4: Labeling
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Labeling for Supervised Learning")
    logger.info("=" * 70)

    labeler = ReversalLabeler(
        method="multiclass",
        lookahead_window=10,
        threshold=0.02,  # 2% threshold
        num_classes=5,
    )

    df["label"] = labeler.label(df)

    # Remove samples without labels
    df = df[df["label"] != -1]
    logger.info(f"✓ Labels created: {len(df)} labeled samples")

    # Check class distribution
    class_dist = df["label"].value_counts().sort_index()
    logger.info("\n  Class distribution:")
    for cls, count in class_dist.items():
        logger.info(f"    Class {cls}: {count} ({count/len(df)*100:.1f}%)")

    # ========================================================================
    # STEP 5: Prepare Training Data
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Prepare Training Data")
    logger.info("=" * 70)

    # Split features and labels
    feature_cols = [col for col in df.columns if col not in ["label"]]
    X = df[feature_cols].values
    y = df["label"].values

    logger.info(f"  Features: {X.shape}")
    logger.info(f"  Labels: {y.shape}")

    # Split into train/val/test BEFORE normalization
    # This is CRITICAL to prevent data leakage!
    splitter = TimeSeriesSplitter(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        shuffle=False,  # Keep temporal order
    )

    # Create indices for splitting
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train_raw, y_train = X[:train_end], y[:train_end]
    X_val_raw, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test_raw, y_test = X[val_end:], y[val_end:]

    logger.info(f"✓ Data split:")
    logger.info(f"  Train: {len(X_train_raw)} samples")
    logger.info(f"  Val:   {len(X_val_raw)} samples")
    logger.info(f"  Test:  {len(X_test_raw)} samples")

    # Validate temporal split (ensure no forward-looking bias)
    logger.info("\nValidating temporal split...")
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    TemporalValidator.validate_temporal_split(
        train_data=train_df,
        val_data=val_df,
        test_data=test_df,
        raise_on_error=True,
    )

    # Normalize features - fit ONLY on training data to prevent data leakage
    logger.info("\nNormalizing features...")
    normalizer = Normalizer(method="minmax")

    # Fit on TRAINING data only
    X_train = normalizer.fit_transform(pd.DataFrame(X_train_raw, columns=feature_cols)).values

    # Transform val/test using TRAINING statistics (no data leakage!)
    X_val = normalizer.transform(pd.DataFrame(X_val_raw, columns=feature_cols)).values
    X_test = normalizer.transform(pd.DataFrame(X_test_raw, columns=feature_cols)).values

    logger.info("✓ Features normalized using ONLY training statistics")
    logger.info("  (No data leakage - test/val statistics not used in normalization)")

    # ========================================================================
    # STEP 6: Model Training (MLP)
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: Training MLP Model")
    logger.info("=" * 70)

    # Create model
    input_size = X_train.shape[1]
    output_size = 5  # 5 classes

    model_mlp = MLP(
        input_size=input_size,
        output_size=output_size,
        hidden_layers=[128, 64, 32],
        dropout=0.3,
        activation="relu",
        batch_norm=True,
    )

    logger.info(f"Model: {model_mlp}")

    # Create trainer
    trainer_mlp = Trainer(
        model=model_mlp,
        learning_rate=0.001,
        batch_size=64,
        epochs=50,
        early_stopping_patience=10,
        checkpoint_dir="models/checkpoints",
    )

    # Train model
    logger.info("\nTraining MLP...")
    history_mlp = trainer_mlp.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        task="classification",
    )

    # Evaluate
    logger.info("\nEvaluating MLP on test set...")
    metrics_mlp = trainer_mlp.evaluate(X_test, y_test, task="classification")

    logger.info("\n✓ MLP Training complete!")
    logger.info(f"  Final test accuracy: {metrics_mlp['test_accuracy']:.3f}")
    logger.info(f"  Final test loss: {metrics_mlp['test_loss']:.4f}")

    # ========================================================================
    # STEP 6.5: Generate Visualization Reports (MLP)
    # ========================================================================
    logger.info("\nGenerating visualization reports...")

    # Get predictions and probabilities
    mlp_predictions = trainer_mlp.predict(X_test)
    mlp_pred_classes = np.argmax(mlp_predictions, axis=1)

    # Training report (labels on chart)
    train_report_path = report_gen.generate_training_report(
        df=train_df,
        labels=y_train,
        label_type="reversal",
        model_name="MLP",
        symbol=symbol,
        timeframe=timeframe,
        metadata={
            "num_samples": len(train_df),
            "model_params": model_mlp.get_num_parameters(),
            "hidden_layers": str([128, 64, 32]),
            "dropout": 0.3,
        }
    )
    logger.info(f"  Training report: {train_report_path}")

    # Validation report
    val_report_path = report_gen.generate_prediction_report(
        df=val_df,
        true_labels=y_val,
        predictions=np.argmax(trainer_mlp.predict(X_val), axis=1),
        probabilities=trainer_mlp.predict(X_val),
        model_name="MLP",
        symbol=symbol,
        timeframe=timeframe,
        split_type="validation",
        metadata={
            "num_samples": len(val_df),
            "model_params": model_mlp.get_num_parameters(),
        }
    )
    logger.info(f"  Validation report: {val_report_path}")

    # Test report
    test_report_path = report_gen.generate_prediction_report(
        df=test_df,
        true_labels=y_test,
        predictions=mlp_pred_classes,
        probabilities=mlp_predictions,
        model_name="MLP",
        symbol=symbol,
        timeframe=timeframe,
        split_type="test",
        metadata={
            "num_samples": len(test_df),
            "model_params": model_mlp.get_num_parameters(),
            "test_accuracy": metrics_mlp['test_accuracy'],
            "test_loss": metrics_mlp['test_loss'],
        }
    )
    logger.info(f"  Test report: {test_report_path}")

    # ========================================================================
    # STEP 7: Training LSTM Model (with sequences)
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: Training LSTM Model")
    logger.info("=" * 70)

    # Create sequences for LSTM from already-normalized splits
    # This ensures no data leakage in sequence creation
    from tradeAI.preprocessing.splitter import create_sequences

    sequence_length = 30  # Use 30 time steps

    # Concatenate normalized data for sequence creation (maintains temporal order)
    X_all_normalized = np.concatenate([X_train, X_val, X_test], axis=0)
    y_all = np.concatenate([y_train, y_val, y_test], axis=0)

    # Create sequences
    X_seq_full = []
    y_seq_full = []

    for i in range(sequence_length, len(X_all_normalized)):
        X_seq_full.append(X_all_normalized[i-sequence_length:i])
        y_seq_full.append(y_all[i])

    X_seq_full = np.array(X_seq_full)
    y_seq_full = np.array(y_seq_full)

    logger.info(f"✓ Created sequences: {X_seq_full.shape}")

    # Split sequences (maintaining same temporal splits)
    n_seq = len(X_seq_full)
    train_end_seq = int(n_seq * 0.7)
    val_end_seq = int(n_seq * 0.85)

    X_train_seq = X_seq_full[:train_end_seq]
    y_train_seq = y_seq_full[:train_end_seq]
    X_val_seq = X_seq_full[train_end_seq:val_end_seq]
    y_val_seq = y_seq_full[train_end_seq:val_end_seq]
    X_test_seq = X_seq_full[val_end_seq:]
    y_test_seq = y_seq_full[val_end_seq:]

    logger.info(f"  Train sequences: {len(X_train_seq)}")
    logger.info(f"  Val sequences:   {len(X_val_seq)}")
    logger.info(f"  Test sequences:  {len(X_test_seq)}")

    # Create LSTM model
    model_lstm = LSTMModel(
        input_size=input_size,
        output_size=output_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=False,
    )

    logger.info(f"Model: {model_lstm}")

    # Create trainer
    trainer_lstm = Trainer(
        model=model_lstm,
        learning_rate=0.001,
        batch_size=32,
        epochs=30,
        early_stopping_patience=10,
        checkpoint_dir="models/checkpoints",
    )

    # Train model
    logger.info("\nTraining LSTM...")
    history_lstm = trainer_lstm.fit(
        X_train=X_train_seq,
        y_train=y_train_seq,
        X_val=X_val_seq,
        y_val=y_val_seq,
        task="classification",
    )

    # Evaluate
    logger.info("\nEvaluating LSTM on test set...")
    metrics_lstm = trainer_lstm.evaluate(X_test_seq, y_test_seq, task="classification")

    logger.info("\n✓ LSTM Training complete!")
    logger.info(f"  Final test accuracy: {metrics_lstm['test_accuracy']:.3f}")
    logger.info(f"  Final test loss: {metrics_lstm['test_loss']:.4f}")

    # ========================================================================
    # STEP 7.5: Generate Visualization Reports (LSTM)
    # ========================================================================
    logger.info("\nGenerating LSTM visualization reports...")

    # Get predictions and probabilities
    lstm_predictions = trainer_lstm.predict(X_test_seq)
    lstm_pred_classes = np.argmax(lstm_predictions, axis=1)

    # Test report for LSTM
    lstm_test_report_path = report_gen.generate_prediction_report(
        df=test_df.iloc[sequence_length:],  # Adjust for sequence length
        true_labels=y_test_seq,
        predictions=lstm_pred_classes,
        probabilities=lstm_predictions,
        model_name="LSTM",
        symbol=symbol,
        timeframe=timeframe,
        split_type="test",
        metadata={
            "num_samples": len(y_test_seq),
            "model_params": model_lstm.get_num_parameters(),
            "sequence_length": sequence_length,
            "test_accuracy": metrics_lstm['test_accuracy'],
            "test_loss": metrics_lstm['test_loss'],
        }
    )
    logger.info(f"  LSTM test report: {lstm_test_report_path}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\nData:")
    logger.info(f"  Symbol: {symbol}")
    logger.info(f"  Timeframe: {timeframe}")
    logger.info(f"  Total samples: {len(df)}")
    logger.info(f"  Features: {input_size}")
    logger.info(f"  Classes: {output_size}")

    logger.info(f"\nMLP Model:")
    logger.info(f"  Parameters: {model_mlp.get_num_parameters():,}")
    logger.info(f"  Test Accuracy: {metrics_mlp['test_accuracy']:.3f}")
    logger.info(f"  Test Loss: {metrics_mlp['test_loss']:.4f}")

    logger.info(f"\nLSTM Model:")
    logger.info(f"  Parameters: {model_lstm.get_num_parameters():,}")
    logger.info(f"  Test Accuracy: {metrics_lstm['test_accuracy']:.3f}")
    logger.info(f"  Test Loss: {metrics_lstm['test_loss']:.4f}")

    logger.info("\n" + "=" * 70)
    logger.info("✓ Pipeline completed successfully!")
    logger.info("=" * 70)

    logger.info("\nGenerated Reports (saved to results/):")
    logger.info(f"  1. Training report (with labels): {train_report_path}")
    logger.info(f"  2. Validation report (MLP): {val_report_path}")
    logger.info(f"  3. Test report (MLP): {test_report_path}")
    logger.info(f"  4. Test report (LSTM): {lstm_test_report_path}")

    logger.info("\nNext steps:")
    logger.info("  1. Open HTML reports in browser to analyze results")
    logger.info("  2. Experiment with different hyperparameters")
    logger.info("  3. Try different symbols and timeframes")
    logger.info("  4. Train ensemble models: python scripts/train_ensemble_models.py")
    logger.info("  5. Implement backtesting to evaluate trading performance")

    return 0


if __name__ == "__main__":
    sys.exit(main())
