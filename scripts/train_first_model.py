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

# Setup logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)


def main():
    """Run the complete training pipeline."""
    logger.info("=" * 70)
    logger.info("TradeAI: End-to-End Model Training Pipeline")
    logger.info("=" * 70)

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

    # Normalize features
    normalizer = Normalizer(method="minmax")
    X = normalizer.fit_transform(pd.DataFrame(X, columns=feature_cols)).values

    logger.info("✓ Features normalized")

    # Split into train/val/test
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

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    logger.info(f"✓ Data split:")
    logger.info(f"  Train: {len(X_train)} samples")
    logger.info(f"  Val:   {len(X_val)} samples")
    logger.info(f"  Test:  {len(X_test)} samples")

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
    # STEP 7: Training LSTM Model (with sequences)
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: Training LSTM Model")
    logger.info("=" * 70)

    # Create sequences for LSTM
    from tradeAI.preprocessing.splitter import create_sequences

    sequence_length = 30  # Use 30 time steps

    # Create sequences
    X_seq_full = []
    y_seq_full = []

    for i in range(sequence_length, len(X)):
        X_seq_full.append(X[i-sequence_length:i])
        y_seq_full.append(y[i])

    X_seq_full = np.array(X_seq_full)
    y_seq_full = np.array(y_seq_full)

    logger.info(f"✓ Created sequences: {X_seq_full.shape}")

    # Split sequences
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

    logger.info("\nNext steps:")
    logger.info("  1. Experiment with different hyperparameters")
    logger.info("  2. Try different symbols and timeframes")
    logger.info("  3. Implement backtesting to evaluate trading performance")
    logger.info("  4. Add more sophisticated features and indicators")
    logger.info("  5. Ensemble multiple models for better predictions")

    return 0


if __name__ == "__main__":
    sys.exit(main())
