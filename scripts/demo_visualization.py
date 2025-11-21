#!/usr/bin/env python3
"""
Demo script to generate example visualization reports

Creates synthetic data and example reports to demonstrate the
visualization capabilities without running full training.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from tradeAI.visualization import TradeAIReportGenerator
from tradeAI.utils.logger import setup_logging, get_logger

setup_logging(log_level="INFO")
logger = get_logger(__name__)


def generate_synthetic_ohlcv(n_samples=500, start_price=100):
    """Generate synthetic OHLCV data with realistic patterns."""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='1H')

    # Generate price with trend and noise
    trend = np.linspace(0, 20, n_samples)
    noise = np.cumsum(np.random.randn(n_samples) * 0.5)
    close_prices = start_price + trend + noise

    # Create OHLC from close
    open_prices = close_prices + np.random.randn(n_samples) * 0.3
    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.randn(n_samples) * 0.5)
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.randn(n_samples) * 0.5)
    volume = np.random.randint(1000000, 5000000, n_samples)

    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume,
    }, index=dates)

    return df


def generate_synthetic_labels(n_samples, label_type='reversal'):
    """Generate synthetic labels."""
    np.random.seed(42)

    if label_type == 'reversal':
        # More neutral, some reversals
        labels = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    elif label_type == 'continuation':
        # More continuation
        labels = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.1, 0.15, 0.5, 0.15, 0.1])
    else:  # direction
        # Balanced
        labels = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.15, 0.2, 0.3, 0.2, 0.15])

    return labels


def generate_synthetic_predictions(true_labels, accuracy=0.65):
    """Generate predictions with specified accuracy."""
    np.random.seed(43)

    predictions = true_labels.copy()
    n_incorrect = int(len(true_labels) * (1 - accuracy))
    incorrect_indices = np.random.choice(len(true_labels), n_incorrect, replace=False)

    for idx in incorrect_indices:
        # Random wrong class
        wrong_class = np.random.choice([c for c in range(5) if c != true_labels[idx]])
        predictions[idx] = wrong_class

    return predictions


def generate_synthetic_probabilities(predictions, confidence_mean=0.7):
    """Generate probability distributions."""
    np.random.seed(44)

    n_samples = len(predictions)
    probabilities = np.zeros((n_samples, 5))

    for i in range(n_samples):
        pred_class = predictions[i]
        # Generate confidence around mean
        max_prob = np.random.beta(5, 2) * 0.4 + 0.5  # 0.5 to 0.9

        # Assign to predicted class
        probabilities[i, pred_class] = max_prob

        # Distribute remaining probability
        remaining = 1.0 - max_prob
        other_probs = np.random.dirichlet([1] * 4)
        other_idx = [j for j in range(5) if j != pred_class]
        probabilities[i, other_idx] = other_probs * remaining

    return probabilities


def main():
    """Generate example reports."""
    logger.info("=" * 70)
    logger.info("TradeAI Visualization Demo")
    logger.info("=" * 70)
    logger.info("\nGenerating synthetic data and example reports...\n")

    # Initialize report generator
    report_gen = TradeAIReportGenerator(output_dir="results/examples")

    # Generate data
    df_full = generate_synthetic_ohlcv(n_samples=500)

    # Split into train/val/test
    train_end = 350
    val_end = 425

    df_train = df_full.iloc[:train_end]
    df_val = df_full.iloc[train_end:val_end]
    df_test = df_full.iloc[val_end:]

    # ========================================================================
    # 1. Training Report Example
    # ========================================================================
    logger.info("1. Generating training report example...")

    y_train = generate_synthetic_labels(len(df_train), 'reversal')

    train_report = report_gen.generate_training_report(
        df=df_train,
        labels=y_train,
        label_type="reversal",
        model_name="MLP_Example",
        symbol="SPY",
        timeframe="1h",
        metadata={
            "num_samples": len(df_train),
            "model_params": 145_678,
            "hidden_layers": "[128, 64, 32]",
            "dropout": 0.3,
            "note": "This is a demo with synthetic data"
        }
    )
    logger.info(f"   ✓ Training report: {train_report}")

    # ========================================================================
    # 2. Validation Report Example
    # ========================================================================
    logger.info("\n2. Generating validation report example...")

    y_val = generate_synthetic_labels(len(df_val), 'reversal')
    val_predictions = generate_synthetic_predictions(y_val, accuracy=0.68)
    val_probabilities = generate_synthetic_probabilities(val_predictions)

    val_report = report_gen.generate_prediction_report(
        df=df_val,
        true_labels=y_val,
        predictions=val_predictions,
        probabilities=val_probabilities,
        model_name="MLP_Example",
        symbol="SPY",
        timeframe="1h",
        split_type="validation",
        metadata={
            "num_samples": len(df_val),
            "model_params": 145_678,
            "accuracy": 0.68,
            "note": "This is a demo with synthetic data"
        }
    )
    logger.info(f"   ✓ Validation report: {val_report}")

    # ========================================================================
    # 3. Test Report Example
    # ========================================================================
    logger.info("\n3. Generating test report example...")

    y_test = generate_synthetic_labels(len(df_test), 'reversal')
    test_predictions = generate_synthetic_predictions(y_test, accuracy=0.65)
    test_probabilities = generate_synthetic_probabilities(test_predictions)

    test_report = report_gen.generate_prediction_report(
        df=df_test,
        true_labels=y_test,
        predictions=test_predictions,
        probabilities=test_probabilities,
        model_name="MLP_Example",
        symbol="SPY",
        timeframe="1h",
        split_type="test",
        metadata={
            "num_samples": len(df_test),
            "model_params": 145_678,
            "test_accuracy": 0.65,
            "test_loss": 0.8523,
            "note": "This is a demo with synthetic data"
        }
    )
    logger.info(f"   ✓ Test report: {test_report}")

    # ========================================================================
    # 4. Ensemble Report Example
    # ========================================================================
    logger.info("\n4. Generating ensemble report example...")

    # Generate predictions for all three models
    reversal_probs = generate_synthetic_probabilities(
        generate_synthetic_predictions(y_test, accuracy=0.63)
    )
    continuation_probs = generate_synthetic_probabilities(
        generate_synthetic_predictions(y_test, accuracy=0.61)
    )
    direction_probs = generate_synthetic_probabilities(
        generate_synthetic_predictions(y_test, accuracy=0.59)
    )

    # Generate combined signals
    np.random.seed(45)
    combined_signals = np.random.choice([-1, 0, 1], len(df_test), p=[0.25, 0.5, 0.25])
    confidence = np.random.beta(3, 2, len(df_test))
    agreement = np.random.beta(4, 2, len(df_test))

    ensemble_report = report_gen.generate_ensemble_report(
        df=df_test,
        reversal_probs=reversal_probs,
        continuation_probs=continuation_probs,
        direction_probs=direction_probs,
        combined_signals=combined_signals,
        confidence=confidence,
        agreement=agreement,
        symbol="SPY",
        timeframe="1h",
        metadata={
            "num_samples": len(df_test),
            "total_params": 437_034,
            "reversal_accuracy": 0.63,
            "continuation_accuracy": 0.61,
            "direction_accuracy": 0.59,
            "avg_accuracy": 0.61,
            "note": "This is a demo with synthetic data"
        }
    )
    logger.info(f"   ✓ Ensemble report: {ensemble_report}")

    # ========================================================================
    # 5. Additional Label Type Examples
    # ========================================================================
    logger.info("\n5. Generating additional label type examples...")

    # Continuation training report
    y_train_cont = generate_synthetic_labels(len(df_train), 'continuation')
    cont_train_report = report_gen.generate_training_report(
        df=df_train,
        labels=y_train_cont,
        label_type="continuation",
        model_name="Continuation_Example",
        symbol="SPY",
        timeframe="1h",
        metadata={
            "num_samples": len(df_train),
            "model_params": 145_678,
            "note": "Continuation labeling example"
        }
    )
    logger.info(f"   ✓ Continuation training report: {cont_train_report}")

    # Direction training report
    y_train_dir = generate_synthetic_labels(len(df_train), 'direction')
    dir_train_report = report_gen.generate_training_report(
        df=df_train,
        labels=y_train_dir,
        label_type="direction",
        model_name="Direction_Example",
        symbol="SPY",
        timeframe="1h",
        metadata={
            "num_samples": len(df_train),
            "model_params": 145_678,
            "note": "Direction labeling example"
        }
    )
    logger.info(f"   ✓ Direction training report: {dir_train_report}")

    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("✓ Demo completed successfully!")
    logger.info("=" * 70)

    logger.info("\nGenerated 6 example reports in results/examples/:")
    logger.info("  1. Training report (reversal labels)")
    logger.info("  2. Validation report (predictions + probabilities)")
    logger.info("  3. Test report (predictions + probabilities)")
    logger.info("  4. Ensemble report (3 models combined)")
    logger.info("  5. Continuation training report")
    logger.info("  6. Direction training report")

    logger.info("\n💡 Open these HTML files in your browser to see:")
    logger.info("   - Interactive candlestick charts")
    logger.info("   - Color-coded labels and signals")
    logger.info("   - Probability distributions")
    logger.info("   - Performance metrics tables")
    logger.info("   - Zoom, pan, and hover features")

    logger.info("\n📁 All reports are in: results/examples/")
    logger.info("   (These use synthetic data for demonstration)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
