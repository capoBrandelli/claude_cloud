"""
Signal combination methods for ensemble models

This module provides different strategies to combine predictions from
the three-model ensemble (reversal, continuation, direction).

All combination methods operate on model predictions (not raw data),
so they maintain forward-looking bias prevention.
"""

from typing import Dict, List, Tuple, Optional, Literal
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from tradeAI.utils.logger import get_logger

logger = get_logger(__name__)


class SignalCombiner(ABC):
    """
    Abstract base class for signal combination strategies.

    All combiners take predictions from three models and produce
    a combined signal with confidence score.
    """

    @abstractmethod
    def combine(
        self,
        reversal_probs: np.ndarray,
        continuation_probs: np.ndarray,
        direction_probs: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Combine predictions from three models.

        Args:
            reversal_probs: Reversal model probabilities (N, num_classes)
            continuation_probs: Continuation model probabilities (N, num_classes)
            direction_probs: Direction model probabilities (N, num_classes)

        Returns:
            Dictionary with:
                - 'signal': Combined signal (N,)
                - 'confidence': Confidence score 0-1 (N,)
                - 'agreement': Model agreement score 0-1 (N,)
        """
        pass


class RuleBasedCombiner(SignalCombiner):
    """
    Rule-based signal combination using trading logic.

    Uses explicit trading rules to combine model predictions:
    - High reversal + Low continuation = Strong reversal signal
    - High continuation + Aligned direction = Strong trend signal
    - Model disagreement = Low confidence / no signal
    """

    def __init__(
        self,
        num_classes: int = 5,
        reversal_threshold: float = 0.6,
        continuation_threshold: float = 0.6,
        direction_threshold: float = 0.5,
    ):
        """
        Initialize rule-based combiner.

        Args:
            num_classes: Number of classes (must match model outputs)
            reversal_threshold: Minimum probability for reversal signal
            continuation_threshold: Minimum probability for continuation signal
            direction_threshold: Minimum probability for direction signal
        """
        self.num_classes = num_classes
        self.reversal_threshold = reversal_threshold
        self.continuation_threshold = continuation_threshold
        self.direction_threshold = direction_threshold

        logger.info(
            f"RuleBasedCombiner initialized: {num_classes} classes, "
            f"thresholds: R={reversal_threshold}, C={continuation_threshold}, D={direction_threshold}"
        )

    def combine(
        self,
        reversal_probs: np.ndarray,
        continuation_probs: np.ndarray,
        direction_probs: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Combine using trading rules.

        Rules:
        1. Strong Bullish Reversal:
           - High reversal prob for bullish class (4)
           - Low continuation prob OR continuation shows break
           - Direction aligned bullish

        2. Strong Bearish Reversal:
           - High reversal prob for bearish class (0)
           - Low continuation prob OR continuation shows break
           - Direction aligned bearish

        3. Strong Bullish Continuation:
           - High continuation prob (4)
           - Low reversal prob
           - Direction aligned bullish

        4. Strong Bearish Continuation:
           - High continuation prob (4)
           - Low reversal prob
           - Direction aligned bearish

        5. Neutral/No Signal:
           - Models disagree
           - Low confidence across all models

        Args:
            reversal_probs: (N, num_classes)
            continuation_probs: (N, num_classes)
            direction_probs: (N, num_classes)

        Returns:
            Dictionary with signals, confidence, and agreement scores
        """
        N = reversal_probs.shape[0]

        # Initialize outputs
        signal = np.zeros(N)  # 1: bullish, -1: bearish, 0: neutral
        confidence = np.zeros(N)
        agreement = np.zeros(N)

        for i in range(N):
            r_probs = reversal_probs[i]
            c_probs = continuation_probs[i]
            d_probs = direction_probs[i]

            # Get max probability classes
            r_class = np.argmax(r_probs)
            c_class = np.argmax(c_probs)
            d_class = np.argmax(d_probs)

            r_conf = r_probs[r_class]
            c_conf = c_probs[c_class]
            d_conf = d_probs[d_class]

            # Calculate agreement (how well models align)
            # For 5-class: 0=strong bear, 1=weak bear, 2=neutral, 3=weak bull, 4=strong bull
            classes = np.array([r_class, c_class, d_class])
            agreement_score = 1.0 - (classes.std() / (self.num_classes - 1))
            agreement[i] = agreement_score

            # Rule 1: Strong Bullish Reversal
            if (
                r_class >= 3  # Bullish reversal (3 or 4)
                and r_conf >= self.reversal_threshold
                and c_class <= 2  # Continuation weak or break
                and d_class >= 3  # Direction bullish
                and d_conf >= self.direction_threshold
            ):
                signal[i] = 1  # Bullish
                confidence[i] = (r_conf + d_conf) / 2 * agreement_score

            # Rule 2: Strong Bearish Reversal
            elif (
                r_class <= 1  # Bearish reversal (0 or 1)
                and r_conf >= self.reversal_threshold
                and c_class <= 2  # Continuation weak or break
                and d_class <= 1  # Direction bearish
                and d_conf >= self.direction_threshold
            ):
                signal[i] = -1  # Bearish
                confidence[i] = (r_conf + d_conf) / 2 * agreement_score

            # Rule 3: Strong Bullish Continuation
            elif (
                c_class == 4  # Strong continuation
                and c_conf >= self.continuation_threshold
                and r_class >= 2  # No strong bearish reversal
                and d_class >= 3  # Direction bullish
                and d_conf >= self.direction_threshold
            ):
                signal[i] = 1  # Bullish
                confidence[i] = (c_conf + d_conf) / 2 * agreement_score

            # Rule 4: Strong Bearish Continuation
            elif (
                c_class == 4  # Strong continuation (in bearish trend)
                and c_conf >= self.continuation_threshold
                and r_class <= 2  # No strong bullish reversal
                and d_class <= 1  # Direction bearish
                and d_conf >= self.direction_threshold
            ):
                signal[i] = -1  # Bearish
                confidence[i] = (c_conf + d_conf) / 2 * agreement_score

            # Rule 5: Weak Signals (lower confidence)
            elif agreement_score > 0.7:  # Models mostly agree
                # Average signal based on class positions
                avg_class = (r_class + c_class + d_class) / 3
                if avg_class > 3:
                    signal[i] = 1  # Bullish
                    confidence[i] = (r_conf + c_conf + d_conf) / 3 * 0.5  # Lower confidence
                elif avg_class < 1.5:
                    signal[i] = -1  # Bearish
                    confidence[i] = (r_conf + c_conf + d_conf) / 3 * 0.5
                else:
                    signal[i] = 0  # Neutral
                    confidence[i] = 0.3

            else:
                # Models disagree - no signal
                signal[i] = 0
                confidence[i] = 0.2  # Low confidence

        logger.debug(
            f"Rule-based combination: "
            f"Bullish={np.sum(signal == 1)}, "
            f"Bearish={np.sum(signal == -1)}, "
            f"Neutral={np.sum(signal == 0)}, "
            f"Avg confidence={confidence.mean():.3f}"
        )

        return {
            "signal": signal,
            "confidence": confidence,
            "agreement": agreement,
        }


class WeightedCombiner(SignalCombiner):
    """
    Weighted average combination of model predictions.

    Learns optimal weights for each model based on validation performance.
    Can also use fixed weights if specified.
    """

    def __init__(
        self,
        num_classes: int = 5,
        weights: Optional[Dict[str, float]] = None,
        normalize_weights: bool = True,
    ):
        """
        Initialize weighted combiner.

        Args:
            num_classes: Number of classes
            weights: Optional fixed weights {'reversal': w1, 'continuation': w2, 'direction': w3}
                     If None, will use equal weights
            normalize_weights: Whether to normalize weights to sum to 1
        """
        self.num_classes = num_classes

        if weights is None:
            weights = {"reversal": 1.0, "continuation": 1.0, "direction": 1.0}

        self.weights = weights

        if normalize_weights:
            total = sum(weights.values())
            self.weights = {k: v / total for k, v in weights.items()}

        logger.info(
            f"WeightedCombiner initialized: {num_classes} classes, "
            f"weights={self.weights}"
        )

    def combine(
        self,
        reversal_probs: np.ndarray,
        continuation_probs: np.ndarray,
        direction_probs: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Combine using weighted average of probabilities.

        Args:
            reversal_probs: (N, num_classes)
            continuation_probs: (N, num_classes)
            direction_probs: (N, num_classes)

        Returns:
            Dictionary with signals, confidence, and agreement scores
        """
        # Weighted average of probabilities
        combined_probs = (
            reversal_probs * self.weights["reversal"] +
            continuation_probs * self.weights["continuation"] +
            direction_probs * self.weights["direction"]
        )

        # Get predicted class and confidence
        predicted_class = np.argmax(combined_probs, axis=1)
        max_prob = np.max(combined_probs, axis=1)

        # Convert class to signal
        # For 5 classes: 0,1=bearish, 2=neutral, 3,4=bullish
        signal = np.zeros(len(predicted_class))
        signal[predicted_class <= 1] = -1  # Bearish
        signal[predicted_class >= 3] = 1  # Bullish

        # Confidence is the max probability
        confidence = max_prob

        # Calculate agreement (entropy-based)
        # Low entropy = high agreement
        def calc_entropy(probs):
            """Calculate normalized entropy (0=certain, 1=uniform)"""
            p = probs[probs > 0]  # Remove zeros
            if len(p) == 0:
                return 1.0
            entropy = -np.sum(p * np.log(p))
            max_entropy = np.log(len(probs))
            return entropy / max_entropy if max_entropy > 0 else 0

        agreement = np.array([1.0 - calc_entropy(p) for p in combined_probs])

        logger.debug(
            f"Weighted combination: "
            f"Bullish={np.sum(signal == 1)}, "
            f"Bearish={np.sum(signal == -1)}, "
            f"Neutral={np.sum(signal == 0)}, "
            f"Avg confidence={confidence.mean():.3f}"
        )

        return {
            "signal": signal,
            "confidence": confidence,
            "agreement": agreement,
        }

    def optimize_weights(
        self,
        reversal_probs: np.ndarray,
        continuation_probs: np.ndarray,
        direction_probs: np.ndarray,
        true_labels: np.ndarray,
        method: Literal["grid", "gradient"] = "grid",
    ) -> Dict[str, float]:
        """
        Learn optimal weights based on validation data.

        Args:
            reversal_probs: Reversal predictions on validation set
            continuation_probs: Continuation predictions on validation set
            direction_probs: Direction predictions on validation set
            true_labels: True labels (use any of the three label sets)
            method: Optimization method ('grid' or 'gradient')

        Returns:
            Dictionary with optimal weights
        """
        if method == "grid":
            # Grid search over weight combinations
            best_accuracy = 0
            best_weights = None

            for w1 in np.linspace(0, 1, 11):
                for w2 in np.linspace(0, 1, 11):
                    w3 = 1.0 - w1 - w2
                    if w3 < 0 or w3 > 1:
                        continue

                    weights = {"reversal": w1, "continuation": w2, "direction": w3}
                    self.weights = weights

                    result = self.combine(reversal_probs, continuation_probs, direction_probs)
                    predicted_class = np.argmax(
                        reversal_probs * w1 + continuation_probs * w2 + direction_probs * w3,
                        axis=1
                    )

                    accuracy = (predicted_class == true_labels).mean()

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_weights = weights

            self.weights = best_weights
            logger.info(f"Optimized weights (grid): {best_weights}, accuracy={best_accuracy:.3f}")

        return self.weights


class MetaModelCombiner(SignalCombiner):
    """
    Meta-model (stacking) combination.

    Trains a second-level model to combine predictions from the three base models.
    The meta-model learns how to optimally weight and combine predictions.

    IMPORTANT: Meta-model must be trained on VALIDATION predictions only,
    never on training predictions (to prevent overfitting).
    """

    def __init__(
        self,
        num_classes: int = 5,
        meta_model: Optional[any] = None,
    ):
        """
        Initialize meta-model combiner.

        Args:
            num_classes: Number of classes
            meta_model: Optional pre-trained meta-model (if None, use LogisticRegression)
        """
        self.num_classes = num_classes

        if meta_model is None:
            # Default: logistic regression
            self.meta_model = LogisticRegression(
                multi_class="multinomial",
                max_iter=1000,
                random_state=42,
            )
        else:
            self.meta_model = meta_model

        self.is_fitted = False

        logger.info(f"MetaModelCombiner initialized: {num_classes} classes")

    def fit(
        self,
        reversal_probs: np.ndarray,
        continuation_probs: np.ndarray,
        direction_probs: np.ndarray,
        true_labels: np.ndarray,
    ) -> "MetaModelCombiner":
        """
        Train meta-model on validation predictions.

        CRITICAL: Only use validation predictions, NOT training predictions!

        Args:
            reversal_probs: Reversal predictions on VALIDATION set
            continuation_probs: Continuation predictions on VALIDATION set
            direction_probs: Direction predictions on VALIDATION set
            true_labels: True labels for validation set

        Returns:
            Self (for chaining)
        """
        # Concatenate all predictions as features for meta-model
        meta_features = np.concatenate(
            [reversal_probs, continuation_probs, direction_probs],
            axis=1
        )

        # Train meta-model
        self.meta_model.fit(meta_features, true_labels)
        self.is_fitted = True

        # Calculate training accuracy
        predictions = self.meta_model.predict(meta_features)
        accuracy = (predictions == true_labels).mean()

        logger.info(f"Meta-model trained: validation accuracy={accuracy:.3f}")

        return self

    def combine(
        self,
        reversal_probs: np.ndarray,
        continuation_probs: np.ndarray,
        direction_probs: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Combine using meta-model predictions.

        Args:
            reversal_probs: (N, num_classes)
            continuation_probs: (N, num_classes)
            direction_probs: (N, num_classes)

        Returns:
            Dictionary with signals, confidence, and agreement scores
        """
        if not self.is_fitted:
            raise ValueError("Meta-model must be fitted before combining. Call fit() first.")

        # Concatenate predictions
        meta_features = np.concatenate(
            [reversal_probs, continuation_probs, direction_probs],
            axis=1
        )

        # Get meta-model predictions
        predicted_class = self.meta_model.predict(meta_features)
        predicted_probs = self.meta_model.predict_proba(meta_features)

        # Convert class to signal
        signal = np.zeros(len(predicted_class))
        signal[predicted_class <= 1] = -1  # Bearish
        signal[predicted_class >= 3] = 1  # Bullish

        # Confidence is the max probability from meta-model
        confidence = np.max(predicted_probs, axis=1)

        # Agreement: how consistent are base models with meta-model prediction
        agreement = np.zeros(len(predicted_class))
        for i in range(len(predicted_class)):
            # Check if base models agree with meta prediction
            r_agree = np.argmax(reversal_probs[i]) == predicted_class[i]
            c_agree = np.argmax(continuation_probs[i]) == predicted_class[i]
            d_agree = np.argmax(direction_probs[i]) == predicted_class[i]

            agreement[i] = (r_agree + c_agree + d_agree) / 3

        logger.debug(
            f"Meta-model combination: "
            f"Bullish={np.sum(signal == 1)}, "
            f"Bearish={np.sum(signal == -1)}, "
            f"Neutral={np.sum(signal == 0)}, "
            f"Avg confidence={confidence.mean():.3f}"
        )

        return {
            "signal": signal,
            "confidence": confidence,
            "agreement": agreement,
        }


# Convenience function
def create_combiner(
    method: Literal["rule", "weighted", "meta"] = "rule",
    num_classes: int = 5,
    **kwargs
) -> SignalCombiner:
    """
    Factory function to create signal combiner.

    Args:
        method: Combination method
        num_classes: Number of classes
        **kwargs: Additional arguments for specific combiner

    Returns:
        SignalCombiner instance
    """
    if method == "rule":
        return RuleBasedCombiner(num_classes=num_classes, **kwargs)
    elif method == "weighted":
        return WeightedCombiner(num_classes=num_classes, **kwargs)
    elif method == "meta":
        return MetaModelCombiner(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown combination method: {method}")
