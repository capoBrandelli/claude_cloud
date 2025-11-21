"""
Ensemble models package

This package contains ensemble model implementations and utilities
for combining multiple models to create robust trading predictions.
"""

from tradeAI.models.ensemble.signal_combiner import (
    SignalCombiner,
    RuleBasedCombiner,
    WeightedCombiner,
    MetaModelCombiner,
)

__all__ = [
    "SignalCombiner",
    "RuleBasedCombiner",
    "WeightedCombiner",
    "MetaModelCombiner",
]
