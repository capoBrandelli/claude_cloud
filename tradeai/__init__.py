"""
TradeAI: Neural Network-Based Trading System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A modular Python application for training neural networks to identify
trade reversals across multiple financial instruments and timeframes.

:copyright: (c) 2025 TradeAI Team
:license: MIT, see LICENSE for more details.
"""

__version__ = "0.1.0"
__author__ = "TradeAI Team"
__license__ = "MIT"

from tradeai.utils.logger import get_logger
from tradeai.utils.config_loader import load_config

# Initialize package-level logger
logger = get_logger(__name__)

__all__ = [
    "__version__",
    "get_logger",
    "load_config",
]
