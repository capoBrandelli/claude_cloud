"""
Visualization module for TradeAI

Provides interactive HTML reports with Plotly for:
- Training reports with labels
- Validation/test reports with predictions
- Ensemble reports with multi-model predictions
- Backtest reports with trading signals
"""

from tradeAI.visualization.plotly_reports import TradeAIReportGenerator

__all__ = ["TradeAIReportGenerator"]
