"""
Plotly-based HTML report generation for TradeAI

This module creates interactive HTML reports with candlestick charts,
labels, signals, probabilities, and performance metrics.
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tradeAI.utils.logger import get_logger

logger = get_logger(__name__)


class TradeAIReportGenerator:
    """
    Generate interactive HTML reports for training, validation, and backtesting.

    Reports include:
    - Candlestick charts with volume
    - Labels/signals overlaid
    - Probabilities and confidence scores
    - Performance metrics
    - Clear metadata and navigation
    """

    def __init__(self, output_dir: str = "results"):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save HTML reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Report generator initialized. Output directory: {self.output_dir}")

    def generate_training_report(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        label_type: str,
        model_name: str,
        symbol: str,
        timeframe: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate training report with candlesticks and labels.

        Args:
            df: OHLCV DataFrame with datetime index
            labels: Training labels (same length as df)
            label_type: Type of labels ('reversal', 'continuation', 'direction')
            model_name: Name of the model
            symbol: Trading symbol
            timeframe: Timeframe
            metadata: Additional metadata to include

        Returns:
            Path to generated HTML file
        """
        logger.info(f"Generating training report for {symbol} {timeframe}")

        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(
                f'{symbol} {timeframe} - Candlestick with {label_type.capitalize()} Labels',
                'Volume',
                'Label Distribution'
            )
        )

        # 1. Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )

        # 2. Add label markers
        self._add_label_markers(fig, df, labels, label_type, row=1)

        # 3. Volume
        colors = ['green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red'
                  for i in range(len(df))]

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )

        # 4. Label distribution over time (rolling count)
        self._add_label_distribution(fig, df.index, labels, row=3)

        # Update layout
        title = self._create_title(
            f"Training Report - {model_name}",
            symbol,
            timeframe,
            df['timestamp'].iloc[0] if 'timestamp' in df.columns else df.index[0],
            df['timestamp'].iloc[-1] if 'timestamp' in df.columns else df.index[-1],
            metadata
        )

        fig.update_layout(
            title=title,
            xaxis3_title="Date",
            yaxis_title="Price",
            yaxis2_title="Volume",
            yaxis3_title="Label Count",
            height=1200,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )

        # Add performance metrics
        metrics_html = self._create_training_metrics_table(df, labels, label_type, metadata)

        # Save HTML
        filename = self._generate_filename(
            f"training_{label_type}_{model_name}",
            symbol,
            timeframe
        )
        filepath = self.output_dir / filename

        # Combine figure and metrics
        full_html = self._combine_figure_and_metrics(fig, metrics_html, title)

        with open(filepath, 'w') as f:
            f.write(full_html)

        logger.info(f"Training report saved to: {filepath}")
        return str(filepath)

    def generate_prediction_report(
        self,
        df: pd.DataFrame,
        true_labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        model_name: str,
        symbol: str,
        timeframe: str,
        split_type: str = "validation",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate validation/test report with predictions and probabilities.

        Args:
            df: OHLCV DataFrame
            true_labels: True labels
            predictions: Model predictions
            probabilities: Prediction probabilities (N, num_classes)
            model_name: Name of the model
            symbol: Trading symbol
            timeframe: Timeframe
            split_type: 'validation' or 'test'
            metadata: Additional metadata

        Returns:
            Path to generated HTML file
        """
        logger.info(f"Generating {split_type} report for {symbol} {timeframe}")

        # Create figure with subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=(
                f'{symbol} {timeframe} - Predictions vs True Labels',
                'Volume',
                'Prediction Confidence',
                'Class Probabilities'
            )
        )

        # 1. Candlestick with predictions
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )

        # 2. Add prediction markers
        self._add_prediction_markers(fig, df, true_labels, predictions, row=1)

        # 3. Volume
        colors = ['green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red'
                  for i in range(len(df))]

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )

        # 4. Prediction confidence (max probability)
        confidence = np.max(probabilities, axis=1)

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=confidence,
                name='Confidence',
                line=dict(color='purple', width=2),
                fill='tozeroy',
                fillcolor='rgba(128, 0, 128, 0.2)'
            ),
            row=3, col=1
        )

        # 5. Class probabilities
        self._add_class_probabilities(fig, df.index, probabilities, row=4)

        # Update layout
        title = self._create_title(
            f"{split_type.capitalize()} Report - {model_name}",
            symbol,
            timeframe,
            df['timestamp'].iloc[0] if 'timestamp' in df.columns else df.index[0],
            df['timestamp'].iloc[-1] if 'timestamp' in df.columns else df.index[-1],
            metadata
        )

        fig.update_layout(
            title=title,
            xaxis4_title="Date",
            yaxis_title="Price",
            yaxis2_title="Volume",
            yaxis3_title="Confidence",
            yaxis4_title="Probability",
            height=1400,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )

        # Add performance metrics
        metrics_html = self._create_prediction_metrics_table(
            true_labels, predictions, probabilities, split_type, metadata
        )

        # Save HTML
        filename = self._generate_filename(
            f"{split_type}_{model_name}",
            symbol,
            timeframe
        )
        filepath = self.output_dir / filename

        full_html = self._combine_figure_and_metrics(fig, metrics_html, title)

        with open(filepath, 'w') as f:
            f.write(full_html)

        logger.info(f"{split_type.capitalize()} report saved to: {filepath}")
        return str(filepath)

    def generate_ensemble_report(
        self,
        df: pd.DataFrame,
        reversal_probs: np.ndarray,
        continuation_probs: np.ndarray,
        direction_probs: np.ndarray,
        combined_signals: np.ndarray,
        confidence: np.ndarray,
        agreement: np.ndarray,
        symbol: str,
        timeframe: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate ensemble report with all three models' predictions.

        Args:
            df: OHLCV DataFrame
            reversal_probs: Reversal model probabilities
            continuation_probs: Continuation model probabilities
            direction_probs: Direction model probabilities
            combined_signals: Combined signals (-1, 0, 1)
            confidence: Confidence scores
            agreement: Agreement scores
            symbol: Trading symbol
            timeframe: Timeframe
            metadata: Additional metadata

        Returns:
            Path to generated HTML file
        """
        logger.info(f"Generating ensemble report for {symbol} {timeframe}")

        # Create figure with subplots
        fig = make_subplots(
            rows=6, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.35, 0.12, 0.15, 0.13, 0.13, 0.12],
            subplot_titles=(
                f'{symbol} {timeframe} - Ensemble Predictions',
                'Volume',
                'Combined Signals',
                'Reversal Probabilities',
                'Continuation Probabilities',
                'Direction Probabilities'
            )
        )

        # 1. Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )

        # Add signal markers
        self._add_signal_markers(fig, df, combined_signals, confidence, row=1)

        # 2. Volume
        colors = ['green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red'
                  for i in range(len(df))]

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )

        # 3. Combined signals with confidence and agreement
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=combined_signals,
                name='Signal',
                line=dict(color='blue', width=2),
                mode='lines+markers'
            ),
            row=3, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=confidence,
                name='Confidence',
                line=dict(color='purple', width=1, dash='dot'),
                yaxis='y3'
            ),
            row=3, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=agreement,
                name='Agreement',
                line=dict(color='orange', width=1, dash='dash'),
                yaxis='y3'
            ),
            row=3, col=1
        )

        # 4-6. Model probabilities
        self._add_class_probabilities(fig, df.index, reversal_probs, row=4, name_prefix='Rev')
        self._add_class_probabilities(fig, df.index, continuation_probs, row=5, name_prefix='Cont')
        self._add_class_probabilities(fig, df.index, direction_probs, row=6, name_prefix='Dir')

        # Update layout
        title = self._create_title(
            "Ensemble Report - Three-Model System",
            symbol,
            timeframe,
            df['timestamp'].iloc[0] if 'timestamp' in df.columns else df.index[0],
            df['timestamp'].iloc[-1] if 'timestamp' in df.columns else df.index[-1],
            metadata
        )

        fig.update_layout(
            title=title,
            xaxis6_title="Date",
            yaxis_title="Price",
            yaxis2_title="Volume",
            yaxis3_title="Signal",
            yaxis4_title="Probability",
            yaxis5_title="Probability",
            yaxis6_title="Probability",
            height=1800,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )

        # Add performance metrics
        metrics_html = self._create_ensemble_metrics_table(
            combined_signals, confidence, agreement, metadata
        )

        # Save HTML
        filename = self._generate_filename(
            "ensemble",
            symbol,
            timeframe
        )
        filepath = self.output_dir / filename

        full_html = self._combine_figure_and_metrics(fig, metrics_html, title)

        with open(filepath, 'w') as f:
            f.write(full_html)

        logger.info(f"Ensemble report saved to: {filepath}")
        return str(filepath)

    def _add_label_markers(
        self,
        fig: go.Figure,
        df: pd.DataFrame,
        labels: np.ndarray,
        label_type: str,
        row: int
    ):
        """Add label markers to candlestick chart."""
        # Define label mappings
        label_configs = {
            'reversal': {
                0: ('Strong Bearish Reversal', 'red', 'triangle-down'),
                1: ('Weak Bearish Reversal', 'orange', 'triangle-down'),
                2: ('Neutral', 'gray', 'circle'),
                3: ('Weak Bullish Reversal', 'lightgreen', 'triangle-up'),
                4: ('Strong Bullish Reversal', 'darkgreen', 'triangle-up'),
            },
            'continuation': {
                0: ('Strong Break', 'red', 'x'),
                1: ('Weak Break', 'orange', 'x'),
                2: ('Continue', 'blue', 'circle'),
                3: ('Weak Continue', 'lightblue', 'circle'),
                4: ('Strong Continue', 'darkblue', 'circle'),
            },
            'direction': {
                0: ('Strong Down', 'darkred', 'triangle-down'),
                1: ('Weak Down', 'red', 'triangle-down'),
                2: ('Neutral', 'gray', 'circle'),
                3: ('Weak Up', 'lightgreen', 'triangle-up'),
                4: ('Strong Up', 'darkgreen', 'triangle-up'),
            }
        }

        config = label_configs.get(label_type, label_configs['reversal'])

        # Add markers for each label class
        for label_val, (name, color, symbol) in config.items():
            mask = labels == label_val
            if mask.sum() == 0:
                continue

            fig.add_trace(
                go.Scatter(
                    x=df.index[mask],
                    y=df['high'][mask] * 1.02,  # Slightly above high
                    mode='markers',
                    name=name,
                    marker=dict(
                        symbol=symbol,
                        size=10,
                        color=color,
                        line=dict(width=1, color='black')
                    ),
                    showlegend=True
                ),
                row=row, col=1
            )

    def _add_prediction_markers(
        self,
        fig: go.Figure,
        df: pd.DataFrame,
        true_labels: np.ndarray,
        predictions: np.ndarray,
        row: int
    ):
        """Add prediction markers showing correct/incorrect predictions."""
        correct_mask = true_labels == predictions
        incorrect_mask = ~correct_mask

        # Correct predictions (green checkmarks)
        if correct_mask.sum() > 0:
            fig.add_trace(
                go.Scatter(
                    x=df.index[correct_mask],
                    y=df['high'][correct_mask] * 1.02,
                    mode='markers',
                    name='Correct',
                    marker=dict(
                        symbol='circle',
                        size=8,
                        color='green',
                        line=dict(width=1, color='darkgreen')
                    )
                ),
                row=row, col=1
            )

        # Incorrect predictions (red X)
        if incorrect_mask.sum() > 0:
            fig.add_trace(
                go.Scatter(
                    x=df.index[incorrect_mask],
                    y=df['high'][incorrect_mask] * 1.02,
                    mode='markers',
                    name='Incorrect',
                    marker=dict(
                        symbol='x',
                        size=10,
                        color='red',
                        line=dict(width=2)
                    )
                ),
                row=row, col=1
            )

    def _add_signal_markers(
        self,
        fig: go.Figure,
        df: pd.DataFrame,
        signals: np.ndarray,
        confidence: np.ndarray,
        row: int
    ):
        """Add entry/exit signal markers."""
        # Buy signals (signal == 1)
        buy_mask = signals == 1
        high_conf_buy = buy_mask & (confidence > 0.7)
        low_conf_buy = buy_mask & (confidence <= 0.7)

        if high_conf_buy.sum() > 0:
            fig.add_trace(
                go.Scatter(
                    x=df.index[high_conf_buy],
                    y=df['low'][high_conf_buy] * 0.98,
                    mode='markers',
                    name='BUY (High Conf)',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color='darkgreen',
                        line=dict(width=2, color='black')
                    )
                ),
                row=row, col=1
            )

        if low_conf_buy.sum() > 0:
            fig.add_trace(
                go.Scatter(
                    x=df.index[low_conf_buy],
                    y=df['low'][low_conf_buy] * 0.98,
                    mode='markers',
                    name='BUY (Low Conf)',
                    marker=dict(
                        symbol='triangle-up',
                        size=10,
                        color='lightgreen',
                        line=dict(width=1, color='green')
                    )
                ),
                row=row, col=1
            )

        # Sell signals (signal == -1)
        sell_mask = signals == -1
        high_conf_sell = sell_mask & (confidence > 0.7)
        low_conf_sell = sell_mask & (confidence <= 0.7)

        if high_conf_sell.sum() > 0:
            fig.add_trace(
                go.Scatter(
                    x=df.index[high_conf_sell],
                    y=df['high'][high_conf_sell] * 1.02,
                    mode='markers',
                    name='SELL (High Conf)',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color='darkred',
                        line=dict(width=2, color='black')
                    )
                ),
                row=row, col=1
            )

        if low_conf_sell.sum() > 0:
            fig.add_trace(
                go.Scatter(
                    x=df.index[low_conf_sell],
                    y=df['high'][low_conf_sell] * 1.02,
                    mode='markers',
                    name='SELL (Low Conf)',
                    marker=dict(
                        symbol='triangle-down',
                        size=10,
                        color='orange',
                        line=dict(width=1, color='red')
                    )
                ),
                row=row, col=1
            )

    def _add_label_distribution(
        self,
        fig: go.Figure,
        index: pd.DatetimeIndex,
        labels: np.ndarray,
        row: int,
        window: int = 50
    ):
        """Add rolling label distribution."""
        df_labels = pd.DataFrame({'label': labels}, index=index)

        for label_val in range(5):
            rolling_count = (df_labels['label'] == label_val).rolling(window=window).sum()

            fig.add_trace(
                go.Scatter(
                    x=index,
                    y=rolling_count,
                    name=f'Class {label_val}',
                    mode='lines',
                    stackgroup='one'
                ),
                row=row, col=1
            )

    def _add_class_probabilities(
        self,
        fig: go.Figure,
        index: pd.DatetimeIndex,
        probabilities: np.ndarray,
        row: int,
        name_prefix: str = ''
    ):
        """Add class probability traces."""
        num_classes = probabilities.shape[1]
        colors = ['darkred', 'red', 'gray', 'lightgreen', 'darkgreen']

        for i in range(num_classes):
            fig.add_trace(
                go.Scatter(
                    x=index,
                    y=probabilities[:, i],
                    name=f'{name_prefix} Class {i}',
                    mode='lines',
                    line=dict(width=1),
                    stackgroup='one' if num_classes > 3 else None,
                    fillcolor=colors[i] if i < len(colors) else None
                ),
                row=row, col=1
            )

    def _create_title(
        self,
        base_title: str,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create detailed report title."""
        title = f"{base_title}<br>"
        title += f"<sub>Symbol: {symbol} | Timeframe: {timeframe} | "
        title += f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

        if metadata:
            if 'num_samples' in metadata:
                title += f" | Samples: {metadata['num_samples']}"
            if 'model_params' in metadata:
                title += f" | Parameters: {metadata['model_params']:,}"

        title += f"<br>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</sub>"

        return title

    def _generate_filename(
        self,
        prefix: str,
        symbol: str,
        timeframe: str
    ) -> str:
        """Generate unique filename."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{prefix}_{symbol}_{timeframe}_{timestamp}.html"

    def _create_training_metrics_table(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        label_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create HTML table with training metrics."""
        # Count labels
        unique, counts = np.unique(labels[labels != -1], return_counts=True)

        html = "<h2>Training Metrics</h2>"
        html += "<table border='1' style='border-collapse: collapse; width: 100%; margin: 20px 0;'>"
        html += "<tr style='background-color: #f0f0f0;'><th>Metric</th><th>Value</th></tr>"

        html += f"<tr><td>Label Type</td><td>{label_type.capitalize()}</td></tr>"
        html += f"<tr><td>Total Samples</td><td>{len(df)}</td></tr>"
        html += f"<tr><td>Labeled Samples</td><td>{len(labels[labels != -1])}</td></tr>"
        html += f"<tr><td>Unlabeled Samples</td><td>{len(labels[labels == -1])}</td></tr>"

        html += "<tr><td colspan='2' style='background-color: #e0e0e0; font-weight: bold;'>Label Distribution</td></tr>"
        for label_val, count in zip(unique, counts):
            pct = count / len(labels[labels != -1]) * 100
            html += f"<tr><td>Class {int(label_val)}</td><td>{count} ({pct:.1f}%)</td></tr>"

        if metadata:
            html += "<tr><td colspan='2' style='background-color: #e0e0e0; font-weight: bold;'>Additional Info</td></tr>"
            for key, value in metadata.items():
                html += f"<tr><td>{key}</td><td>{value}</td></tr>"

        html += "</table>"
        return html

    def _create_prediction_metrics_table(
        self,
        true_labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        split_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create HTML table with prediction metrics."""
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        accuracy = accuracy_score(true_labels, predictions)
        conf_matrix = confusion_matrix(true_labels, predictions)

        html = f"<h2>{split_type.capitalize()} Metrics</h2>"
        html += "<table border='1' style='border-collapse: collapse; width: 100%; margin: 20px 0;'>"
        html += "<tr style='background-color: #f0f0f0;'><th>Metric</th><th>Value</th></tr>"

        html += f"<tr><td>Accuracy</td><td>{accuracy:.4f}</td></tr>"
        html += f"<tr><td>Total Samples</td><td>{len(true_labels)}</td></tr>"
        html += f"<tr><td>Correct Predictions</td><td>{(true_labels == predictions).sum()}</td></tr>"
        html += f"<tr><td>Incorrect Predictions</td><td>{(true_labels != predictions).sum()}</td></tr>"
        html += f"<tr><td>Mean Confidence</td><td>{np.max(probabilities, axis=1).mean():.4f}</td></tr>"

        # Confusion matrix
        html += "<tr><td colspan='2' style='background-color: #e0e0e0; font-weight: bold;'>Confusion Matrix</td></tr>"
        html += "<tr><td colspan='2'><pre>"
        html += str(conf_matrix)
        html += "</pre></td></tr>"

        if metadata:
            html += "<tr><td colspan='2' style='background-color: #e0e0e0; font-weight: bold;'>Additional Info</td></tr>"
            for key, value in metadata.items():
                html += f"<tr><td>{key}</td><td>{value}</td></tr>"

        html += "</table>"
        return html

    def _create_ensemble_metrics_table(
        self,
        signals: np.ndarray,
        confidence: np.ndarray,
        agreement: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create HTML table with ensemble metrics."""
        html = "<h2>Ensemble Metrics</h2>"
        html += "<table border='1' style='border-collapse: collapse; width: 100%; margin: 20px 0;'>"
        html += "<tr style='background-color: #f0f0f0;'><th>Metric</th><th>Value</th></tr>"

        html += f"<tr><td>Total Signals</td><td>{len(signals)}</td></tr>"
        html += f"<tr><td>Buy Signals</td><td>{(signals == 1).sum()} ({(signals == 1).sum()/len(signals)*100:.1f}%)</td></tr>"
        html += f"<tr><td>Sell Signals</td><td>{(signals == -1).sum()} ({(signals == -1).sum()/len(signals)*100:.1f}%)</td></tr>"
        html += f"<tr><td>Neutral Signals</td><td>{(signals == 0).sum()} ({(signals == 0).sum()/len(signals)*100:.1f}%)</td></tr>"
        html += f"<tr><td>Mean Confidence</td><td>{confidence.mean():.4f}</td></tr>"
        html += f"<tr><td>Mean Agreement</td><td>{agreement.mean():.4f}</td></tr>"
        html += f"<tr><td>High Confidence Signals (>0.7)</td><td>{(confidence > 0.7).sum()}</td></tr>"
        html += f"<tr><td>High Agreement Signals (>0.8)</td><td>{(agreement > 0.8).sum()}</td></tr>"

        if metadata:
            html += "<tr><td colspan='2' style='background-color: #e0e0e0; font-weight: bold;'>Additional Info</td></tr>"
            for key, value in metadata.items():
                html += f"<tr><td>{key}</td><td>{value}</td></tr>"

        html += "</table>"
        return html

    def _combine_figure_and_metrics(
        self,
        fig: go.Figure,
        metrics_html: str,
        title: str
    ) -> str:
        """Combine plotly figure and metrics into single HTML."""
        fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1600px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h2 {{
                    color: #333;
                    border-bottom: 2px solid #007bff;
                    padding-bottom: 10px;
                }}
                table {{
                    font-size: 14px;
                }}
                th {{
                    background-color: #007bff;
                    color: white;
                    padding: 10px;
                    text-align: left;
                }}
                td {{
                    padding: 8px;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                {fig_html}
                <hr style="margin: 40px 0;">
                {metrics_html}
            </div>
        </body>
        </html>
        """

        return full_html
