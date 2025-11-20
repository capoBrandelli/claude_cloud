"""
Sentiment Comparison Engine
Compares News sentiment vs Futures sentiment to identify alignment and disagreement
"""
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import statistics


class SentimentComparator:
    """Compares news sentiment with futures sentiment to identify alignment/disagreement"""

    def __init__(self):
        self.alignment_thresholds = {
            'strong_alignment': 0.8,
            'moderate_alignment': 0.6,
            'weak_alignment': 0.4,
            'divergence': 0.2
        }

    def calculate_alignment(self, news_sentiment: float,
                           futures_sentiment: float) -> Dict:
        """
        Calculate alignment between news and futures sentiment

        Args:
            news_sentiment: Sentiment score from news (-1 to 1)
            futures_sentiment: Sentiment score from futures (-1 to 1)

        Returns:
            Dictionary with alignment metrics
        """
        # Calculate difference and correlation
        difference = abs(news_sentiment - futures_sentiment)

        # Alignment score: 1 means perfect alignment, 0 means maximum divergence
        alignment_score = 1 - (difference / 2.0)

        # Check if both point in same direction
        same_direction = (news_sentiment * futures_sentiment) >= 0

        # Determine alignment status
        if alignment_score >= self.alignment_thresholds['strong_alignment']:
            status = "Strong Alignment"
            description = "News and futures sentiment strongly agree"
        elif alignment_score >= self.alignment_thresholds['moderate_alignment']:
            status = "Moderate Alignment"
            description = "News and futures sentiment moderately agree"
        elif alignment_score >= self.alignment_thresholds['weak_alignment']:
            status = "Weak Alignment"
            description = "News and futures sentiment show slight agreement"
        elif same_direction:
            status = "Mild Divergence"
            description = "Both signals point same direction but with different magnitudes"
        else:
            status = "Strong Divergence"
            description = "News and futures sentiment strongly disagree"

        # Calculate divergence magnitude (only if divergent)
        if not same_direction:
            divergence_magnitude = difference
        else:
            divergence_magnitude = 0.0

        return {
            'alignment_score': round(alignment_score, 3),
            'alignment_status': status,
            'description': description,
            'same_direction': same_direction,
            'divergence_magnitude': round(divergence_magnitude, 3),
            'difference': round(difference, 3)
        }

    def analyze_sentiment_group(self, news_data: List[Dict],
                                futures_data: List[Dict]) -> Dict:
        """
        Analyze a group of news and futures data for a specific symbol/market

        Args:
            news_data: List of news sentiment records
            futures_data: List of futures sentiment records

        Returns:
            Comprehensive comparison analysis
        """
        if not news_data and not futures_data:
            return {
                'news_sentiment_avg': 0.0,
                'futures_sentiment_avg': 0.0,
                'alignment_score': 0.0,
                'alignment_status': 'Insufficient Data',
                'confidence_score': 0.0,
                'insights': 'No data available for analysis'
            }

        # Calculate average news sentiment
        news_sentiments = [item['sentiment_score'] for item in news_data]
        news_avg = statistics.mean(news_sentiments) if news_sentiments else 0.0
        news_std = statistics.stdev(news_sentiments) if len(news_sentiments) > 1 else 0.0

        # Calculate average futures sentiment
        futures_sentiments = [item['sentiment_score'] for item in futures_data]
        futures_avg = statistics.mean(futures_sentiments) if futures_sentiments else 0.0
        futures_std = statistics.stdev(futures_sentiments) if len(futures_sentiments) > 1 else 0.0

        # Calculate alignment
        alignment = self.calculate_alignment(news_avg, futures_avg)

        # Calculate confidence score based on data quantity and consistency
        confidence_score = self._calculate_confidence(
            len(news_data), len(futures_data), news_std, futures_std
        )

        # Generate insights
        insights = self._generate_insights(
            news_avg, futures_avg, alignment, news_data, futures_data
        )

        return {
            'news_sentiment_avg': round(news_avg, 3),
            'futures_sentiment_avg': round(futures_avg, 3),
            'alignment_score': alignment['alignment_score'],
            'alignment_status': alignment['alignment_status'],
            'divergence_magnitude': alignment['divergence_magnitude'],
            'confidence_score': round(confidence_score, 3),
            'news_count': len(news_data),
            'futures_count': len(futures_data),
            'news_std': round(news_std, 3),
            'futures_std': round(futures_std, 3),
            'insights': insights
        }

    def _calculate_confidence(self, news_count: int, futures_count: int,
                             news_std: float, futures_std: float) -> float:
        """Calculate confidence score for the analysis"""
        # Base confidence on data quantity
        data_confidence = min(1.0, (news_count + futures_count) / 20.0)

        # Reduce confidence if high standard deviation (inconsistent data)
        consistency_penalty = (news_std + futures_std) / 4.0
        consistency_confidence = max(0.0, 1.0 - consistency_penalty)

        # Combined confidence
        confidence = (data_confidence * 0.6) + (consistency_confidence * 0.4)

        return confidence

    def _generate_insights(self, news_avg: float, futures_avg: float,
                          alignment: Dict, news_data: List[Dict],
                          futures_data: List[Dict]) -> str:
        """Generate human-readable insights from the comparison"""
        insights = []

        # Sentiment direction insights
        news_label = self._get_sentiment_label(news_avg)
        futures_label = self._get_sentiment_label(futures_avg)

        insights.append(f"News sentiment: {news_label} ({news_avg:+.2f})")
        insights.append(f"Futures sentiment: {futures_label} ({futures_avg:+.2f})")

        # Alignment insights
        if alignment['alignment_status'] in ['Strong Alignment', 'Moderate Alignment']:
            insights.append(f"✓ Market signals aligned - both indicate {news_label.lower()} sentiment")
        elif alignment['alignment_status'] == 'Strong Divergence':
            insights.append(f"⚠ Strong divergence detected - news shows {news_label.lower()} "
                          f"while futures show {futures_label.lower()}")
            insights.append("This could indicate market uncertainty or pending reversal")
        elif alignment['alignment_status'] == 'Mild Divergence':
            insights.append(f"△ Mild divergence - both lean {news_label.lower()} but with different confidence")

        # Data quality insights
        if len(news_data) < 3:
            insights.append("⚠ Limited news data - consider collecting more sources")
        if len(futures_data) < 2:
            insights.append("⚠ Limited futures data - confidence may be lower")

        # Trend insights
        if news_data and len(news_data) >= 2:
            recent_news = news_data[-1]['sentiment_score']
            older_news = news_data[0]['sentiment_score']
            if recent_news > older_news + 0.3:
                insights.append("↗ News sentiment trending more positive")
            elif recent_news < older_news - 0.3:
                insights.append("↘ News sentiment trending more negative")

        return " | ".join(insights)

    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score > 0.2:
            return "Bullish"
        elif score < -0.2:
            return "Bearish"
        else:
            return "Neutral"

    def compare_markets(self, all_news: Dict[str, List[Dict]],
                       all_futures: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """
        Compare sentiment across multiple markets/symbols

        Args:
            all_news: Dictionary of {symbol: [news_records]}
            all_futures: Dictionary of {symbol: [futures_records]}

        Returns:
            Dictionary of {symbol: comparison_analysis}
        """
        comparisons = {}

        # Get all unique symbols
        all_symbols = set(list(all_news.keys()) + list(all_futures.keys()))

        for symbol in all_symbols:
            news_data = all_news.get(symbol, [])
            futures_data = all_futures.get(symbol, [])

            comparison = self.analyze_sentiment_group(news_data, futures_data)
            comparison['symbol'] = symbol
            comparisons[symbol] = comparison

        return comparisons

    def identify_opportunities(self, comparisons: Dict[str, Dict],
                              min_divergence: float = 0.5) -> List[Dict]:
        """
        Identify potential trading opportunities based on divergence

        Args:
            comparisons: Dictionary of comparison analyses
            min_divergence: Minimum divergence magnitude to flag

        Returns:
            List of potential opportunities
        """
        opportunities = []

        for symbol, comparison in comparisons.items():
            if comparison['alignment_status'] == 'Strong Divergence':
                if comparison['divergence_magnitude'] >= min_divergence:
                    opportunity = {
                        'symbol': symbol,
                        'type': 'Divergence Opportunity',
                        'divergence_magnitude': comparison['divergence_magnitude'],
                        'news_sentiment': comparison['news_sentiment_avg'],
                        'futures_sentiment': comparison['futures_sentiment_avg'],
                        'confidence': comparison['confidence_score'],
                        'description': self._describe_opportunity(comparison)
                    }
                    opportunities.append(opportunity)

        # Sort by divergence magnitude
        opportunities.sort(key=lambda x: x['divergence_magnitude'], reverse=True)

        return opportunities

    def _describe_opportunity(self, comparison: Dict) -> str:
        """Describe the trading opportunity"""
        news_sent = comparison['news_sentiment_avg']
        futures_sent = comparison['futures_sentiment_avg']

        if news_sent > futures_sent:
            return (f"News is bullish while futures are bearish - "
                   f"potential for futures to catch up with news sentiment")
        else:
            return (f"Futures are bullish while news is bearish - "
                   f"potential contrarian play or news lag")
