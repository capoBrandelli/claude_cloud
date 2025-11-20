"""
Sentiment Analysis Module for News and Text
"""
from typing import Dict, List, Tuple
import re
from datetime import datetime


class NewsSentimentAnalyzer:
    """Analyzes sentiment of financial news articles"""

    def __init__(self):
        # Financial sentiment lexicons
        self.positive_words = {
            'surge', 'rally', 'gain', 'profit', 'growth', 'rise', 'boom',
            'bullish', 'positive', 'upgrade', 'outperform', 'buy', 'strong',
            'optimistic', 'recovery', 'expansion', 'breakthrough', 'success',
            'upward', 'higher', 'increase', 'soar', 'jump', 'climb', 'advance',
            'improving', 'robust', 'solid', 'beat', 'exceed', 'top', 'momentum'
        }

        self.negative_words = {
            'crash', 'plunge', 'loss', 'decline', 'fall', 'drop', 'bearish',
            'negative', 'downgrade', 'underperform', 'sell', 'weak', 'pessimistic',
            'recession', 'contraction', 'failure', 'loss', 'downward', 'lower',
            'decrease', 'tumble', 'slide', 'slump', 'concern', 'worry', 'risk',
            'trouble', 'crisis', 'volatile', 'unstable', 'miss', 'disappoint'
        }

        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'highly': 1.5, 'significantly': 1.7,
            'substantially': 1.8, 'considerably': 1.6, 'remarkably': 1.7,
            'dramatically': 1.9, 'sharply': 1.8, 'strongly': 1.6
        }

        self.negations = {'not', 'no', 'never', 'neither', 'nor', 'cannot', 'won\'t'}

    def analyze_text(self, text: str, title: str = "") -> Dict:
        """
        Analyze sentiment of text content

        Returns:
            Dictionary with sentiment_score (-1 to 1), sentiment_label, and confidence
        """
        combined_text = f"{title} {text}".lower()
        words = re.findall(r'\b\w+\b', combined_text)

        positive_score = 0
        negative_score = 0
        word_count = len(words)

        for i, word in enumerate(words):
            # Check for intensifiers
            multiplier = 1.0
            if i > 0 and words[i-1] in self.intensifiers:
                multiplier = self.intensifiers[words[i-1]]

            # Check for negations
            is_negated = i > 0 and words[i-1] in self.negations

            if word in self.positive_words:
                if is_negated:
                    negative_score += multiplier
                else:
                    positive_score += multiplier

            elif word in self.negative_words:
                if is_negated:
                    positive_score += multiplier
                else:
                    negative_score += multiplier

        # Calculate normalized sentiment score
        total_sentiment_words = positive_score + negative_score
        if total_sentiment_words == 0:
            sentiment_score = 0.0
            confidence = 0.0
        else:
            sentiment_score = (positive_score - negative_score) / total_sentiment_words
            # Confidence based on number of sentiment words found
            confidence = min(1.0, total_sentiment_words / max(word_count / 20, 1))

        # Determine sentiment label
        if sentiment_score > 0.2:
            sentiment_label = "Bullish"
        elif sentiment_score < -0.2:
            sentiment_label = "Bearish"
        else:
            sentiment_label = "Neutral"

        return {
            'sentiment_score': round(sentiment_score, 3),
            'sentiment_label': sentiment_label,
            'confidence': round(confidence, 3),
            'positive_score': positive_score,
            'negative_score': negative_score
        }

    def calculate_relevance_score(self, text: str, symbol: str,
                                  market_type: str, keywords: List[str] = None) -> float:
        """
        Calculate how relevant the news is to the specific symbol/market

        Returns:
            Relevance score between 0 and 1
        """
        text_lower = text.lower()
        symbol_lower = symbol.lower()

        relevance = 0.0

        # Check symbol presence
        symbol_count = text_lower.count(symbol_lower)
        relevance += min(0.4, symbol_count * 0.1)

        # Check market type keywords
        market_keywords = {
            'forex': ['currency', 'exchange rate', 'forex', 'fx', 'dollar', 'euro'],
            'commodity': ['commodity', 'oil', 'gold', 'silver', 'wheat', 'corn', 'metal'],
            'stocks': ['stock', 'equity', 'shares', 'company', 'earnings', 'revenue'],
            'etf': ['etf', 'fund', 'index', 'basket', 'portfolio']
        }

        if market_type.lower() in market_keywords:
            for keyword in market_keywords[market_type.lower()]:
                if keyword in text_lower:
                    relevance += 0.1

        # Check custom keywords
        if keywords:
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    relevance += 0.05

        return min(1.0, relevance)

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract key financial terms from text"""
        # Common financial terms to look for
        financial_terms = {
            'earnings', 'revenue', 'profit', 'loss', 'market', 'trading',
            'price', 'volume', 'growth', 'decline', 'forecast', 'outlook',
            'quarter', 'annual', 'report', 'guidance', 'upgrade', 'downgrade',
            'buy', 'sell', 'hold', 'target', 'estimate', 'beat', 'miss'
        }

        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        # Find financial terms present in text
        found_keywords = []
        for term in financial_terms:
            if term in words:
                found_keywords.append(term)

        return found_keywords[:top_n]


class FuturesSentimentAnalyzer:
    """Analyzes sentiment based on futures market data"""

    def analyze_positions(self, long_positions: float, short_positions: float,
                         open_interest: float = None) -> Dict:
        """
        Analyze sentiment from futures positioning data

        Returns:
            Dictionary with sentiment_score, sentiment_label, and confidence
        """
        if long_positions == 0 and short_positions == 0:
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'Neutral',
                'confidence': 0.0,
                'net_positions': 0.0
            }

        # Calculate net positions
        net_positions = long_positions - short_positions
        total_positions = long_positions + short_positions

        # Sentiment score based on net long/short ratio
        if total_positions > 0:
            sentiment_score = net_positions / total_positions
        else:
            sentiment_score = 0.0

        # Confidence based on open interest (if available)
        confidence = 0.7  # Base confidence
        if open_interest and open_interest > 0:
            position_ratio = total_positions / open_interest
            confidence = min(1.0, 0.5 + (position_ratio * 0.5))

        # Determine sentiment label
        if sentiment_score > 0.15:
            sentiment_label = "Bullish"
        elif sentiment_score < -0.15:
            sentiment_label = "Bearish"
        else:
            sentiment_label = "Neutral"

        return {
            'sentiment_score': round(sentiment_score, 3),
            'sentiment_label': sentiment_label,
            'confidence': round(confidence, 3),
            'net_positions': net_positions,
            'long_ratio': round(long_positions / total_positions, 3) if total_positions > 0 else 0,
            'short_ratio': round(short_positions / total_positions, 3) if total_positions > 0 else 0
        }

    def analyze_price_momentum(self, prices: List[float],
                               volumes: List[float] = None) -> Dict:
        """
        Analyze sentiment from price momentum

        Args:
            prices: List of recent prices (oldest to newest)
            volumes: Optional list of trading volumes

        Returns:
            Sentiment analysis based on price trends
        """
        if len(prices) < 2:
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'Neutral',
                'confidence': 0.0
            }

        # Calculate price change percentage
        price_change = (prices[-1] - prices[0]) / prices[0]

        # Calculate momentum (rate of change acceleration)
        if len(prices) >= 3:
            recent_change = (prices[-1] - prices[-2]) / prices[-2]
            earlier_change = (prices[-2] - prices[-3]) / prices[-3]
            momentum = recent_change - earlier_change
        else:
            momentum = 0

        # Sentiment score based on price change and momentum
        sentiment_score = price_change * 10  # Scale to -1 to 1 range
        sentiment_score += momentum * 5  # Add momentum influence
        sentiment_score = max(-1, min(1, sentiment_score))  # Clamp to [-1, 1]

        # Confidence based on volume (if available)
        confidence = 0.6
        if volumes and len(volumes) >= 2:
            avg_volume = sum(volumes) / len(volumes)
            recent_volume = volumes[-1]
            if avg_volume > 0:
                volume_ratio = recent_volume / avg_volume
                confidence = min(1.0, 0.5 + (volume_ratio * 0.25))

        # Determine sentiment label
        if sentiment_score > 0.2:
            sentiment_label = "Bullish"
        elif sentiment_score < -0.2:
            sentiment_label = "Bearish"
        else:
            sentiment_label = "Neutral"

        return {
            'sentiment_score': round(sentiment_score, 3),
            'sentiment_label': sentiment_label,
            'confidence': round(confidence, 3),
            'price_change_pct': round(price_change * 100, 2)
        }
