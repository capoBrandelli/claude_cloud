"""
Database schema and management for Financial Sentiment Analyzer
"""
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import json


class Database:
    """Manages SQLite database operations for sentiment data"""

    def __init__(self, db_path: str = "financial_sentiment.db"):
        self.db_path = db_path
        self.init_database()

    def get_connection(self):
        """Create and return a database connection"""
        return sqlite3.connect(self.db_path)

    def init_database(self):
        """Initialize database schema"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # News sentiment table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT,
                source TEXT NOT NULL,
                publication_date TIMESTAMP NOT NULL,
                url TEXT,
                sentiment_score REAL NOT NULL,
                sentiment_label TEXT NOT NULL,
                relevance_score REAL NOT NULL,
                keywords TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Futures sentiment table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS futures_sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                data_date TIMESTAMP NOT NULL,
                open_interest REAL,
                long_positions REAL,
                short_positions REAL,
                net_positions REAL,
                sentiment_score REAL NOT NULL,
                sentiment_label TEXT NOT NULL,
                price REAL,
                volume REAL,
                data_source TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Sentiment comparison table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_comparison (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                analysis_date TIMESTAMP NOT NULL,
                news_sentiment_avg REAL NOT NULL,
                futures_sentiment_avg REAL NOT NULL,
                alignment_score REAL NOT NULL,
                alignment_status TEXT NOT NULL,
                divergence_magnitude REAL,
                confidence_score REAL,
                news_count INTEGER,
                futures_count INTEGER,
                insights TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Market symbols reference table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                is_active BOOLEAN DEFAULT 1,
                UNIQUE(market_type, symbol)
            )
        """)

        conn.commit()
        conn.close()

    def insert_news_sentiment(self, data: Dict) -> int:
        """Insert news sentiment record"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO news_sentiment
            (market_type, symbol, title, content, source, publication_date,
             url, sentiment_score, sentiment_label, relevance_score, keywords)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data['market_type'],
            data['symbol'],
            data['title'],
            data.get('content'),
            data['source'],
            data['publication_date'],
            data.get('url'),
            data['sentiment_score'],
            data['sentiment_label'],
            data['relevance_score'],
            json.dumps(data.get('keywords', []))
        ))

        news_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return news_id

    def insert_futures_sentiment(self, data: Dict) -> int:
        """Insert futures sentiment record"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO futures_sentiment
            (market_type, symbol, data_date, open_interest, long_positions,
             short_positions, net_positions, sentiment_score, sentiment_label,
             price, volume, data_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data['market_type'],
            data['symbol'],
            data['data_date'],
            data.get('open_interest'),
            data.get('long_positions'),
            data.get('short_positions'),
            data.get('net_positions'),
            data['sentiment_score'],
            data['sentiment_label'],
            data.get('price'),
            data.get('volume'),
            data['data_source']
        ))

        futures_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return futures_id

    def insert_sentiment_comparison(self, data: Dict) -> int:
        """Insert sentiment comparison record"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO sentiment_comparison
            (market_type, symbol, analysis_date, news_sentiment_avg,
             futures_sentiment_avg, alignment_score, alignment_status,
             divergence_magnitude, confidence_score, news_count, futures_count, insights)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data['market_type'],
            data['symbol'],
            data['analysis_date'],
            data['news_sentiment_avg'],
            data['futures_sentiment_avg'],
            data['alignment_score'],
            data['alignment_status'],
            data.get('divergence_magnitude'),
            data.get('confidence_score'),
            data.get('news_count', 0),
            data.get('futures_count', 0),
            data.get('insights')
        ))

        comparison_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return comparison_id

    def get_news_sentiment(self, market_type: Optional[str] = None,
                          symbol: Optional[str] = None,
                          days: int = 7) -> List[Dict]:
        """Retrieve news sentiment data"""
        conn = self.get_connection()
        cursor = conn.cursor()

        query = """
            SELECT * FROM news_sentiment
            WHERE publication_date >= datetime('now', '-' || ? || ' days')
        """
        params = [days]

        if market_type:
            query += " AND market_type = ?"
            params.append(market_type)

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        query += " ORDER BY publication_date DESC"

        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        return results

    def get_futures_sentiment(self, market_type: Optional[str] = None,
                             symbol: Optional[str] = None,
                             days: int = 7) -> List[Dict]:
        """Retrieve futures sentiment data"""
        conn = self.get_connection()
        cursor = conn.cursor()

        query = """
            SELECT * FROM futures_sentiment
            WHERE data_date >= datetime('now', '-' || ? || ' days')
        """
        params = [days]

        if market_type:
            query += " AND market_type = ?"
            params.append(market_type)

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        query += " ORDER BY data_date DESC"

        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        return results

    def get_sentiment_comparisons(self, market_type: Optional[str] = None,
                                  symbol: Optional[str] = None,
                                  days: int = 7) -> List[Dict]:
        """Retrieve sentiment comparison data"""
        conn = self.get_connection()
        cursor = conn.cursor()

        query = """
            SELECT * FROM sentiment_comparison
            WHERE analysis_date >= datetime('now', '-' || ? || ' days')
        """
        params = [days]

        if market_type:
            query += " AND market_type = ?"
            params.append(market_type)

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        query += " ORDER BY analysis_date DESC"

        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        return results

    def add_market_symbol(self, market_type: str, symbol: str,
                         name: str, description: str = None):
        """Add a market symbol to track"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR IGNORE INTO market_symbols
            (market_type, symbol, name, description)
            VALUES (?, ?, ?, ?)
        """, (market_type, symbol, name, description))

        conn.commit()
        conn.close()

    def get_market_symbols(self, market_type: Optional[str] = None) -> List[Dict]:
        """Get tracked market symbols"""
        conn = self.get_connection()
        cursor = conn.cursor()

        if market_type:
            cursor.execute("""
                SELECT * FROM market_symbols
                WHERE market_type = ? AND is_active = 1
            """, (market_type,))
        else:
            cursor.execute("""
                SELECT * FROM market_symbols
                WHERE is_active = 1
            """)

        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        return results
