"""
Data normalization utilities for TradeAI
"""

from typing import Literal, Optional, Dict, Any

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from tradeAI.utils.logger import get_logger

logger = get_logger(__name__)


class Normalizer:
    """Normalize data for machine learning."""

    def __init__(
        self,
        method: Literal["minmax", "zscore", "robust", "log", "pct_change"] = "minmax",
        feature_range: tuple = (0, 1),
        per_column: bool = True,
    ):
        """
        Initialize Normalizer.

        Args:
            method: Normalization method
            feature_range: Range for minmax scaling
            per_column: Whether to normalize each column separately
        """
        self.method = method
        self.feature_range = feature_range
        self.per_column = per_column
        self.scalers_: Dict[str, Any] = {}
        self.fitted_ = False

    def fit(self, df: pd.DataFrame) -> "Normalizer":
        """
        Fit normalizer on data.

        Args:
            df: DataFrame to fit

        Returns:
            Self
        """
        if self.method == "minmax":
            if self.per_column:
                for col in df.columns:
                    scaler = MinMaxScaler(feature_range=self.feature_range)
                    scaler.fit(df[[col]])
                    self.scalers_[col] = scaler
            else:
                scaler = MinMaxScaler(feature_range=self.feature_range)
                scaler.fit(df)
                self.scalers_["all"] = scaler

        elif self.method == "zscore":
            if self.per_column:
                for col in df.columns:
                    scaler = StandardScaler()
                    scaler.fit(df[[col]])
                    self.scalers_[col] = scaler
            else:
                scaler = StandardScaler()
                scaler.fit(df)
                self.scalers_["all"] = scaler

        elif self.method == "robust":
            if self.per_column:
                for col in df.columns:
                    scaler = RobustScaler()
                    scaler.fit(df[[col]])
                    self.scalers_[col] = scaler
            else:
                scaler = RobustScaler()
                scaler.fit(df)
                self.scalers_["all"] = scaler

        self.fitted_ = True
        logger.debug(f"Fitted {self.method} normalizer on {len(df.columns)} columns")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted normalizer.

        Args:
            df: DataFrame to transform

        Returns:
            Normalized DataFrame
        """
        if not self.fitted_ and self.method in ["minmax", "zscore", "robust"]:
            raise ValueError("Normalizer must be fitted before transform")

        df = df.copy()

        if self.method == "minmax":
            df = self._transform_sklearn(df)

        elif self.method == "zscore":
            df = self._transform_sklearn(df)

        elif self.method == "robust":
            df = self._transform_sklearn(df)

        elif self.method == "log":
            df = self._transform_log(df)

        elif self.method == "pct_change":
            df = self._transform_pct_change(df)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform data.

        Args:
            df: DataFrame to fit and transform

        Returns:
            Normalized DataFrame
        """
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform normalized data back to original scale.

        Args:
            df: Normalized DataFrame

        Returns:
            Original scale DataFrame
        """
        if not self.fitted_:
            raise ValueError("Normalizer must be fitted before inverse_transform")

        if self.method not in ["minmax", "zscore", "robust"]:
            raise ValueError(f"Inverse transform not supported for method: {self.method}")

        df = df.copy()

        if self.per_column:
            for col in df.columns:
                if col in self.scalers_:
                    df[[col]] = self.scalers_[col].inverse_transform(df[[col]])
        else:
            df[:] = self.scalers_["all"].inverse_transform(df)

        return df

    def _transform_sklearn(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform using sklearn scalers."""
        if self.per_column:
            for col in df.columns:
                if col in self.scalers_:
                    df[[col]] = self.scalers_[col].transform(df[[col]])
        else:
            df[:] = self.scalers_["all"].transform(df)

        return df

    def _transform_log(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log transformation."""
        # Add small constant to avoid log(0)
        df = np.log1p(df)
        return df

    def _transform_pct_change(self, df: pd.DataFrame) -> pd.DataFrame:
        """Percentage change transformation."""
        df = df.pct_change()
        df = df.fillna(0)  # First row will be NaN
        return df


def normalize_ohlcv(
    df: pd.DataFrame,
    method: Literal["price_only", "all", "returns"] = "price_only",
) -> pd.DataFrame:
    """
    Normalize OHLCV data with appropriate methods for each column type.

    Args:
        df: OHLCV DataFrame
        method: Normalization strategy

    Returns:
        Normalized DataFrame
    """
    df = df.copy()

    if method == "price_only":
        # Normalize prices, keep volume as-is (will normalize separately)
        price_cols = ["open", "high", "low", "close"]
        if all(col in df.columns for col in price_cols):
            normalizer = Normalizer(method="minmax")
            df[price_cols] = normalizer.fit_transform(df[price_cols])

    elif method == "all":
        # Normalize all columns
        normalizer = Normalizer(method="minmax")
        df = normalizer.fit_transform(df)

    elif method == "returns":
        # Convert to returns
        price_cols = ["open", "high", "low", "close"]
        if all(col in df.columns for col in price_cols):
            df[price_cols] = df[price_cols].pct_change()
            df = df.fillna(0)

    logger.debug(f"Normalized OHLCV data using method: {method}")

    return df
