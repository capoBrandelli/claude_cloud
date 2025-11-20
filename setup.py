"""
TradeAI: Neural Network-Based Trading System
Setup configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "TRADEAI_ARCHITECTURE_PLAN.md").read_text()

setup(
    name="tradeai",
    version="0.1.0",
    author="TradeAI Team",
    author_email="your.email@example.com",
    description="Neural network-based trading system for financial market reversal prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tradeai",
    packages=find_packages(exclude=["tests", "docs", "notebooks", "scripts"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",

        # Machine Learning
        "torch>=2.0.0",
        "scikit-learn>=1.0.0",
        "optuna>=3.0.0",
        "imbalanced-learn>=0.9.0",

        # Data providers
        "yfinance>=0.2.0",
        "alpha-vantage>=2.3.0",
        "python-binance>=1.0.0",
        "polygon-api-client>=1.0.0",
        "twelvedata>=1.2.0",

        # Technical analysis
        "pandas-ta>=0.3.0",

        # Visualization
        "matplotlib>=3.5.0",
        "plotly>=5.0.0",
        "seaborn>=0.11.0",
        "mplfinance>=0.12.0",

        # Utilities
        "pydantic>=2.0.0",
        "loguru>=0.6.0",
        "tqdm>=4.60.0",
        "joblib>=1.1.0",
        "mlflow>=2.0.0",
        "python-dotenv>=0.19.0",
        "pyyaml>=6.0",
        "requests>=2.26.0",
        "aiohttp>=3.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "isort>=5.10.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.18.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.7.0",
        ],
        "talib": [
            "TA-Lib>=0.4.24",  # Optional, requires system installation
        ],
    },
    entry_points={
        "console_scripts": [
            "tradeai-collect=scripts.collect_data:main",
            "tradeai-train=scripts.train_model:main",
            "tradeai-evaluate=scripts.evaluate_model:main",
            "tradeai-backtest=scripts.run_backtest:main",
        ],
    },
    include_package_data=True,
    package_data={
        "tradeAI": ["config/*.yaml"],
    },
    zip_safe=False,
)
