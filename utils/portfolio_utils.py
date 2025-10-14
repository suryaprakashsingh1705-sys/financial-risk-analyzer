"""
Portfolio Utilities

Shared functions for portfolio analysis, including:
- Historical data fetching (prices and returns)
- Portfolio weight calculation
- Currency formatting
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
import sys
import logging

from config import YFINANCE_PROGRESS_BAR, YFINANCE_AUTO_ADJUST

logger = logging.getLogger(__name__)


def fetch_historical_data(
    tickers: List[str],
    period: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    return_prices: bool = False,
) -> pd.DataFrame:
    """
    Fetch historical data from Yahoo Finance.

    Can fetch either:
    - By period (e.g., "1y", "3mo") - specify period parameter
    - By date range - specify start_date and end_date parameters

    Args:
        tickers: List of ticker symbols
        period: Time period (e.g., "1y", "3mo", "5y")
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        return_prices: If False, returns daily returns; if True, returns prices

    Returns:
        DataFrame with prices (if return_prices=True) or returns (if return_prices=False)
    """
    try:
        # Determine download parameters
        if period:
            logger.debug(
                f"Fetching {period} of historical data for {len(tickers)} ticker(s)"
            )
            data = yf.download(
                tickers,
                period=period,
                progress=YFINANCE_PROGRESS_BAR,
                auto_adjust=YFINANCE_AUTO_ADJUST,
            )
        elif start_date and end_date:
            logger.debug(f"Fetching historical data from {start_date} to {end_date}")
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                progress=YFINANCE_PROGRESS_BAR,
                auto_adjust=YFINANCE_AUTO_ADJUST,
            )
        else:
            raise ValueError(
                "Must specify either 'period' or both 'start_date' and 'end_date'"
            )

        # Handle different data structures based on number of tickers
        if len(tickers) == 1:
            # Single ticker - yfinance may return MultiIndex or regular DataFrame
            if isinstance(data, pd.Series):
                # If it's a Series, convert to DataFrame
                prices = data.to_frame(name=tickers[0])
            elif isinstance(data.columns, pd.MultiIndex):
                # MultiIndex columns like ('Close', 'AAPL')
                if "Close" in [col[0] for col in data.columns]:
                    # Extract Close prices (already a DataFrame)
                    prices = data["Close"]
                    # Rename column to just the ticker name
                    if isinstance(prices, pd.DataFrame):
                        prices.columns = [tickers[0]]
                    else:
                        prices = prices.to_frame(name=tickers[0])
                else:
                    prices = data.iloc[:, 0].to_frame(name=tickers[0])
            elif "Close" in data.columns:
                # Regular DataFrame with Close column
                prices = data["Close"].to_frame(name=tickers[0])
            else:
                # Already a DataFrame with price data
                prices = data
                if prices.shape[1] == 1 and prices.columns[0] != tickers[0]:
                    prices.columns = [tickers[0]]
        else:
            # Multiple tickers - returns MultiIndex DataFrame
            if isinstance(data.columns, pd.MultiIndex):
                # Extract Close prices from MultiIndex
                prices = data["Close"]
            elif "Close" in data.columns:
                prices = data["Close"]
            elif "Adj Close" in data.columns:
                prices = data["Adj Close"]
            else:
                # Data might already be just the Close prices
                prices = data

        if return_prices:
            return prices
        else:
            # Calculate daily returns
            returns = prices.pct_change().dropna()

            # Verify we have valid data
            if returns.empty:
                raise ValueError("No valid return data available")

            logger.debug(f"Successfully fetched {len(returns)} days of return data")
            return returns

    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        print(f"❌ Error fetching data: {e}", file=sys.stderr)
        print(f"\nTroubleshooting:", file=sys.stderr)
        print(
            f"  - Check that all tickers are valid Yahoo Finance symbols",
            file=sys.stderr,
        )
        print(f"  - Try with fewer tickers to identify the problem", file=sys.stderr)
        if period:
            print(
                f"  - Some tickers may not have {period} of history available",
                file=sys.stderr,
            )
        else:
            print(
                f"  - Check that the date range {start_date} to {end_date} is valid",
                file=sys.stderr,
            )
        print(f"  - Verify your internet connection", file=sys.stderr)
        return None


def calculate_portfolio_weights(
    tickers: List[str],
    weights: Optional[List[float]] = None,
    values: Optional[List[float]] = None,
    portfolio_value: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    Calculate portfolio weights from various input formats.

    Args:
        tickers: List of ticker symbols
        weights: Optional list of weights (must sum to 1.0)
        values: Optional list of dollar values for each position
        portfolio_value: Total portfolio value (used with weights)

    Returns:
        Tuple of (weights_array, total_portfolio_value)

    Raises:
        ValueError: If inputs are invalid or inconsistent
    """
    if values:
        # Value-based input
        if len(values) != len(tickers):
            raise ValueError(
                f"Number of values ({len(values)}) must match number of tickers ({len(tickers)})"
            )

        total_value = sum(values)
        weights_array = np.array([v / total_value for v in values])

        logger.info(
            f"Using value-based allocation: Total portfolio value = ${total_value:,.2f}"
        )
        return weights_array, total_value

    elif weights:
        # Weight-based input
        if len(weights) != len(tickers):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match number of tickers ({len(tickers)})"
            )
        if not np.isclose(sum(weights), 1.0):
            raise ValueError(
                f"Weights must sum to 1.0 (currently sum to {sum(weights):.4f})"
            )
        if portfolio_value is None or portfolio_value <= 0:
            raise ValueError(
                "Portfolio value must be specified and positive when using weights"
            )

        weights_array = np.array(weights)

        logger.info(
            f"Using weight-based allocation: {', '.join(f'{t}={w:.1%}' for t, w in zip(tickers, weights))}"
        )
        return weights_array, portfolio_value

    else:
        # Equal weights (default)
        if portfolio_value is None or portfolio_value <= 0:
            raise ValueError("Portfolio value must be specified and positive")

        weights_array = np.array([1.0 / len(tickers)] * len(tickers))

        logger.info(f"Using equal weights for {len(tickers)} ticker(s)")
        return weights_array, portfolio_value


def format_currency(value: float) -> str:
    """
    Format currency values with proper sign placement.

    Examples:
        1234.56 -> "$1,234.56"
        -1234.56 -> "-$1,234.56"
    """
    if value < 0:
        return f"-${abs(value):,.2f}"
    else:
        return f"${value:,.2f}"


def format_currency_with_sign(value: float) -> str:
    """
    Format currency with explicit +/- sign.

    Examples:
        1234.56 -> "+$1,234.56"
        -1234.56 -> "-$1,234.56"
    """
    if value < 0:
        return f"-${abs(value):,.2f}"
    else:
        return f"+${value:,.2f}"


def validate_tickers(tickers: List[str]) -> List[str]:
    """
    Validate and normalize ticker symbols.

    Args:
        tickers: List of ticker symbols

    Returns:
        List of uppercase ticker symbols
    """
    if not tickers:
        raise ValueError("At least one ticker must be provided")

    return [t.upper() for t in tickers]


def display_and_save_results(
    df: pd.DataFrame,
    output_file: str,
    success_message: Optional[str] = None,
    separator: bool = True,
) -> None:
    """
    Display results DataFrame and save to CSV file.

    Args:
        df: DataFrame containing results to display and save
        output_file: Path to output CSV file
        success_message: Optional custom success message (defaults to "Results saved to {file}")
        separator: Whether to add separator lines before/after output (default True)

    Example:
        >>> df = pd.DataFrame({'Ticker': ['AAPL'], 'Score': [2.5]})
        >>> display_and_save_results(df, 'output.csv')
    """
    # Display results with optional separator
    if separator:
        print("\n" + "=" * 80)
    else:
        print()

    print(df.to_string(index=False))

    if separator:
        print("=" * 80)

    # Save to CSV
    df.to_csv(output_file, index=False)

    # Display success message
    if success_message:
        print(f"\n{success_message}")
    else:
        print(f"\n✅ Results saved to {output_file}")

    logger.info(f"Results saved to {output_file}")
