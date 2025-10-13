"""
Value at Risk (VaR) Module

Calculates portfolio risk using three methods:
- Parametric (Variance-Covariance)
- Historical Simulation
- Monte Carlo Simulation
"""

from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import sys
import logging

from config import (
    VAR_DEFAULT_CONFIDENCE, VAR_DEFAULT_DAYS, VAR_DEFAULT_PORTFOLIO_VALUE,
    MONTE_CARLO_DEFAULT_SIMULATIONS, VAR_HISTORICAL_DATA_PERIOD,
    YFINANCE_PROGRESS_BAR, YFINANCE_AUTO_ADJUST,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class VaRResult:
    ticker: str
    method: str
    confidence: float
    days: int
    var_amount: float
    var_percentage: float
    position_value: float


def fetch_historical_returns(tickers: List[str], period: str = VAR_HISTORICAL_DATA_PERIOD) -> pd.DataFrame:
    """Fetch historical price data and calculate returns."""
    try:
        logger.info(f"Fetching {period} of historical data for {len(tickers)} ticker(s)...")
        data = yf.download(tickers, period=period, progress=YFINANCE_PROGRESS_BAR, auto_adjust=YFINANCE_AUTO_ADJUST)

        # Handle different data structures based on number of tickers
        if len(tickers) == 1:
            # Single ticker - yfinance may return MultiIndex or regular DataFrame
            if isinstance(data, pd.Series):
                # If it's a Series, convert to DataFrame
                prices = data.to_frame(name=tickers[0])
            elif isinstance(data.columns, pd.MultiIndex):
                # MultiIndex columns like ('Close', 'AAPL')
                if 'Close' in [col[0] for col in data.columns]:
                    # Extract Close prices (already a DataFrame)
                    prices = data['Close']
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
                prices = data['Close']
            elif "Close" in data.columns:
                prices = data["Close"]
            elif "Adj Close" in data.columns:
                prices = data["Adj Close"]
            else:
                # Data might already be just the Close prices
                prices = data

        # Calculate daily returns
        returns = prices.pct_change().dropna()

        # Verify we have valid data
        if returns.empty:
            raise ValueError("No valid return data available")

        logger.info(f"Successfully fetched {len(returns)} days of return data")
        return returns

    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        print(f"❌ Error fetching data: {e}", file=sys.stderr)
        print(f"\nTroubleshooting:", file=sys.stderr)
        print(f"  - Check that all tickers are valid Yahoo Finance symbols", file=sys.stderr)
        print(f"  - Try with fewer tickers to identify the problem", file=sys.stderr)
        print(f"  - Some tickers may not have {period} of history available", file=sys.stderr)
        print(f"  - Verify your internet connection", file=sys.stderr)
        sys.exit(1)


def calculate_parametric_var(
    returns: pd.Series, confidence: float, days: int, position_value: float
) -> Dict:
    """
    Calculate VaR using parametric (variance-covariance) method.
    Assumes returns are normally distributed.
    """
    if position_value <= 0:
        raise ValueError("Position value must be positive")
    if days <= 0:
        raise ValueError("Days must be positive")

    mean = returns.mean()
    std = returns.std()

    # Z-score for confidence level (negative value for left tail)
    z_score = stats.norm.ppf(1 - confidence / 100)

    # VaR for specified time horizon
    # For losses, we want the left tail: mean + (negative z_score) * std
    # Taking absolute value to express VaR as positive loss amount
    var_1day = position_value * abs(mean + z_score * std)
    var_nday = var_1day * np.sqrt(days)

    var_percentage = abs(var_nday / position_value * 100) if position_value > 0 else 0.0

    return {
        "var_amount": abs(var_nday),
        "var_percentage": var_percentage,
    }


def calculate_historical_var(
    returns: pd.Series, confidence: float, days: int, position_value: float
) -> Dict:
    """
    Calculate VaR using historical simulation.
    Uses actual historical returns distribution.
    """
    if position_value <= 0:
        raise ValueError("Position value must be positive")
    if days <= 0:
        raise ValueError("Days must be positive")

    # Scale returns to time horizon
    if days > 1:
        # Compound returns properly for multi-day periods (geometric returns)
        # (1+r1)*(1+r2)*...*(1+rn) - 1
        scaled_returns = returns.rolling(window=days).apply(
            lambda x: np.prod(1 + x) - 1, raw=False
        ).dropna()
    else:
        scaled_returns = returns

    # Calculate percentile
    var_percentile = np.percentile(scaled_returns, 100 - confidence)
    var_amount = abs(position_value * var_percentile)

    return {"var_amount": var_amount, "var_percentage": abs(var_percentile * 100)}


def calculate_monte_carlo_var(
    returns: pd.Series,
    confidence: float,
    days: int,
    position_value: float,
    simulations: int = 10000,
) -> Dict:
    """
    Calculate VaR using Monte Carlo simulation.
    Simulates future returns based on historical statistics.
    """
    if position_value <= 0:
        raise ValueError("Position value must be positive")
    if days <= 0:
        raise ValueError("Days must be positive")
    if simulations <= 0:
        raise ValueError("Number of simulations must be positive")

    mean = returns.mean()
    std = returns.std()

    # Simulate returns
    simulated_returns = np.random.normal(mean, std, (simulations, days))

    # Calculate cumulative returns for each simulation
    cumulative_returns = (1 + simulated_returns).prod(axis=1) - 1

    # Calculate VaR at specified confidence level
    var_percentile = np.percentile(cumulative_returns, 100 - confidence)
    var_amount = abs(position_value * var_percentile)

    return {"var_amount": var_amount, "var_percentage": abs(var_percentile * 100)}


def calculate_portfolio_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float,
    days: int,
    portfolio_value: float,
    method: str,
) -> Dict:
    """Calculate VaR for a multi-asset portfolio."""
    # Calculate portfolio returns
    portfolio_returns = (returns * weights).sum(axis=1)

    if method == "parametric":
        result = calculate_parametric_var(
            portfolio_returns, confidence, days, portfolio_value
        )
    elif method == "historical":
        result = calculate_historical_var(
            portfolio_returns, confidence, days, portfolio_value
        )
    elif method == "monte_carlo":
        result = calculate_monte_carlo_var(
            portfolio_returns, confidence, days, portfolio_value
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return result


def run_var(args):
    """Main function to run VaR analysis."""
    logger.info("Starting Value at Risk (VaR) Analysis")
    print(f"📊 Running Value at Risk (VaR) Analysis...\n")
    print(f"Method: {args.method.title()}")
    print(f"Confidence Level: {args.confidence}%")
    print(f"Time Horizon: {args.days} day(s)")

    tickers = [t.upper() for t in args.tickers]

    # Determine weights and values
    if hasattr(args, 'values') and args.values:
        # Value-based input
        if len(args.values) != len(tickers):
            logger.error(f"Number of values ({len(args.values)}) must match number of tickers ({len(tickers)})")
            print(
                f"❌ Error: Number of values ({len(args.values)}) must match number of tickers ({len(tickers)})",
                file=sys.stderr,
            )
            sys.exit(1)

        total_value = sum(args.values)
        weights = np.array([v / total_value for v in args.values])
        args.portfolio_value = total_value

        logger.info(f"Using value-based allocation: Total portfolio value = ${total_value:,.2f}")
        print(f"Portfolio Value: ${total_value:,.2f}")
        print(f"Position Values: {', '.join(f'{t}=${v:,.2f}' for t, v in zip(tickers, args.values))}\n")

    elif hasattr(args, 'weights') and args.weights:
        # Weight-based input
        if len(args.weights) != len(tickers):
            logger.error(f"Number of weights ({len(args.weights)}) must match number of tickers ({len(tickers)})")
            print(
                f"❌ Error: Number of weights ({len(args.weights)}) must match number of tickers ({len(tickers)})",
                file=sys.stderr,
            )
            sys.exit(1)
        if not np.isclose(sum(args.weights), 1.0):
            logger.error(f"Weights must sum to 1.0 (currently sum to {sum(args.weights):.4f})")
            print(
                f"❌ Error: Weights must sum to 1.0 (currently sum to {sum(args.weights):.4f})",
                file=sys.stderr,
            )
            sys.exit(1)
        weights = np.array(args.weights)

        logger.info(f"Using weight-based allocation: {', '.join(f'{t}={w:.1%}' for t, w in zip(tickers, weights))}")
        print(f"Portfolio Value: ${args.portfolio_value:,.2f}\n")
    else:
        # Equal weights (default)
        weights = np.array([1.0 / len(tickers)] * len(tickers))
        logger.info(f"Using equal weights for {len(tickers)} ticker(s)")
        print(f"Portfolio Value: ${args.portfolio_value:,.2f}\n")

    # Fetch historical returns
    print("Fetching historical data...")
    returns = fetch_historical_returns(tickers)

    results = []

    if len(tickers) == 1:
        # Single asset VaR
        ticker = tickers[0]
        position_value = args.portfolio_value

        if args.method == "parametric":
            var_result = calculate_parametric_var(
                returns[ticker], args.confidence, args.days, position_value
            )
        elif args.method == "historical":
            var_result = calculate_historical_var(
                returns[ticker], args.confidence, args.days, position_value
            )
        elif args.method == "monte_carlo":
            var_result = calculate_monte_carlo_var(
                returns[ticker], args.confidence, args.days, position_value
            )

        results.append(
            VaRResult(
                ticker=ticker,
                method=args.method.title(),
                confidence=args.confidence,
                days=args.days,
                var_amount=var_result["var_amount"],
                var_percentage=var_result["var_percentage"],
                position_value=position_value,
            )
        )
    else:
        # Portfolio VaR
        portfolio_result = calculate_portfolio_var(
            returns,
            weights,
            args.confidence,
            args.days,
            args.portfolio_value,
            args.method,
        )

        # Also calculate individual VaRs
        for i, ticker in enumerate(tickers):
            position_value = args.portfolio_value * weights[i]

            if args.method == "parametric":
                var_result = calculate_parametric_var(
                    returns[ticker], args.confidence, args.days, position_value
                )
            elif args.method == "historical":
                var_result = calculate_historical_var(
                    returns[ticker], args.confidence, args.days, position_value
                )
            elif args.method == "monte_carlo":
                var_result = calculate_monte_carlo_var(
                    returns[ticker], args.confidence, args.days, position_value
                )

            results.append(
                VaRResult(
                    ticker=ticker,
                    method=args.method.title(),
                    confidence=args.confidence,
                    days=args.days,
                    var_amount=var_result["var_amount"],
                    var_percentage=var_result["var_percentage"],
                    position_value=position_value,
                )
            )

        # Add portfolio result
        results.append(
            VaRResult(
                ticker="PORTFOLIO",
                method=args.method.title(),
                confidence=args.confidence,
                days=args.days,
                var_amount=portfolio_result["var_amount"],
                var_percentage=portfolio_result["var_percentage"],
                position_value=args.portfolio_value,
            )
        )

    # Create DataFrame
    df = pd.DataFrame(
        [
            {
                "Ticker": r.ticker,
                "Method": r.method,
                "Position Value": f"${r.position_value:,.2f}",
                f"VaR ({r.confidence}%)": f"${r.var_amount:,.2f}",
                "VaR %": f"{r.var_percentage:.2f}%",
                "Days": r.days,
            }
            for r in results
        ]
    )

    # Display results
    print("\n" + "=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    # Interpretation
    print(f"\n💡 Interpretation:")
    if len(tickers) == 1:
        print(f"   With {args.confidence}% confidence, you will not lose more than")
        print(
            f"   ${results[0].var_amount:,.2f} ({results[0].var_percentage:.2f}%) over {args.days} day(s)."
        )
    else:
        portfolio_var = results[-1]
        print(
            f"   Portfolio VaR: With {args.confidence}% confidence, you will not lose more than"
        )
        print(
            f"   ${portfolio_var.var_amount:,.2f} ({portfolio_var.var_percentage:.2f}%) over {args.days} day(s)."
        )

    # Save to CSV
    df.to_csv(args.out, index=False)
    print(f"\n✅ Results saved to {args.out}")
