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
from scipy import stats
import sys
import logging

from config import (
    VAR_DEFAULT_CONFIDENCE,
    VAR_DEFAULT_DAYS,
    VAR_DEFAULT_PORTFOLIO_VALUE,
    MONTE_CARLO_DEFAULT_SIMULATIONS,
    VAR_HISTORICAL_DATA_PERIOD,
)
from utils.portfolio_utils import (
    fetch_historical_data,
    calculate_portfolio_weights,
    validate_tickers,
    display_and_save_results,
)
from utils.currency_converter import (
    convert_prices_to_currency,
    format_currency_value,
    format_currency_value_with_sign,
    get_currency_symbol,
    print_currency_conversion_info,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
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


def fetch_historical_returns(
    tickers: List[str], period: str = VAR_HISTORICAL_DATA_PERIOD
) -> pd.DataFrame:
    """Fetch historical price data and calculate returns."""
    logger.info(f"Fetching {period} of historical data for {len(tickers)} ticker(s)...")
    returns = fetch_historical_data(tickers, period=period, return_prices=False)

    if returns is None:
        sys.exit(1)

    return returns


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
        scaled_returns = (
            returns.rolling(window=days)
            .apply(lambda x: np.prod(1 + x) - 1, raw=False)
            .dropna()
        )
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
    print(f"Portfolio Currency: {args.currency}")

    tickers = validate_tickers(args.tickers)
    currency_symbol = get_currency_symbol(args.currency)

    # Determine weights and values using shared utility
    try:
        weights, portfolio_value = calculate_portfolio_weights(
            tickers,
            weights=getattr(args, "weights", None),
            values=getattr(args, "values", None),
            portfolio_value=getattr(args, "portfolio_value", None),
        )
        args.portfolio_value = portfolio_value

        # Display portfolio configuration
        print(
            f"Portfolio Value: {format_currency_value(portfolio_value, args.currency)}"
        )
        if hasattr(args, "values") and args.values:
            print(
                f"Position Values: {', '.join(f'{t}={format_currency_value(v, args.currency)}' for t, v in zip(tickers, args.values))}\n"
            )
        elif hasattr(args, "weights") and args.weights:
            print(
                f"Weights: {', '.join(f'{t}={w:.1%}' for t, w in zip(tickers, weights))}\n"
            )
        else:
            print(f"Weights: Equal ({1.0/len(tickers):.1%} each)\n")

    except ValueError as e:
        logger.error(str(e))
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Fetch historical data and convert currencies
    print("Fetching historical data...")
    prices = fetch_historical_data(
        tickers, period=VAR_HISTORICAL_DATA_PERIOD, return_prices=True
    )

    if prices is None:
        sys.exit(1)

    # Convert prices to portfolio currency using historical exchange rates
    try:
        converted_prices, exchange_rates = convert_prices_to_currency(
            prices, tickers, args.currency, period=VAR_HISTORICAL_DATA_PERIOD
        )
        print_currency_conversion_info(tickers, args.currency, exchange_rates)
    except ValueError as e:
        logger.error(str(e))
        print(f"❌ Currency conversion error: {e}", file=sys.stderr)
        sys.exit(1)

    # Calculate returns from converted prices
    returns = converted_prices.pct_change(fill_method=None).dropna()

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

    # Create DataFrame with currency-aware formatting
    df = pd.DataFrame(
        [
            {
                "Ticker": r.ticker,
                "Method": r.method,
                "Position Value": format_currency_value(
                    r.position_value, args.currency
                ),
                f"VaR ({r.confidence}%)": format_currency_value(
                    r.var_amount, args.currency
                ),
                "VaR %": f"{r.var_percentage:.2f}%",
                "Days": r.days,
            }
            for r in results
        ]
    )

    # Display and save results
    display_and_save_results(df, args.out)

    # Interpretation
    print(f"\n💡 Interpretation:")
    if len(tickers) == 1:
        print(f"   With {args.confidence}% confidence, you will not lose more than")
        print(
            f"   {format_currency_value(results[0].var_amount, args.currency)} ({results[0].var_percentage:.2f}%) over {args.days} day(s)."
        )
    else:
        portfolio_var = results[-1]
        print(
            f"   Portfolio VaR: With {args.confidence}% confidence, you will not lose more than"
        )
        print(
            f"   {format_currency_value(portfolio_var.var_amount, args.currency)} ({portfolio_var.var_percentage:.2f}%) over {args.days} day(s)."
        )
