"""
Stress Test Module

Simulates portfolio performance under historical crisis scenarios:
- Dot-Com Bubble Burst (2000-2002)
- Global Financial Crisis (2008-2009)
- COVID-19 Crash (Q1 2020)
"""

from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import sys
import logging

from config import (
    STRESS_DEFAULT_PORTFOLIO_VALUE,
    CRISIS_DOTCOM_START, CRISIS_DOTCOM_END, CRISIS_DOTCOM_MARKET_DROP,
    CRISIS_GFC_START, CRISIS_GFC_END, CRISIS_GFC_MARKET_DROP,
    CRISIS_COVID_START, CRISIS_COVID_END, CRISIS_COVID_MARKET_DROP,
    YFINANCE_PROGRESS_BAR, YFINANCE_AUTO_ADJUST,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CrisisScenario:
    name: str
    description: str
    start_date: str
    end_date: str
    market_drop: float  # Percentage drop for reference


# Historical crisis scenarios
SCENARIOS = {
    "dotcom": CrisisScenario(
        name="Dot-Com Bubble Burst",
        description="Tech bubble burst & 9/11 aftermath",
        start_date=CRISIS_DOTCOM_START,
        end_date=CRISIS_DOTCOM_END,
        market_drop=CRISIS_DOTCOM_MARKET_DROP,
    ),
    "gfc": CrisisScenario(
        name="Global Financial Crisis",
        description="Subprime mortgage crisis & Lehman Brothers collapse",
        start_date=CRISIS_GFC_START,
        end_date=CRISIS_GFC_END,
        market_drop=CRISIS_GFC_MARKET_DROP,
    ),
    "covid": CrisisScenario(
        name="COVID-19 Crash",
        description="Global pandemic market crash",
        start_date=CRISIS_COVID_START,
        end_date=CRISIS_COVID_END,
        market_drop=CRISIS_COVID_MARKET_DROP,
    ),
}


@dataclass
class StressTestResult:
    ticker: str
    scenario: str
    start_price: float
    end_price: float
    return_pct: float
    position_value_start: float
    position_value_end: float
    loss_amount: float
    max_drawdown: float


def fetch_historical_prices(
    tickers: List[str], start_date: str, end_date: str
) -> pd.DataFrame:
    """Fetch historical prices for the crisis period."""
    try:
        logger.debug(f"Fetching historical prices from {start_date} to {end_date}")
        data = yf.download(
            tickers, start=start_date, end=end_date, progress=YFINANCE_PROGRESS_BAR, auto_adjust=YFINANCE_AUTO_ADJUST
        )

        # Handle different data structures
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

        return prices

    except Exception as e:
        logger.error(f"Error fetching historical data for period {start_date} to {end_date}: {e}")
        print(f"❌ Error fetching historical data for period {start_date} to {end_date}: {e}", file=sys.stderr)
        print(f"   Tickers: {', '.join(tickers)}", file=sys.stderr)
        return None


def format_currency(value: float) -> str:
    """Format currency values with proper sign placement: -$1,234.56 not $-1,234.56"""
    if value < 0:
        return f"-${abs(value):,.2f}"
    else:
        return f"${value:,.2f}"


def format_currency_with_sign(value: float) -> str:
    """Format currency with explicit +/- sign: +$1,234.56 or -$1,234.56"""
    if value < 0:
        return f"-${abs(value):,.2f}"
    else:
        return f"+${value:,.2f}"


def calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculate maximum drawdown during the period."""
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    return drawdown.min() * 100  # Convert to percentage


def calculate_portfolio_stress(
    prices: pd.DataFrame,
    weights: np.ndarray,
    portfolio_value: float,
    scenario_name: str
) -> StressTestResult:
    """Calculate portfolio-level stress test considering correlations."""
    # Calculate returns for each asset
    returns = prices.pct_change().dropna()

    # Calculate portfolio returns (weighted sum)
    portfolio_returns = (returns * weights).sum(axis=1)

    # Calculate portfolio prices from returns
    portfolio_prices = portfolio_value * (1 + portfolio_returns).cumprod()
    portfolio_prices = pd.concat([pd.Series([portfolio_value]), portfolio_prices])

    start_value = portfolio_prices.iloc[0]
    end_value = portfolio_prices.iloc[-1]
    return_pct = ((end_value - start_value) / start_value) * 100
    loss_amount = end_value - start_value

    max_dd = calculate_max_drawdown(portfolio_prices)

    return StressTestResult(
        ticker="PORTFOLIO",
        scenario=scenario_name,
        start_price=start_value,
        end_price=end_value,
        return_pct=return_pct,
        position_value_start=start_value,
        position_value_end=end_value,
        loss_amount=loss_amount,
        max_drawdown=max_dd
    )


def run_single_scenario(
    tickers: List[str],
    weights: np.ndarray,
    portfolio_value: float,
    scenario: CrisisScenario,
) -> List[StressTestResult]:
    """Run stress test for a single crisis scenario."""
    print(f"\n{'='*80}")
    print(f"📉 {scenario.name} ({scenario.start_date} to {scenario.end_date})")
    print(f"   {scenario.description}")
    print(f"   Historical Market Drop: {scenario.market_drop:.1f}%")
    print(f"{'='*80}\n")

    # Fetch historical data
    prices = fetch_historical_prices(tickers, scenario.start_date, scenario.end_date)

    if prices is None or prices.empty:
        print(f"⚠️  Warning: No data available for this period\n")
        return []

    results = []

    for i, ticker in enumerate(tickers):
        if ticker not in prices.columns:
            print(f"⚠️  {ticker}: No data available for this period")
            continue

        ticker_prices = prices[ticker].dropna()

        if len(ticker_prices) < 2:
            print(f"⚠️  {ticker}: Insufficient data")
            continue

        start_price = ticker_prices.iloc[0]
        end_price = ticker_prices.iloc[-1]
        return_pct = ((end_price - start_price) / start_price) * 100

        position_value_start = portfolio_value * weights[i]
        position_value_end = position_value_start * (1 + return_pct / 100)
        loss_amount = (
            position_value_end - position_value_start
        )  # Positive = gain, Negative = loss

        max_dd = calculate_max_drawdown(ticker_prices)

        results.append(
            StressTestResult(
                ticker=ticker,
                scenario=scenario.name,
                start_price=start_price,
                end_price=end_price,
                return_pct=return_pct,
                position_value_start=position_value_start,
                position_value_end=position_value_end,
                loss_amount=loss_amount,
                max_drawdown=max_dd,
            )
        )

    # Calculate portfolio-level stress test (with correlations)
    if len(tickers) > 1 and results:
        # Only calculate if we have multiple tickers with valid data
        valid_tickers = [r.ticker for r in results]
        valid_weights = np.array([weights[tickers.index(t)] for t in valid_tickers])

        # Renormalize weights if some tickers were missing
        if len(valid_weights) > 0:
            valid_weights = valid_weights / valid_weights.sum()

            # Filter prices to only include valid tickers
            valid_prices = prices[valid_tickers].dropna()

            if not valid_prices.empty:
                portfolio_result = calculate_portfolio_stress(
                    valid_prices, valid_weights, portfolio_value, scenario.name
                )
                results.append(portfolio_result)

    return results


def run_stress_test(args):
    """Main function to run stress test analysis."""
    logger.info("Starting Historical Stress Test Analysis")
    print("🔬 Running Historical Stress Test Analysis...\n")

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
        print(f"Tickers: {', '.join(tickers)}")
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
        print(f"Portfolio Value: ${args.portfolio_value:,.2f}")
        print(f"Tickers: {', '.join(tickers)}")
        print(f"Weights: {', '.join(f'{w:.1%}' for w in weights)}\n")
    else:
        # Equal weights (default)
        weights = np.array([1.0 / len(tickers)] * len(tickers))
        logger.info(f"Using equal weights for {len(tickers)} ticker(s)")
        print(f"Portfolio Value: ${args.portfolio_value:,.2f}")
        print(f"Tickers: {', '.join(tickers)}")
        print(f"Weights: {', '.join(f'{w:.1%}' for w in weights)}\n")

    # Determine which scenarios to run
    if args.scenario == "all":
        scenarios_to_run = list(SCENARIOS.values())
    else:
        scenarios_to_run = [SCENARIOS[args.scenario]]

    all_results = []

    # Run stress tests
    for scenario in scenarios_to_run:
        scenario_results = run_single_scenario(
            tickers, weights, args.portfolio_value, scenario
        )
        all_results.extend(scenario_results)

        if scenario_results:
            # Separate individual and portfolio results
            individual_results = [r for r in scenario_results if r.ticker != "PORTFOLIO"]
            portfolio_result = [r for r in scenario_results if r.ticker == "PORTFOLIO"]

            # Display individual asset results
            df_scenario = pd.DataFrame(
                [
                    {
                        "Ticker": r.ticker,
                        "Start Price": f"${r.start_price:.2f}",
                        "End Price": f"${r.end_price:.2f}",
                        "Return": f"{r.return_pct:+.2f}%",
                        "Gain/Loss": format_currency_with_sign(r.loss_amount),
                        "Max Drawdown": f"{r.max_drawdown:.2f}%",
                    }
                    for r in individual_results
                ]
            )

            print(df_scenario.to_string(index=False))

            # Display portfolio impact
            print(f"\n📊 Portfolio Impact:")
            if portfolio_result:
                # Use correlation-based portfolio calculation
                port = portfolio_result[0]
                print(f"   Portfolio Return (with correlations): {port.return_pct:+.2f}%")
                print(f"   Portfolio Gain/Loss: {format_currency_with_sign(port.loss_amount)}")
                print(f"   Portfolio Max Drawdown: {port.max_drawdown:.2f}%")
                print(f"   Final Portfolio Value: {format_currency(port.position_value_end)}")
            else:
                # Fallback for single ticker
                total_gain = sum(r.loss_amount for r in individual_results)
                portfolio_return = (total_gain / args.portfolio_value) * 100
                print(f"   Total Gain/Loss: {format_currency_with_sign(total_gain)}")
                print(f"   Portfolio Return: {portfolio_return:+.2f}%")
                print(
                    f"   Final Portfolio Value: {format_currency(args.portfolio_value + total_gain)}"
                )

    if not all_results:
        print(
            "❌ No results generated. Check if tickers existed during the crisis periods.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create comprehensive DataFrame
    df = pd.DataFrame(
        [
            {
                "Ticker": r.ticker,
                "Scenario": r.scenario,
                "Start Price": format_currency(r.start_price),
                "End Price": format_currency(r.end_price),
                "Return %": f"{r.return_pct:+.2f}%",
                "Position Start": format_currency(r.position_value_start),
                "Position End": format_currency(r.position_value_end),
                "Gain/Loss": format_currency_with_sign(r.loss_amount),
                "Max Drawdown": f"{r.max_drawdown:.2f}%",
            }
            for r in all_results
        ]
    )

    # Save to CSV
    df.to_csv(args.out, index=False)
    print(f"\n✅ Results saved to {args.out}")

    # Summary insights
    print(f"\n💡 Key Insights:")
    worst_scenario = min(
        all_results, key=lambda x: x.loss_amount
    )  # Most negative = worst loss
    best_scenario = max(
        all_results, key=lambda x: x.loss_amount
    )  # Most positive = best gain

    print(f"   Worst Loss: {worst_scenario.ticker} in {worst_scenario.scenario}")
    print(
        f"   Loss: {format_currency_with_sign(worst_scenario.loss_amount)} ({worst_scenario.return_pct:+.2f}%)"
    )
    print(f"   Max Drawdown: {worst_scenario.max_drawdown:.2f}%")

    if best_scenario.loss_amount > 0:
        print(
            f"\n   Best Performance: {best_scenario.ticker} in {best_scenario.scenario}"
        )
        print(
            f"   Gain: {format_currency_with_sign(best_scenario.loss_amount)} ({best_scenario.return_pct:+.2f}%)"
        )
