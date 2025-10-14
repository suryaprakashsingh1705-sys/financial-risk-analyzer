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
from datetime import datetime
import sys
import logging

from config import (
    STRESS_DEFAULT_PORTFOLIO_VALUE,
    CRISIS_DOTCOM_START,
    CRISIS_DOTCOM_END,
    CRISIS_DOTCOM_MARKET_DROP,
    CRISIS_GFC_START,
    CRISIS_GFC_END,
    CRISIS_GFC_MARKET_DROP,
    CRISIS_COVID_START,
    CRISIS_COVID_END,
    CRISIS_COVID_MARKET_DROP,
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
    logger.debug(f"Fetching historical prices from {start_date} to {end_date}")
    prices = fetch_historical_data(
        tickers, start_date=start_date, end_date=end_date, return_prices=True
    )

    if prices is None:
        print(f"   Tickers: {', '.join(tickers)}", file=sys.stderr)

    return prices


def calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculate maximum drawdown during the period."""
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    return drawdown.min() * 100  # Convert to percentage


def calculate_portfolio_stress(
    prices: pd.DataFrame,
    weights: np.ndarray,
    portfolio_value: float,
    scenario_name: str,
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
        max_drawdown=max_dd,
    )


def run_single_scenario(
    tickers: List[str],
    weights: np.ndarray,
    portfolio_value: float,
    scenario: CrisisScenario,
    currency: str = "USD",
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

    # Convert prices to portfolio currency using historical exchange rates
    try:
        converted_prices, exchange_rates = convert_prices_to_currency(
            prices,
            tickers,
            currency,
            start_date=scenario.start_date,
            end_date=scenario.end_date,
        )
        # Only print conversion info once per scenario
        if len([t for t in tickers if t in converted_prices.columns]) > 0:
            print_currency_conversion_info(tickers, currency, exchange_rates)
    except ValueError as e:
        logger.error(str(e))
        print(f"❌ Currency conversion error: {e}", file=sys.stderr)
        return []

    # Use converted prices for analysis
    prices = converted_prices

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
        print(f"Tickers: {', '.join(tickers)}")
        if hasattr(args, "values") and args.values:
            print(
                f"Position Values: {', '.join(f'{t}={format_currency_value(v, args.currency)}' for t, v in zip(tickers, args.values))}\n"
            )
        elif hasattr(args, "weights") and args.weights:
            print(f"Weights: {', '.join(f'{w:.1%}' for w in weights)}\n")
        else:
            print(f"Weights: {', '.join(f'{w:.1%}' for w in weights)}\n")

    except ValueError as e:
        logger.error(str(e))
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine which scenarios to run
    if args.scenario == "all":
        scenarios_to_run = list(SCENARIOS.values())
    else:
        scenarios_to_run = [SCENARIOS[args.scenario]]

    all_results = []

    # Run stress tests
    for scenario in scenarios_to_run:
        scenario_results = run_single_scenario(
            tickers, weights, args.portfolio_value, scenario, args.currency
        )
        all_results.extend(scenario_results)

        if scenario_results:
            # Separate individual and portfolio results
            individual_results = [
                r for r in scenario_results if r.ticker != "PORTFOLIO"
            ]
            portfolio_result = [r for r in scenario_results if r.ticker == "PORTFOLIO"]

            # Display individual asset results
            df_scenario = pd.DataFrame(
                [
                    {
                        "Ticker": r.ticker,
                        "Start Price": format_currency_value(
                            r.start_price, args.currency
                        ),
                        "End Price": format_currency_value(r.end_price, args.currency),
                        "Return": f"{r.return_pct:+.2f}%",
                        "Gain/Loss": format_currency_value_with_sign(
                            r.loss_amount, args.currency
                        ),
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
                print(
                    f"   Portfolio Return (with correlations): {port.return_pct:+.2f}%"
                )
                print(
                    f"   Portfolio Gain/Loss: {format_currency_value_with_sign(port.loss_amount, args.currency)}"
                )
                print(f"   Portfolio Max Drawdown: {port.max_drawdown:.2f}%")
                print(
                    f"   Final Portfolio Value: {format_currency_value(port.position_value_end, args.currency)}"
                )
            else:
                # Fallback for single ticker
                total_gain = sum(r.loss_amount for r in individual_results)
                portfolio_return = (total_gain / args.portfolio_value) * 100
                print(
                    f"   Total Gain/Loss: {format_currency_value_with_sign(total_gain, args.currency)}"
                )
                print(f"   Portfolio Return: {portfolio_return:+.2f}%")
                print(
                    f"   Final Portfolio Value: {format_currency_value(args.portfolio_value + total_gain, args.currency)}"
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
                "Start Price": format_currency_value(r.start_price, args.currency),
                "End Price": format_currency_value(r.end_price, args.currency),
                "Return %": f"{r.return_pct:+.2f}%",
                "Position Start": format_currency_value(
                    r.position_value_start, args.currency
                ),
                "Position End": format_currency_value(
                    r.position_value_end, args.currency
                ),
                "Gain/Loss": format_currency_value_with_sign(
                    r.loss_amount, args.currency
                ),
                "Max Drawdown": f"{r.max_drawdown:.2f}%",
            }
            for r in all_results
        ]
    )

    # Display and save results
    display_and_save_results(df, args.out, separator=False)

    # Summary insights
    print(f"💡 Key Insights:")
    worst_scenario = min(
        all_results, key=lambda x: x.loss_amount
    )  # Most negative = worst loss
    best_scenario = max(
        all_results, key=lambda x: x.loss_amount
    )  # Most positive = best gain

    print(f"   Worst Loss: {worst_scenario.ticker} in {worst_scenario.scenario}")
    print(
        f"   Loss: {format_currency_value_with_sign(worst_scenario.loss_amount, args.currency)} ({worst_scenario.return_pct:+.2f}%)"
    )
    print(f"   Max Drawdown: {worst_scenario.max_drawdown:.2f}%")

    if best_scenario.loss_amount > 0:
        print(
            f"\n   Best Performance: {best_scenario.ticker} in {best_scenario.scenario}"
        )
        print(
            f"   Gain: {format_currency_value_with_sign(best_scenario.loss_amount, args.currency)} ({best_scenario.return_pct:+.2f}%)"
        )
