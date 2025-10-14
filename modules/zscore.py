"""
Altman Z-Score Module

Calculates bankruptcy risk using the Altman Z-Score model.
Automatically classifies companies as manufacturing vs non-manufacturing
and applies the appropriate formula.
"""

from dataclasses import dataclass
from typing import Optional, List
import pandas as pd
import sys
import logging

from utils.data_fetcher import fetch_company_info, fetch_financial_components
from utils.helpers import detect_market_region, get_company_profile
from utils.portfolio_utils import display_and_save_results
from config import (
    ZSCORE_ORIGINAL_WC_COEF,
    ZSCORE_ORIGINAL_RE_COEF,
    ZSCORE_ORIGINAL_EBIT_COEF,
    ZSCORE_ORIGINAL_MVE_COEF,
    ZSCORE_ORIGINAL_SALES_COEF,
    ZSCORE_MODIFIED_WC_COEF,
    ZSCORE_MODIFIED_RE_COEF,
    ZSCORE_MODIFIED_EBIT_COEF,
    ZSCORE_MODIFIED_BVE_COEF,
    ZSCORE_ORIGINAL_DISTRESS_THRESHOLD,
    ZSCORE_ORIGINAL_GREY_THRESHOLD,
    ZSCORE_MODIFIED_DISTRESS_THRESHOLD,
    ZSCORE_MODIFIED_GREY_THRESHOLD,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ZScoreResult:
    ticker: str
    company: str
    region: str
    classification: str
    model: str
    z_score: float
    zone: str
    industry: str


def calculate_z_score_original(
    wc: float, re: float, ebit: float, ta: float, mve: float, tl: float, sales: float
) -> float:
    """
    Calculate original Altman Z-Score for manufacturing companies.

    Z = 1.2*(WC/TA) + 1.4*(RE/TA) + 3.3*(EBIT/TA) + 0.6*(MVE/TL) + 1.0*(Sales/TA)
    """
    if ta <= 0:
        raise ValueError("Total assets must be positive for Z-Score calculation")
    if tl <= 0:
        raise ValueError("Total liabilities must be positive for Z-Score calculation")

    return (
        ZSCORE_ORIGINAL_WC_COEF * (wc / ta)
        + ZSCORE_ORIGINAL_RE_COEF * (re / ta)
        + ZSCORE_ORIGINAL_EBIT_COEF * (ebit / ta)
        + ZSCORE_ORIGINAL_MVE_COEF * (mve / tl)
        + ZSCORE_ORIGINAL_SALES_COEF * (sales / ta)
    )


def calculate_z_score_modified(
    wc: float, re: float, ebit: float, ta: float, mve: float, tl: float
) -> float:
    """
    Calculate modified Altman Z''-Score for non-manufacturing companies.

    Z'' = 6.56*(WC/TA) + 3.26*(RE/TA) + 6.72*(EBIT/TA) + 1.05*(BVE/TL)
    """
    if ta <= 0:
        raise ValueError("Total assets must be positive for Z-Score calculation")
    if tl <= 0:
        raise ValueError("Total liabilities must be positive for Z-Score calculation")

    bve = ta - tl  # Book Value of Equity
    return (
        ZSCORE_MODIFIED_WC_COEF * (wc / ta)
        + ZSCORE_MODIFIED_RE_COEF * (re / ta)
        + ZSCORE_MODIFIED_EBIT_COEF * (ebit / ta)
        + ZSCORE_MODIFIED_BVE_COEF * (bve / tl)
    )


def get_zone(z_score: float, model: str) -> str:
    """Determine risk zone based on Z-score and model type."""
    if model == "Original":
        if z_score < ZSCORE_ORIGINAL_DISTRESS_THRESHOLD:
            return "Distress"
        elif z_score < ZSCORE_ORIGINAL_GREY_THRESHOLD:
            return "Grey Zone"
        return "Safe"
    else:  # Modified
        if z_score < ZSCORE_MODIFIED_DISTRESS_THRESHOLD:
            return "Distress"
        elif z_score < ZSCORE_MODIFIED_GREY_THRESHOLD:
            return "Grey Zone"
        return "Safe"


def analyze_ticker(
    ticker: str, verbose: bool = False, debug: bool = False
) -> Optional[ZScoreResult]:
    """Analyze a single ticker and return Z-Score result."""
    try:
        logger.debug(f"Analyzing ticker: {ticker}")

        # Fetch company info
        info = fetch_company_info(ticker)
        if not info:
            logger.warning(f"Skipping {ticker}: Invalid ticker or unable to fetch data")
            if verbose:
                print(f"  Skipping {ticker}: Invalid ticker", file=sys.stderr)
            return None

        # Get company profile and classification
        profile = get_company_profile(info, ticker)

        if verbose:
            print(f"\n{ticker}: {profile.classification_reason}", file=sys.stderr)

        # Get company name
        company_name = info.get("longName") or info.get("shortName") or ticker

        # Fetch financial components
        components = fetch_financial_components(ticker, debug=debug)

        if not components:
            logger.warning(f"Skipping {ticker}: Missing required financial data")
            if verbose:
                print(f"  Skipping {ticker}: Missing financial data", file=sys.stderr)
            return None

        # Calculate Z-Score based on company type
        if profile.is_manufacturing:
            model = "Original"
            logger.debug(f"{ticker}: Using Original Z-Score model (Manufacturing)")
            z_score = calculate_z_score_original(
                components["wc"],
                components["re"],
                components["ebit"],
                components["ta"],
                components["mve"],
                components["tl"],
                components["sales"],
            )
        else:
            model = "Modified"
            logger.debug(f"{ticker}: Using Modified Z-Score model (Non-Manufacturing)")
            z_score = calculate_z_score_modified(
                components["wc"],
                components["re"],
                components["ebit"],
                components["ta"],
                components["mve"],
                components["tl"],
            )

        logger.info(
            f"{ticker}: Z-Score = {z_score:.2f}, Zone = {get_zone(z_score, model)}"
        )

        return ZScoreResult(
            ticker=ticker,
            company=company_name,
            region=profile.region,
            classification="Mfg" if profile.is_manufacturing else "Non-Mfg",
            model=model,
            z_score=z_score,
            zone=get_zone(z_score, model),
            industry=profile.industry,
        )

    except ValueError as e:
        logger.error(f"Skipping {ticker}: Invalid financial data - {str(e)}")
        if verbose:
            print(f"  Skipping {ticker}: {str(e)}", file=sys.stderr)
        return None
    except Exception as e:
        logger.error(f"Skipping {ticker}: Unexpected error - {str(e)}")
        if verbose:
            print(f"  Skipping {ticker}: {str(e)}", file=sys.stderr)
        return None


def run_zscore(args):
    """Main function to run Z-Score analysis."""
    logger.info("Starting Altman Z-Score Analysis")
    print("🔍 Running Altman Z-Score Analysis...\n")

    results = []
    for ticker in args.tickers:
        result = analyze_ticker(ticker.upper(), verbose=args.verbose, debug=args.debug)
        if result:
            results.append(result)

    if not results:
        logger.error("No valid results generated. All tickers were skipped.")
        print("❌ No valid results. All tickers were skipped.", file=sys.stderr)
        sys.exit(1)

    logger.info(f"Successfully analyzed {len(results)} ticker(s)")

    # Create DataFrame
    df = pd.DataFrame(
        [
            {
                "Ticker": r.ticker,
                "Company": r.company,
                "Region": r.region,
                "Classification": r.classification,
                "Model": r.model,
                "ZScore": f"{r.z_score:.2f}",
                "Zone": r.zone,
                "Industry": r.industry,
            }
            for r in results
        ]
    )

    # Display and save results
    display_and_save_results(df, args.out, separator=False)
