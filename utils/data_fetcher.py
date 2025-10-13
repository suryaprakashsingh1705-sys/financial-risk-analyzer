"""
Data Fetcher Utilities

Handles fetching financial data from Yahoo Finance.
"""

from typing import Optional, Dict
import pandas as pd
import yfinance as yf
import sys


def fetch_company_info(ticker: str) -> Optional[Dict]:
    """Fetch company information from Yahoo Finance."""
    try:
        t = yf.Ticker(ticker)
        info = t.info if hasattr(t, "info") else {}

        # Validate ticker
        if not info or (
            not info.get("longName")
            and not info.get("shortName")
            and not info.get("symbol")
        ):
            return None

        return info

    except Exception:
        return None


def fetch_financial_components(ticker: str, debug: bool = False) -> Optional[Dict]:
    """
    Fetch all financial components needed for Z-Score calculation.

    Returns dict with: wc, re, ebit, ta, mve, tl, sales
    """
    try:
        t = yf.Ticker(ticker)
        issues = {}

        balances = t.balance_sheet
        financials = t.financials

        if debug and balances is not None and not balances.empty:
            print(
                f"\n[DEBUG {ticker}] Balance sheet fields:",
                list(balances.index),
                file=sys.stderr,
            )

        if balances is None or balances.empty:
            balances = t.quarterly_balance_sheet
            if balances is None or balances.empty:
                issues["balance_sheet"] = "No balance sheet data."

        if financials is None or financials.empty:
            financials = t.quarterly_financials
            if financials is None or financials.empty:
                issues["financials"] = "No financials data."

        # Market value of equity (MVE): shares * price
        try:
            info = t.info if hasattr(t, "info") else {}
            shares = info.get("sharesOutstanding")
            price = info.get("previousClose") or info.get("currentPrice")

            if shares is None or price is None:
                issues["mve"] = "Missing shares or price."
                mve = None
            else:
                mve = float(shares) * float(price)
        except Exception as e:
            issues["mve"] = f"Failed to compute MVE: {e}"
            mve = None

        def get_value(df: pd.DataFrame, *keys: str) -> Optional[float]:
            """Try multiple field names in order."""
            if df is None or df.empty:
                return None
            try:
                latest_col = df[df.columns[0]]
                for key in keys:
                    val = latest_col.get(key)
                    if pd.notna(val):
                        return float(val)
                return None
            except Exception:
                return None

        current_assets = get_value(balances, "Current Assets", "Total Current Assets")
        current_liabilities = get_value(
            balances, "Current Liabilities", "Total Current Liabilities"
        )
        total_assets = get_value(balances, "Total Assets", "TotalAssets")
        total_liabilities = get_value(
            balances,
            "Total Liabilities Net Minority Interest",
            "Total Liab",
            "Total Liabilities",
        )
        retained_earnings = get_value(balances, "Retained Earnings", "RetainedEarnings")
        ebit = get_value(
            financials, "EBIT", "Ebit", "Operating Income", "OperatingIncome"
        )
        sales = get_value(financials, "Total Revenue", "Revenue", "TotalRevenue")

        # Calculate working capital
        wc = None
        if current_assets is not None and current_liabilities is not None:
            wc = current_assets - current_liabilities
        else:
            if current_assets is None:
                issues["current_assets"] = "Missing"
            if current_liabilities is None:
                issues["current_liabilities"] = "Missing"

        # Check for missing values
        if retained_earnings is None:
            issues["retained_earnings"] = "Missing"
        if ebit is None:
            issues["ebit"] = "Missing"
        if total_assets is None:
            issues["total_assets"] = "Missing"
        if total_liabilities is None:
            issues["total_liabilities"] = "Missing"
        if sales is None:
            issues["sales"] = "Missing"
        if mve is None:
            issues["mve"] = "Missing"

        # If any required value is missing, return None
        if any(
            v is None
            for v in [
                wc,
                retained_earnings,
                ebit,
                total_assets,
                total_liabilities,
                sales,
                mve,
            ]
        ):
            return None

        return {
            "wc": wc,
            "re": retained_earnings,
            "ebit": ebit,
            "ta": total_assets,
            "mve": mve,
            "tl": total_liabilities,
            "sales": sales,
        }

    except Exception as e:
        if debug:
            print(f"Error fetching components for {ticker}: {e}", file=sys.stderr)
        return None
