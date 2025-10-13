"""
Helper Utilities

Contains company classification logic for Z-Score analysis.
"""

from dataclasses import dataclass
from typing import Dict
import pandas as pd
import yfinance as yf


@dataclass
class CompanyProfile:
    is_manufacturing: bool
    sector: str
    industry: str
    region: str
    classification_reason: str


def detect_market_region(info: Dict, ticker: str) -> str:
    """
    Detect if a company is from US, Europe, or other region.

    Priority order:
    1. Ticker suffix (most reliable for non-US stocks)
    2. Exchange
    3. Country
    4. yfinance 'region' field (least reliable)

    Returns: 'US', 'EU', 'CA', 'OTHER'
    """
    ticker_upper = ticker.upper()

    # Method 1: Check ticker suffix FIRST
    EU_SUFFIXES = {
        ".AS",
        ".PA",
        ".BR",
        ".LS",
        ".MI",
        ".MC",
        ".DE",
        ".F",
        ".SW",
        ".ST",
        ".CO",
        ".HE",
        ".OL",
        ".IC",
        ".L",
        ".IR",
        ".VI",
        ".PR",
        ".WA",
        ".AT",
    }
    CA_SUFFIXES = {".TO", ".V"}

    for suffix in EU_SUFFIXES:
        if ticker_upper.endswith(suffix):
            return "EU"
    for suffix in CA_SUFFIXES:
        if ticker_upper.endswith(suffix):
            return "CA"

    # Method 2: Check exchange
    exchange = info.get("exchange", "").upper()
    EU_EXCHANGES = {
        "AMS",
        "EPA",
        "EBR",
        "ELI",
        "MIL",
        "BIT",
        "MCE",
        "BME",
        "FRA",
        "GER",
        "XETRA",
        "ETR",
        "VTX",
        "SIX",
        "STO",
        "CPH",
        "HEL",
        "OSL",
        "ICE",
        "LON",
        "LSE",
        "ISE",
        "DUB",
        "VIE",
        "VSE",
        "PRA",
        "PSE",
        "WSE",
        "ATH",
        "ASE",
    }
    CA_EXCHANGES = {"TOR", "TSE", "CVE", "TSX", "TSXV"}
    US_EXCHANGES = {"NYQ", "NMS", "NGM", "NCM", "NYSE", "NASDAQ", "AMEX", "BATS"}

    if exchange in EU_EXCHANGES:
        return "EU"
    if exchange in CA_EXCHANGES:
        return "CA"
    if exchange in US_EXCHANGES:
        return "US"

    # Method 3: Check country
    country = info.get("country", "").upper()
    EU_COUNTRIES = {
        "AUSTRIA",
        "BELGIUM",
        "BULGARIA",
        "CROATIA",
        "CYPRUS",
        "CZECH REPUBLIC",
        "DENMARK",
        "ESTONIA",
        "FINLAND",
        "FRANCE",
        "GERMANY",
        "GREECE",
        "HUNGARY",
        "IRELAND",
        "ITALY",
        "LATVIA",
        "LITHUANIA",
        "LUXEMBOURG",
        "MALTA",
        "NETHERLANDS",
        "POLAND",
        "PORTUGAL",
        "ROMANIA",
        "SLOVAKIA",
        "SLOVENIA",
        "SPAIN",
        "SWEDEN",
        "UNITED KINGDOM",
        "UK",
        "NORWAY",
        "SWITZERLAND",
        "ICELAND",
        "GREAT BRITAIN",
    }

    if country in EU_COUNTRIES:
        return "EU"
    if country in {"CANADA"}:
        return "CA"
    if country in {"UNITED STATES", "USA", "US"}:
        return "US"

    # Method 4: Check yfinance 'region' field (last resort)
    region = info.get("region", "").upper()
    if region == "US":
        return "US"
    elif region:
        return "OTHER"

    # Default: if ticker has no suffix, assume US
    if "." not in ticker_upper:
        return "US"

    return "OTHER"


def get_company_profile(info: Dict, ticker: str) -> CompanyProfile:
    """
    Determine if a company is a manufacturer based on sector and industry analysis.

    Uses:
    - Sector classification
    - Industry keywords
    - Business description analysis
    - Financial statement ratios (PPE/Assets)
    """
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    industry_key = info.get("industryKey", "").lower()

    region = detect_market_region(info, ticker)

    sector_lower = sector.lower()
    industry_lower = industry.lower()

    classification_reason = ""
    is_manufacturing = False

    # Basic materials - almost always manufacturing
    if sector_lower == "basic materials":
        is_manufacturing = True
        classification_reason = f"[{region}] Sector '{sector}' is manufacturing-focused"

    # Industrials - distinguish manufacturing from services
    elif sector_lower == "industrials":
        if any(
            kw in industry_lower
            for kw in [
                "airline",
                "railroad",
                "trucking",
                "marine shipping",
                "air freight",
                "logistics",
                "courier",
                "delivery",
            ]
        ):
            is_manufacturing = False
            classification_reason = f"[{region}] Industrials sector but industry '{industry}' is transportation/logistics service"
        else:
            is_manufacturing = True
            classification_reason = (
                f"[{region}] Sector '{sector}' is manufacturing-focused"
            )

    # Energy - typically production
    elif sector_lower == "energy":
        is_manufacturing = True
        classification_reason = f"[{region}] Sector '{sector}' involves production"

    # Consumer goods - check for retail vs manufacturing
    elif sector_lower in {"consumer cyclical", "consumer defensive"}:
        if any(
            kw in industry_lower
            for kw in ["retail", "store", "restaurant", "service", "chain"]
        ):
            is_manufacturing = False
            classification_reason = f"[{region}] Sector '{sector}' but industry '{industry}' is retail/service"
        else:
            is_manufacturing = True
            classification_reason = f"[{region}] Sector '{sector}' with industry '{industry}' suggests manufacturing"

    # Healthcare - distinguish manufacturing from services
    elif sector_lower == "healthcare":
        if any(
            kw in industry_lower
            for kw in [
                "drug",
                "pharmaceutical",
                "device",
                "equipment",
                "biotech",
                "instruments",
            ]
        ):
            is_manufacturing = True
            classification_reason = f"[{region}] Healthcare sector but industry '{industry}' involves manufacturing"
        else:
            is_manufacturing = False
            classification_reason = (
                f"[{region}] Healthcare sector but industry '{industry}' is services"
            )

    # Technology - complex classification
    elif sector_lower == "technology":
        if any(
            kw in industry_lower
            for kw in ["equipment", "instruments", "manufacturing", "machinery"]
        ):
            is_manufacturing = True
            classification_reason = f"[{region}] Technology sector but industry '{industry}' involves manufacturing"

        elif any(kw in industry_lower for kw in ["semiconductor", "chip"]):
            is_manufacturing, classification_reason = _classify_semiconductor(
                info, ticker, region
            )

        else:
            is_manufacturing = False
            classification_reason = f"[{region}] Technology sector, industry '{industry}' is software/services"

    # Service sectors
    elif sector_lower in {
        "communication services",
        "financial services",
        "real estate",
        "utilities",
    }:
        is_manufacturing = False
        classification_reason = f"[{region}] Sector '{sector}' is service-based"

    # Unknown
    else:
        is_manufacturing = False
        classification_reason = (
            f"[{region}] Sector '{sector}' ambiguous - defaulting to non-manufacturing"
        )

    return CompanyProfile(
        is_manufacturing=is_manufacturing,
        sector=sector,
        industry=industry,
        region=region,
        classification_reason=classification_reason,
    )


def _classify_semiconductor(info: Dict, ticker: str, region: str) -> tuple:
    """
    Special classification logic for semiconductor companies.
    Distinguishes between fab owners (manufacturers) and fabless (designers).
    """
    industry_key = info.get("industryKey", "").lower()

    # Semiconductor equipment manufacturers (like ASML)
    if industry_key == "semiconductor-equipment-materials":
        return True, f"[{region}] Semiconductor equipment manufacturing"

    # Analyze business description and financials
    company_name = (info.get("longName", "") or info.get("shortName", "")).lower()
    business_summary = info.get("longBusinessSummary", "").lower()

    # Keywords indicating manufacturing (fab ownership)
    fab_keywords = [
        "manufactur",
        "foundry",
        "foundries",
        "fabrication",
        "fabricates",
        "wafer fabrication",
        "owns fab",
        "operates fab",
        "production facilities",
        "semiconductor manufacturing",
        "chip manufacturing",
        "wafer production",
        "producing semiconductors",
        "integrated device manufacturer",
        "fab operations",
    ]

    # Keywords indicating fabless (design only)
    fabless_keywords = [
        "fabless",
        "design-only",
        "designs chips",
        "chip design",
        "outsource manufacturing",
        "outsourced manufacturing",
        "outsources manufacturing",
        "contract manufacturers",
        "third-party foundries",
        "without manufacturing",
    ]

    # Score based on keywords
    fab_score = sum(1 for kw in fab_keywords if kw in business_summary)
    fabless_score = sum(1 for kw in fabless_keywords if kw in business_summary)

    # Check company name
    if any(kw in company_name for kw in ["manufacturing", "foundry", "fabrication"]):
        fab_score += 2

    # Check PPE ratio (Property, Plant & Equipment)
    try:
        t = yf.Ticker(ticker)
        balances = t.balance_sheet if hasattr(t, "balance_sheet") else None

        if balances is not None and not balances.empty:
            latest_col = balances[balances.columns[0]]

            ppe = None
            for key in [
                "Net PPE",
                "Property Plant Equipment",
                "Net Property Plant And Equipment",
                "Property Plant And Equipment Net",
                "Tangible Asset",
            ]:
                val = latest_col.get(key)
                if pd.notna(val):
                    ppe = float(val)
                    break

            ta = latest_col.get("Total Assets")
            if pd.notna(ta):
                ta = float(ta)

            if ppe is not None and ta is not None and ta > 0:
                ppe_ratio = ppe / ta

                # Manufacturers have high PPE (>30%), fabless have low PPE (<10%)
                if ppe_ratio > 0.30:
                    fab_score += 3
                elif ppe_ratio < 0.10:
                    fabless_score += 2
    except Exception:
        pass

    # Determine classification
    if fabless_score > 0:
        return (
            False,
            f"[{region}] Fabless semiconductor (business description indicates design-only)",
        )
    elif fab_score >= 2:
        return (
            True,
            f"[{region}] Semiconductor manufacturer (business description indicates fab ownership)",
        )
    elif fab_score == 1:
        return (
            True,
            f"[{region}] Likely semiconductor manufacturer (business description suggests manufacturing)",
        )
    else:
        return (
            False,
            f"[{region}] Presumed fabless semiconductor (no clear manufacturing indicators)",
        )
