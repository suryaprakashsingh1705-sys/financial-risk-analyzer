"""
Currency Conversion Utilities

Handles currency detection and conversion for international portfolios.
Fetches real-time exchange rates from Yahoo Finance.

Example: If portfolio currency is EUR and you have 5 EUR stocks + 1 USD stock,
         the USD stock prices will be converted to EUR using current exchange rate.
"""

from typing import Dict, List, Tuple
import yfinance as yf
import pandas as pd
import logging
import sys
from config import SUPPORTED_CURRENCIES

logger = logging.getLogger(__name__)

# Mapping of common stock exchange suffixes to their currencies
EXCHANGE_CURRENCY_MAP = {
    # European exchanges (EUR)
    ".AS": "EUR",  # Amsterdam (Euronext Amsterdam)
    ".BR": "EUR",  # Brussels (Euronext Brussels)
    ".PA": "EUR",  # Paris (Euronext Paris)
    ".DE": "EUR",  # XETRA (Germany)
    ".F": "EUR",  # Frankfurt
    ".MI": "EUR",  # Milan
    ".MC": "EUR",  # Madrid
    ".LS": "EUR",  # Lisbon
    ".AT": "EUR",  # Athens
    ".IR": "EUR",  # Irish Stock Exchange
    ".HE": "EUR",  # Helsinki (Nordic)
    ".VI": "EUR",  # Vienna
    # UK (GBP)
    ".L": "GBP",  # London Stock Exchange
    # Switzerland (CHF)
    ".SW": "CHF",  # Swiss Exchange
    # Canada (CAD)
    ".TO": "CAD",  # Toronto
    ".V": "CAD",  # TSX Venture
    # Australia (AUD)
    ".AX": "AUD",  # Australian Securities Exchange
    # Japan (JPY)
    ".T": "JPY",  # Tokyo Stock Exchange
    # Default (no suffix) = USD for most US exchanges
}

# FX cache to avoid redundant network calls
# Key: (from_currency, to_currency, period_or_dates_tuple)
_FX_CACHE = {}

# Maximum days to forward-fill stale FX rates (business days)
MAX_FX_FORWARD_FILL_DAYS = 3


def detect_ticker_currency(ticker: str) -> str:
    """
    Detect the currency of a stock ticker based on its exchange suffix.
    Falls back to yfinance ticker info if suffix not recognized.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "ASML.AS")

    Returns:
        Currency code (e.g., "USD", "EUR")
    """
    ticker_upper = ticker.upper()

    # Check for exchange suffix first
    for suffix, currency in EXCHANGE_CURRENCY_MAP.items():
        if ticker_upper.endswith(suffix):
            return currency

    # Fallback: try to get currency from yfinance ticker info
    # This handles ADR/DR cases and suffixless tickers
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info

        if info and "currency" in info:
            detected_currency = info["currency"].upper()
            # Only use if it's a supported currency
            if detected_currency in SUPPORTED_CURRENCIES:
                logger.debug(f"Detected currency {detected_currency} for {ticker} from yfinance info")
                return detected_currency
    except Exception as e:
        logger.debug(f"Could not fetch currency info for {ticker}: {e}")

    # Default to USD for tickers without recognized suffix or info
    logger.debug(f"Defaulting to USD for {ticker} (no suffix match or info)")
    return "USD"


def get_exchange_rate(from_currency: str, to_currency: str) -> Tuple[float, str]:
    """
    Get current (spot) exchange rate from Yahoo Finance.
    Handles weekends/holidays by falling back to 5-day period.
    Tries reverse pair if direct pair fails.

    Args:
        from_currency: Source currency code (e.g., "USD")
        to_currency: Target currency code (e.g., "EUR")

    Returns:
        Tuple of (exchange_rate, as_of_date_string)

    Raises:
        ValueError: If exchange rate cannot be fetched
    """
    if from_currency == to_currency:
        return 1.0, pd.Timestamp.now().strftime("%Y-%m-%d")

    # Yahoo Finance uses format like "USDEUR=X" for forex pairs
    forex_pair = f"{from_currency}{to_currency}=X"

    try:
        logger.debug(f"Fetching spot rate for {forex_pair}")

        # Try 1d first (normal case)
        data = yf.download(forex_pair, period="1d", progress=False)

        # If empty (weekend/holiday), try 5d and take last valid
        if data.empty:
            logger.debug(f"{forex_pair}: 1d returned empty, trying 5d (weekend/holiday)")
            data = yf.download(forex_pair, period="5d", progress=False)

        if data.empty:
            # Try reverse pair
            reverse_pair = f"{to_currency}{from_currency}=X"
            logger.debug(f"Trying reverse pair: {reverse_pair}")

            data = yf.download(reverse_pair, period="1d", progress=False)
            if data.empty:
                data = yf.download(reverse_pair, period="5d", progress=False)

            if data.empty:
                raise ValueError(
                    f"No exchange rate data available for {from_currency}/{to_currency}"
                )

            # Extract rate and invert
            if isinstance(data.columns, pd.MultiIndex):
                close_data = data["Close"].squeeze()
            elif "Close" in data.columns:
                close_data = data["Close"]
            else:
                close_data = data.iloc[:, 0]

            # Handle both Series and scalar cases
            if isinstance(close_data, pd.Series):
                rate = 1.0 / float(close_data.iloc[-1])
                as_of_date = close_data.index[-1].strftime("%Y-%m-%d")
            else:
                # Scalar case (single value)
                rate = 1.0 / float(close_data)
                as_of_date = data.index[-1].strftime("%Y-%m-%d")

            logger.info(f"Spot rate {from_currency}/{to_currency}: {rate:.4f} (inverted from {reverse_pair}, as of {as_of_date})")
            return rate, as_of_date

        # Extract rate from direct pair
        if isinstance(data.columns, pd.MultiIndex):
            close_data = data["Close"].squeeze()
        elif "Close" in data.columns:
            close_data = data["Close"]
        else:
            close_data = data.iloc[:, 0]

        # Handle both Series and scalar cases
        if isinstance(close_data, pd.Series):
            rate = float(close_data.iloc[-1])
            as_of_date = close_data.index[-1].strftime("%Y-%m-%d")
        else:
            # Scalar case (single value)
            rate = float(close_data)
            as_of_date = data.index[-1].strftime("%Y-%m-%d")

        logger.info(f"Spot rate {from_currency}/{to_currency}: {rate:.4f} (as of {as_of_date})")
        return rate, as_of_date

    except Exception as e:
        logger.error(f"Error fetching spot rate for {forex_pair}: {e}")
        raise ValueError(
            f"Could not fetch exchange rate for {from_currency}/{to_currency}. "
            f"Please check your internet connection."
        )


def detect_portfolio_currencies(tickers: List[str]) -> Dict[str, str]:
    """
    Detect the currency for each ticker in a portfolio.

    Args:
        tickers: List of ticker symbols

    Returns:
        Dictionary mapping ticker to currency code
    """
    currency_map = {}
    for ticker in tickers:
        currency_map[ticker] = detect_ticker_currency(ticker)

    return currency_map


def get_currency_symbol(currency: str) -> str:
    """
    Get the display symbol for a currency.

    Args:
        currency: Currency code (e.g., "USD", "EUR")

    Returns:
        Currency symbol (e.g., "$", "€")
    """
    symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "CHF": "CHF",
        "CAD": "C$",
        "AUD": "A$",
        "JPY": "¥",
        "HKD": "HK$",
    }
    return symbols.get(currency, currency)


def get_historical_exchange_rates(
    from_currency: str,
    to_currency: str,
    start_date: str = None,
    end_date: str = None,
    period: str = None,
) -> pd.Series:
    """
    Get historical exchange rates from Yahoo Finance.

    Args:
        from_currency: Source currency code (e.g., "USD")
        to_currency: Target currency code (e.g., "EUR")
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        period: Time period (e.g., "1y", "2y") - alternative to start/end dates

    Returns:
        pandas Series with historical exchange rates indexed by date

    Raises:
        ValueError: If exchange rates cannot be fetched or if inputs are invalid
    """
    # Validate currency codes
    if not from_currency or not to_currency:
        raise ValueError("Currency codes cannot be empty")

    if not isinstance(from_currency, str) or not isinstance(to_currency, str):
        raise ValueError("Currency codes must be strings")

    from_currency = from_currency.upper().strip()
    to_currency = to_currency.upper().strip()

    if len(from_currency) != 3 or len(to_currency) != 3:
        raise ValueError(
            f"Invalid currency codes: {from_currency}, {to_currency}. Must be 3-letter codes (e.g., USD, EUR)"
        )

    # Validate currencies are in supported list
    if from_currency not in SUPPORTED_CURRENCIES:
        raise ValueError(
            f"Unsupported currency: {from_currency}. "
            f"Supported currencies: {', '.join(SUPPORTED_CURRENCIES)}"
        )

    if to_currency not in SUPPORTED_CURRENCIES:
        raise ValueError(
            f"Unsupported currency: {to_currency}. "
            f"Supported currencies: {', '.join(SUPPORTED_CURRENCIES)}"
        )

    if from_currency == to_currency:
        # Return a series of 1.0 for the requested period
        try:
            if period:
                dummy_data = yf.download("SPY", period=period, progress=False)
            else:
                dummy_data = yf.download(
                    "SPY", start=start_date, end=end_date, progress=False
                )

            if dummy_data.empty:
                raise ValueError(
                    "Could not fetch reference dates for exchange rate series"
                )

            return pd.Series(1.0, index=dummy_data.index)
        except Exception as e:
            logger.error(f"Error creating dummy exchange rate series: {e}")
            raise ValueError(f"Could not create exchange rate series: {e}")

    # Check cache first
    cache_key = (from_currency, to_currency, period or (start_date, end_date))
    if cache_key in _FX_CACHE:
        logger.debug(f"Using cached FX data for {from_currency}/{to_currency}")
        return _FX_CACHE[cache_key].copy()

    # Yahoo Finance uses format like "USDEUR=X" for forex pairs
    forex_pair = f"{from_currency}{to_currency}=X"

    try:
        logger.debug(f"Fetching historical exchange rates for {forex_pair}")

        if period:
            data = yf.download(
                forex_pair, period=period, progress=False, auto_adjust=True
            )
        else:
            data = yf.download(
                forex_pair,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
            )

        if data.empty:
            # Try reverse pair (e.g., if USDEUR=X fails, try EURUSD=X and invert)
            reverse_pair = f"{to_currency}{from_currency}=X"
            logger.debug(f"Trying reverse pair: {reverse_pair}")

            if period:
                data = yf.download(
                    reverse_pair, period=period, progress=False, auto_adjust=True
                )
            else:
                data = yf.download(
                    reverse_pair,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True,
                )

            if data.empty:
                raise ValueError(
                    f"No historical exchange rate data available for {from_currency}/{to_currency}.\n"
                    f"   This currency pair may not be supported by Yahoo Finance.\n"
                    f"   Supported major currencies: USD, EUR, GBP, JPY, CAD, AUD, CHF"
                )

            # Invert the rates
            logger.info(f"Using inverted rates from {reverse_pair}")
            invert_rates = True
        else:
            invert_rates = False

        # Extract Close prices
        if isinstance(data, pd.DataFrame):
            # Handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                if "Close" in [col[0] for col in data.columns]:
                    rates = data["Close"].squeeze()
                else:
                    rates = data.iloc[:, 0].squeeze()
            elif "Close" in data.columns:
                rates = data["Close"]
            else:
                rates = data.iloc[:, 0]
        else:
            rates = data

        # Ensure it's a Series
        if isinstance(rates, pd.DataFrame):
            rates = rates.squeeze()

        # Invert if we used reverse pair
        if invert_rates:
            rates = 1.0 / rates

        # Validate we have valid data
        if rates.isna().all():
            raise ValueError(f"All exchange rate values are NaN for {forex_pair}")

        # Check for sufficient data points
        valid_count = rates.notna().sum()
        if valid_count < 10:
            logger.warning(
                f"Only {valid_count} valid exchange rate points found for {forex_pair}"
            )

        logger.info(
            f"Fetched {len(rates)} historical exchange rates for {from_currency}/{to_currency} ({valid_count} valid)"
        )

        # Cache the result
        _FX_CACHE[cache_key] = rates.copy()

        return rates

    except ValueError:
        # Re-raise ValueError as-is
        raise
    except Exception as e:
        logger.error(f"Error fetching historical exchange rates for {forex_pair}: {e}")
        raise ValueError(
            f"Could not fetch historical exchange rates for {from_currency}/{to_currency}.\n"
            f"   Error: {str(e)}\n"
            f"   Please check your internet connection and verify the currency pair is supported."
        )


def convert_prices_to_currency(
    prices: pd.DataFrame,
    tickers: List[str],
    target_currency: str,
    start_date: str = None,
    end_date: str = None,
    period: str = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """
    Convert price data from multiple currencies to a target currency using HISTORICAL exchange rates.

    Example: If target_currency is EUR and you have 5 EUR stocks + 1 USD stock,
             the USD stock prices will be converted to EUR using historical USD/EUR rates.

    Args:
        prices: DataFrame with historical prices for multiple tickers
        tickers: List of ticker symbols
        target_currency: Target currency code (e.g., "EUR")
        start_date: Start date for historical rates (YYYY-MM-DD format)
        end_date: End date for historical rates (YYYY-MM-DD format)
        period: Time period (e.g., "1y", "2y") - alternative to start/end dates

    Returns:
        Tuple of (converted_prices_dataframe, historical_exchange_rates_dict)
        where historical_exchange_rates_dict maps currency -> pd.Series of rates

    Raises:
        ValueError: If inputs are invalid or conversion fails
    """
    # Validate inputs
    if prices is None or prices.empty:
        raise ValueError("Price data is empty or None")

    if not tickers:
        raise ValueError("Ticker list is empty")

    if not target_currency:
        raise ValueError("Target currency is not specified")

    target_currency = target_currency.upper().strip()

    if len(target_currency) != 3:
        raise ValueError(
            f"Invalid target currency: {target_currency}. Must be a 3-letter code (e.g., USD, EUR)"
        )

    # Validate target currency is supported
    if target_currency not in SUPPORTED_CURRENCIES:
        raise ValueError(
            f"Unsupported target currency: {target_currency}. "
            f"Supported currencies: {', '.join(SUPPORTED_CURRENCIES)}\n\n"
            f"If you need support for additional currencies, please check if they are available "
            f"on Yahoo Finance and submit a feature request."
        )

    currency_map = detect_portfolio_currencies(tickers)
    exchange_rates = {}
    converted_prices = prices.copy()

    # Identify unique currencies
    unique_currencies = set(currency_map.values())

    # If all stocks are already in target currency, no conversion needed
    if unique_currencies == {target_currency}:
        logger.info(f"All stocks already in {target_currency}, no conversion needed")
        return converted_prices, {target_currency: pd.Series(1.0, index=prices.index)}

    # Get historical exchange rates and convert
    logger.info(
        f"Converting prices to {target_currency} using historical exchange rates"
    )
    conversion_errors = []

    for ticker in tickers:
        ticker_currency = currency_map[ticker]

        if ticker_currency != target_currency:
            try:
                # Fetch historical exchange rates
                if ticker_currency not in exchange_rates:
                    rates_series = get_historical_exchange_rates(
                        ticker_currency, target_currency, start_date, end_date, period
                    )
                    exchange_rates[ticker_currency] = rates_series

                # Convert prices using historical rates
                rates_series = exchange_rates[ticker_currency]
                if ticker in converted_prices.columns:
                    # Align the exchange rates with the price data dates
                    # Use forward-fill to handle different trading calendars
                    aligned_rates = rates_series.reindex(
                        converted_prices.index, method="ffill", limit=MAX_FX_FORWARD_FILL_DAYS
                    )

                    # Check for missing alignment and stale forward-fills
                    missing_count = aligned_rates.isna().sum()
                    if missing_count > 0:
                        logger.warning(
                            f"{ticker}: {missing_count}/{len(aligned_rates)} exchange rates missing after forward-fill"
                        )
                        # Backward fill from beginning if needed, then forward-fill any remaining
                        aligned_rates = aligned_rates.fillna(method="bfill").fillna(
                            method="ffill"
                        )

                        # Check if we exceeded the safe window
                        if missing_count > MAX_FX_FORWARD_FILL_DAYS:
                            logger.warning(
                                f"{ticker}: Forward-filled FX rates beyond {MAX_FX_FORWARD_FILL_DAYS} days "
                                f"({missing_count} gaps). Results may use stale exchange rates."
                            )
                            print(
                                f"   ⚠️  Warning: {ticker} has {missing_count} days with stale FX rates "
                                f"(>{MAX_FX_FORWARD_FILL_DAYS} day limit)",
                                file=sys.stderr
                            )

                    # Perform conversion
                    converted_prices[ticker] = converted_prices[ticker] * aligned_rates

                    # Validate conversion
                    if converted_prices[ticker].isna().all():
                        raise ValueError(f"All converted prices are NaN for {ticker}")

                    logger.debug(
                        f"Converted {ticker} from {ticker_currency} to {target_currency} using historical rates"
                    )

            except Exception as e:
                error_msg = f"{ticker} ({ticker_currency}→{target_currency}): {str(e)}"
                conversion_errors.append(error_msg)
                logger.error(error_msg)

    # If any conversions failed, report them
    if conversion_errors:
        error_summary = "\n   - ".join(conversion_errors)

        # Add helpful context based on date range
        date_context = ""
        if start_date:
            year = int(start_date[:4])
            if year < 1999 and target_currency == "EUR":
                date_context = (
                    f"\n   NOTE: EUR was introduced in 1999. Historical data before 1999 is not available.\n"
                    f"         Consider using USD as portfolio currency for pre-1999 analysis."
                )
            elif year < 1990:
                date_context = (
                    f"\n   NOTE: Historical forex data for {start_date[:4]} may be limited or unavailable.\n"
                    f"         Yahoo Finance typically has reliable data from ~1990 onwards."
                )

        raise ValueError(
            f"Currency conversion failed for some tickers:\n   - {error_summary}\n\n"
            f"Possible reasons:\n"
            f"   - Currency pair not supported by Yahoo Finance\n"
            f"   - No historical exchange rate data for the specified period\n"
            f"   - Network connectivity issues{date_context}"
        )

    # Add target currency rate
    exchange_rates[target_currency] = pd.Series(1.0, index=prices.index)

    return converted_prices, exchange_rates


def format_currency_value(value: float, currency: str) -> str:
    """
    Format currency values with proper symbol and sign placement.

    Args:
        value: Numeric value
        currency: Currency code (e.g., "USD", "EUR")

    Returns:
        Formatted string (e.g., "$1,234.56", "€1,234.56")

    Examples:
        1234.56, "USD" -> "$1,234.56"
        -1234.56, "EUR" -> "-€1,234.56"
    """
    symbol = get_currency_symbol(currency)

    if value < 0:
        return f"-{symbol}{abs(value):,.2f}"
    else:
        return f"{symbol}{value:,.2f}"


def format_currency_value_with_sign(value: float, currency: str) -> str:
    """
    Format currency with explicit +/- sign.

    Args:
        value: Numeric value
        currency: Currency code

    Returns:
        Formatted string with explicit sign

    Examples:
        1234.56, "USD" -> "+$1,234.56"
        -1234.56, "EUR" -> "-€1,234.56"
    """
    symbol = get_currency_symbol(currency)

    if value < 0:
        return f"-{symbol}{abs(value):,.2f}"
    else:
        return f"+{symbol}{value:,.2f}"


def print_currency_conversion_info(
    tickers: List[str],
    target_currency: str,
    exchange_rates: Dict[str, pd.Series],
) -> None:
    """
    Print information about currency conversions performed.

    Args:
        tickers: List of ticker symbols
        target_currency: Target currency code
        exchange_rates: Dictionary mapping currency -> pd.Series of historical rates
    """
    currency_map = detect_portfolio_currencies(tickers)
    unique_currencies = set(currency_map.values())

    # Check if conversion was needed
    if len(unique_currencies) > 1 or (
        len(unique_currencies) == 1 and list(unique_currencies)[0] != target_currency
    ):
        print(f"\n💱 Currency Conversion:")
        print(f"   Portfolio Currency: {target_currency}")

        # Group tickers by currency
        by_currency = {}
        for ticker, currency in currency_map.items():
            if currency not in by_currency:
                by_currency[currency] = []
            by_currency[currency].append(ticker)

        # Display conversion info with as-of dates
        for currency, ticker_list in sorted(by_currency.items()):
            if currency != target_currency:
                rates_series = exchange_rates.get(currency)
                if rates_series is not None and len(rates_series) > 0:
                    avg_rate = rates_series.mean()
                    latest_rate = rates_series.iloc[-1]
                    latest_date = rates_series.index[-1].strftime("%Y-%m-%d")
                    first_date = rates_series.index[0].strftime("%Y-%m-%d")

                    print(
                        f"   {currency} → {target_currency}: {latest_rate:.4f} (as of {latest_date}), "
                        f"{avg_rate:.4f} (avg)"
                    )
                    print(f"      Tickers: {', '.join(ticker_list)}")
                    print(f"      Period: {first_date} to {latest_date} ({len(rates_series)} days)")
                else:
                    print(
                        f"   {currency} → {target_currency}: historical rates - {', '.join(ticker_list)}"
                    )
            else:
                print(f"   {currency}: {', '.join(ticker_list)} (no conversion)")
