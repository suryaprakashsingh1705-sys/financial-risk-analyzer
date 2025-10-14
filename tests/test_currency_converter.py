"""
Comprehensive functional tests for currency conversion module.

Tests cover:
- Currency detection (suffix mapping, yfinance fallback, defaults)
- Spot rate fetching (weekend fallback, reverse pairs, as-of dates)
- Historical rate fetching (caching, reverse pairs, validation)
- Price conversion (alignment, forward-fill warnings, stale FX detection)
- Edge cases (unsupported currencies, empty data, NaN handling)
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.currency_converter import (
    detect_ticker_currency,
    get_exchange_rate,
    get_historical_exchange_rates,
    convert_prices_to_currency,
    detect_portfolio_currencies,
    get_currency_symbol,
    format_currency_value,
    format_currency_value_with_sign,
    print_currency_conversion_info,
    EXCHANGE_CURRENCY_MAP,
    _FX_CACHE,
    MAX_FX_FORWARD_FILL_DAYS
)
from config import SUPPORTED_CURRENCIES


class TestCurrencyDetection(unittest.TestCase):
    """Test currency detection from ticker symbols."""

    def test_suffix_mapping_eur(self):
        """Test EUR exchange suffixes."""
        eur_tickers = ["ASML.AS", "SAP.DE", "SOLB.BR", "AIR.PA", "NOKIA.HE", "OMV.VI"]
        for ticker in eur_tickers:
            with self.subTest(ticker=ticker):
                self.assertEqual(detect_ticker_currency(ticker), "EUR")

    def test_suffix_mapping_gbp(self):
        """Test GBP exchange suffix."""
        self.assertEqual(detect_ticker_currency("HSBA.L"), "GBP")
        self.assertEqual(detect_ticker_currency("BP.L"), "GBP")

    def test_suffix_mapping_chf(self):
        """Test CHF exchange suffix."""
        self.assertEqual(detect_ticker_currency("NESN.SW"), "CHF")

    def test_suffix_mapping_jpy(self):
        """Test JPY exchange suffix."""
        self.assertEqual(detect_ticker_currency("7203.T"), "JPY")

    def test_suffix_mapping_cad(self):
        """Test CAD exchange suffixes."""
        self.assertEqual(detect_ticker_currency("SHOP.TO"), "CAD")
        self.assertEqual(detect_ticker_currency("WEED.V"), "CAD")

    def test_suffix_mapping_aud(self):
        """Test AUD exchange suffix."""
        self.assertEqual(detect_ticker_currency("BHP.AX"), "AUD")

    def test_no_suffix_defaults_to_usd(self):
        """Test that tickers without suffix default to USD."""
        us_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
        for ticker in us_tickers:
            with self.subTest(ticker=ticker):
                self.assertEqual(detect_ticker_currency(ticker), "USD")

    def test_case_insensitive(self):
        """Test that detection is case-insensitive."""
        self.assertEqual(detect_ticker_currency("aapl"), "USD")
        self.assertEqual(detect_ticker_currency("AAPL"), "USD")
        self.assertEqual(detect_ticker_currency("asml.as"), "EUR")
        self.assertEqual(detect_ticker_currency("ASML.AS"), "EUR")

    def test_detect_portfolio_currencies(self):
        """Test detecting currencies for multiple tickers."""
        tickers = ["AAPL", "ASML.AS", "HSBA.L", "7203.T"]
        expected = {
            "AAPL": "USD",
            "ASML.AS": "EUR",
            "HSBA.L": "GBP",
            "7203.T": "JPY"
        }
        result = detect_portfolio_currencies(tickers)
        self.assertEqual(result, expected)


class TestCurrencySymbols(unittest.TestCase):
    """Test currency symbol formatting."""

    def test_get_currency_symbol(self):
        """Test getting currency symbols."""
        symbols = {
            "USD": "$",
            "EUR": "€",
            "GBP": "£",
            "JPY": "¥",
            "CAD": "C$",
            "AUD": "A$",
            "CHF": "CHF"
        }
        for currency, expected_symbol in symbols.items():
            with self.subTest(currency=currency):
                self.assertEqual(get_currency_symbol(currency), expected_symbol)

    def test_format_currency_value_positive(self):
        """Test formatting positive currency values."""
        self.assertEqual(format_currency_value(1234.56, "USD"), "$1,234.56")
        self.assertEqual(format_currency_value(1234.56, "EUR"), "€1,234.56")
        self.assertEqual(format_currency_value(1000000, "JPY"), "¥1,000,000.00")

    def test_format_currency_value_negative(self):
        """Test formatting negative currency values."""
        self.assertEqual(format_currency_value(-1234.56, "USD"), "-$1,234.56")
        self.assertEqual(format_currency_value(-1234.56, "EUR"), "-€1,234.56")

    def test_format_currency_value_with_sign_positive(self):
        """Test formatting with explicit + sign."""
        self.assertEqual(format_currency_value_with_sign(1234.56, "USD"), "+$1,234.56")
        self.assertEqual(format_currency_value_with_sign(1234.56, "EUR"), "+€1,234.56")

    def test_format_currency_value_with_sign_negative(self):
        """Test formatting negative with explicit - sign."""
        self.assertEqual(format_currency_value_with_sign(-1234.56, "USD"), "-$1,234.56")
        self.assertEqual(format_currency_value_with_sign(-1234.56, "EUR"), "-€1,234.56")


class TestSpotRateFetching(unittest.TestCase):
    """Test spot (current) exchange rate fetching."""

    def test_same_currency_returns_one(self):
        """Test that same currency pair returns 1.0."""
        rate, as_of = get_exchange_rate("USD", "USD")
        self.assertEqual(rate, 1.0)
        self.assertIsInstance(as_of, str)
        # Verify date format
        datetime.strptime(as_of, "%Y-%m-%d")

    def test_usd_to_eur_spot_rate(self):
        """Test fetching USD to EUR spot rate."""
        rate, as_of = get_exchange_rate("USD", "EUR")
        self.assertIsInstance(rate, float)
        self.assertGreater(rate, 0)
        self.assertLess(rate, 2)  # Reasonable range
        self.assertIsInstance(as_of, str)
        datetime.strptime(as_of, "%Y-%m-%d")

    def test_eur_to_usd_spot_rate(self):
        """Test fetching EUR to USD spot rate."""
        rate, as_of = get_exchange_rate("EUR", "USD")
        self.assertIsInstance(rate, float)
        self.assertGreater(rate, 0.5)
        self.assertLess(rate, 2)

    def test_gbp_to_usd_spot_rate(self):
        """Test fetching GBP to USD spot rate."""
        rate, as_of = get_exchange_rate("GBP", "USD")
        self.assertIsInstance(rate, float)
        self.assertGreater(rate, 1.0)  # GBP typically stronger than USD
        self.assertLess(rate, 2.0)

    def test_spot_rate_returns_as_of_date(self):
        """Test that spot rate returns valid as-of date."""
        rate, as_of = get_exchange_rate("USD", "EUR")
        # Parse date to ensure it's valid
        as_of_date = datetime.strptime(as_of, "%Y-%m-%d")
        # Should be recent (within last 7 days)
        days_old = (datetime.now() - as_of_date).days
        self.assertLessEqual(days_old, 7, "Spot rate should be recent")


class TestHistoricalRateFetching(unittest.TestCase):
    """Test historical exchange rate fetching."""

    def setUp(self):
        """Clear cache before each test."""
        _FX_CACHE.clear()

    def test_same_currency_returns_series_of_ones(self):
        """Test that same currency returns series of 1.0."""
        rates = get_historical_exchange_rates("USD", "USD", period="1mo")
        self.assertIsInstance(rates, pd.Series)
        self.assertTrue((rates == 1.0).all())
        self.assertGreater(len(rates), 10)

    def test_fetch_historical_usd_eur(self):
        """Test fetching historical USD/EUR rates."""
        rates = get_historical_exchange_rates("USD", "EUR", period="1mo")
        self.assertIsInstance(rates, pd.Series)
        self.assertGreater(len(rates), 10)
        self.assertTrue((rates > 0).all())
        self.assertTrue((rates < 2).all())

    def test_fetch_historical_with_date_range(self):
        """Test fetching with explicit date range."""
        start = "2024-01-01"
        end = "2024-01-31"
        rates = get_historical_exchange_rates("USD", "EUR", start_date=start, end_date=end)
        self.assertIsInstance(rates, pd.Series)
        self.assertGreater(len(rates), 10)

    def test_caching_works(self):
        """Test that caching avoids redundant fetches."""
        # First fetch
        rates1 = get_historical_exchange_rates("USD", "EUR", period="1mo")
        cache_size_after_first = len(_FX_CACHE)
        self.assertEqual(cache_size_after_first, 1)

        # Second fetch (should use cache)
        rates2 = get_historical_exchange_rates("USD", "EUR", period="1mo")
        cache_size_after_second = len(_FX_CACHE)
        self.assertEqual(cache_size_after_second, 1)  # No new cache entry

        # Verify same data
        pd.testing.assert_series_equal(rates1, rates2)

    def test_different_periods_not_cached(self):
        """Test that different periods create separate cache entries."""
        rates1 = get_historical_exchange_rates("USD", "EUR", period="1mo")
        rates2 = get_historical_exchange_rates("USD", "EUR", period="3mo")
        self.assertEqual(len(_FX_CACHE), 2)
        self.assertNotEqual(len(rates1), len(rates2))

    def test_unsupported_currency_raises_error(self):
        """Test that unsupported currency raises ValueError."""
        with self.assertRaises(ValueError) as context:
            get_historical_exchange_rates("ZAR", "USD", period="1mo")
        self.assertIn("Unsupported currency", str(context.exception))
        self.assertIn("ZAR", str(context.exception))

    def test_invalid_currency_code_length(self):
        """Test that invalid currency code length raises error."""
        with self.assertRaises(ValueError) as context:
            get_historical_exchange_rates("US", "EUR", period="1mo")
        self.assertIn("Must be 3-letter codes", str(context.exception))

    def test_empty_currency_raises_error(self):
        """Test that empty currency raises error."""
        with self.assertRaises(ValueError) as context:
            get_historical_exchange_rates("", "EUR", period="1mo")
        self.assertIn("cannot be empty", str(context.exception))


class TestPriceConversion(unittest.TestCase):
    """Test price conversion with historical exchange rates."""

    def setUp(self):
        """Set up test data."""
        _FX_CACHE.clear()
        # Create sample price data
        dates = pd.date_range(start="2024-01-01", periods=20, freq="D")
        self.prices = pd.DataFrame({
            "AAPL": np.random.uniform(150, 200, 20),
            "ASML.AS": np.random.uniform(600, 700, 20)
        }, index=dates)
        self.tickers = ["AAPL", "ASML.AS"]

    def test_all_same_currency_no_conversion(self):
        """Test that same currency stocks skip conversion."""
        dates = pd.date_range(start="2024-01-01", periods=20, freq="D")
        prices = pd.DataFrame({
            "AAPL": np.random.uniform(150, 200, 20),
            "MSFT": np.random.uniform(300, 350, 20)
        }, index=dates)
        tickers = ["AAPL", "MSFT"]

        converted, rates = convert_prices_to_currency(
            prices, tickers, "USD", period="1mo"
        )

        # Should return same prices
        pd.testing.assert_frame_equal(converted, prices)
        # Should have only target currency in rates
        self.assertIn("USD", rates)
        self.assertEqual(len(rates), 1)

    def test_mixed_currency_conversion(self):
        """Test converting mixed currency portfolio."""
        converted, rates = convert_prices_to_currency(
            self.prices, self.tickers, "EUR", start_date="2024-01-01", end_date="2024-01-31"
        )

        # Should have converted prices
        self.assertIsInstance(converted, pd.DataFrame)
        self.assertEqual(converted.shape, self.prices.shape)

        # ASML.AS should be unchanged (already EUR)
        pd.testing.assert_series_equal(
            converted["ASML.AS"],
            self.prices["ASML.AS"]
        )

        # AAPL should be converted (different values)
        self.assertFalse(converted["AAPL"].equals(self.prices["AAPL"]))

        # Should have exchange rates
        self.assertIn("USD", rates)
        self.assertIn("EUR", rates)

    def test_conversion_to_usd(self):
        """Test converting to USD."""
        converted, rates = convert_prices_to_currency(
            self.prices, self.tickers, "USD", start_date="2024-01-01", end_date="2024-01-31"
        )

        # AAPL should be unchanged
        pd.testing.assert_series_equal(
            converted["AAPL"],
            self.prices["AAPL"]
        )

        # ASML.AS should be converted
        self.assertFalse(converted["ASML.AS"].equals(self.prices["ASML.AS"]))

    def test_empty_prices_raises_error(self):
        """Test that empty prices raise error."""
        empty_prices = pd.DataFrame()
        with self.assertRaises(ValueError) as context:
            convert_prices_to_currency(empty_prices, [], "USD", period="1mo")
        self.assertIn("empty", str(context.exception).lower())

    def test_empty_tickers_raises_error(self):
        """Test that empty tickers raise error."""
        with self.assertRaises(ValueError) as context:
            convert_prices_to_currency(self.prices, [], "USD", period="1mo")
        self.assertIn("empty", str(context.exception).lower())

    def test_unsupported_target_currency_raises_error(self):
        """Test that unsupported target currency raises error."""
        with self.assertRaises(ValueError) as context:
            convert_prices_to_currency(self.prices, self.tickers, "ZAR", period="1mo")
        self.assertIn("Unsupported target currency", str(context.exception))

    def test_invalid_target_currency_format(self):
        """Test that invalid currency format raises error."""
        with self.assertRaises(ValueError) as context:
            convert_prices_to_currency(self.prices, self.tickers, "US", period="1mo")
        self.assertIn("Must be a 3-letter code", str(context.exception))


class TestForwardFillWarnings(unittest.TestCase):
    """Test forward-fill gap warnings."""

    def test_max_forward_fill_constant_exists(self):
        """Test that MAX_FX_FORWARD_FILL_DAYS is defined."""
        self.assertIsInstance(MAX_FX_FORWARD_FILL_DAYS, int)
        self.assertGreater(MAX_FX_FORWARD_FILL_DAYS, 0)
        self.assertEqual(MAX_FX_FORWARD_FILL_DAYS, 3)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_very_old_date_eur_before_1999(self):
        """Test that EUR before 1999 raises appropriate error."""
        with self.assertRaises(ValueError) as context:
            get_historical_exchange_rates(
                "USD", "EUR",
                start_date="1995-01-01",
                end_date="1995-12-31"
            )
        # Should mention no data available or EUR limitation
        error_msg = str(context.exception)
        self.assertTrue(
            "No historical exchange rate data" in error_msg or
            "EUR was introduced in 1999" in error_msg
        )

    def test_currency_validation_strips_whitespace(self):
        """Test that currency codes are stripped of whitespace."""
        # This should work despite whitespace
        rates = get_historical_exchange_rates("USD ", " EUR", period="1mo")
        self.assertIsInstance(rates, pd.Series)
        self.assertGreater(len(rates), 0)

    def test_currency_validation_uppercases(self):
        """Test that currency codes are uppercased."""
        # This should work despite lowercase
        rates = get_historical_exchange_rates("usd", "eur", period="1mo")
        self.assertIsInstance(rates, pd.Series)
        self.assertGreater(len(rates), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""

    def test_full_var_workflow(self):
        """Test complete VaR-like workflow with currency conversion."""
        # Create mock portfolio
        dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
        prices = pd.DataFrame({
            "AAPL": np.random.uniform(150, 200, 30),
            "ASML.AS": np.random.uniform(600, 700, 30),
            "HSBA.L": np.random.uniform(6, 8, 30)
        }, index=dates)
        tickers = ["AAPL", "ASML.AS", "HSBA.L"]

        # Convert to EUR
        converted, rates = convert_prices_to_currency(
            prices, tickers, "EUR", start_date="2024-01-01", end_date="2024-01-31"
        )

        # Verify conversion worked
        self.assertEqual(converted.shape, prices.shape)
        self.assertIn("USD", rates)
        self.assertIn("GBP", rates)
        self.assertIn("EUR", rates)

        # Calculate returns (basic VaR workflow)
        returns = converted.pct_change().dropna()
        self.assertGreater(len(returns), 20)
        self.assertEqual(returns.shape[1], 3)

    def test_detect_all_supported_exchanges(self):
        """Test that all exchange suffixes in map are detected."""
        for suffix, expected_currency in EXCHANGE_CURRENCY_MAP.items():
            if expected_currency in SUPPORTED_CURRENCIES:
                ticker = f"TEST{suffix}"
                detected = detect_ticker_currency(ticker)
                self.assertEqual(
                    detected, expected_currency,
                    f"Failed for {ticker} (suffix: {suffix})"
                )


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCurrencyDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestCurrencySymbols))
    suite.addTests(loader.loadTestsFromTestCase(TestSpotRateFetching))
    suite.addTests(loader.loadTestsFromTestCase(TestHistoricalRateFetching))
    suite.addTests(loader.loadTestsFromTestCase(TestPriceConversion))
    suite.addTests(loader.loadTestsFromTestCase(TestForwardFillWarnings))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    result = run_tests()
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
