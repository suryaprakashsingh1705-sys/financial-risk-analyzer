"""
Comprehensive tests for the Stress Test module.

Tests historical crisis scenario simulations and portfolio stress analysis.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

from modules.stress_test import (
    calculate_max_drawdown,
    calculate_portfolio_stress,
    SCENARIOS,
    CrisisScenario,
    StressTestResult,
)
from config import (
    CRISIS_DOTCOM_START,
    CRISIS_DOTCOM_END,
    CRISIS_GFC_START,
    CRISIS_GFC_END,
    CRISIS_COVID_START,
    CRISIS_COVID_END,
)


class TestCrisisScenarios(unittest.TestCase):
    """Test crisis scenario definitions."""

    def test_scenarios_exist(self):
        """Test that all crisis scenarios are defined."""
        self.assertIn("dotcom", SCENARIOS)
        self.assertIn("gfc", SCENARIOS)
        self.assertIn("covid", SCENARIOS)

    def test_dotcom_scenario(self):
        """Test Dot-Com Bubble scenario configuration."""
        scenario = SCENARIOS["dotcom"]
        self.assertEqual(scenario.name, "Dot-Com Bubble Burst")
        self.assertEqual(scenario.start_date, CRISIS_DOTCOM_START)
        self.assertEqual(scenario.end_date, CRISIS_DOTCOM_END)
        self.assertLess(scenario.market_drop, 0)  # Should be negative (loss)

    def test_gfc_scenario(self):
        """Test Global Financial Crisis scenario configuration."""
        scenario = SCENARIOS["gfc"]
        self.assertEqual(scenario.name, "Global Financial Crisis")
        self.assertEqual(scenario.start_date, CRISIS_GFC_START)
        self.assertEqual(scenario.end_date, CRISIS_GFC_END)
        self.assertLess(scenario.market_drop, 0)

    def test_covid_scenario(self):
        """Test COVID-19 scenario configuration."""
        scenario = SCENARIOS["covid"]
        self.assertEqual(scenario.name, "COVID-19 Crash")
        self.assertEqual(scenario.start_date, CRISIS_COVID_START)
        self.assertEqual(scenario.end_date, CRISIS_COVID_END)
        self.assertLess(scenario.market_drop, 0)

    def test_scenario_dates_are_valid(self):
        """Test that all scenario dates are valid format."""
        for scenario in SCENARIOS.values():
            # Should be able to parse dates
            start = pd.to_datetime(scenario.start_date)
            end = pd.to_datetime(scenario.end_date)
            # End should be after start
            self.assertGreater(end, start)


class TestCrisisScenarioDataclass(unittest.TestCase):
    """Test CrisisScenario dataclass."""

    def test_create_crisis_scenario(self):
        """Test creating a CrisisScenario object."""
        scenario = CrisisScenario(
            name="Test Crisis",
            description="Test scenario",
            start_date="2020-01-01",
            end_date="2020-12-31",
            market_drop=-25.5,
        )

        self.assertEqual(scenario.name, "Test Crisis")
        self.assertEqual(scenario.description, "Test scenario")
        self.assertEqual(scenario.start_date, "2020-01-01")
        self.assertEqual(scenario.end_date, "2020-12-31")
        self.assertEqual(scenario.market_drop, -25.5)


class TestStressTestResultDataclass(unittest.TestCase):
    """Test StressTestResult dataclass."""

    def test_create_stress_test_result(self):
        """Test creating a StressTestResult object."""
        result = StressTestResult(
            ticker="AAPL",
            scenario="COVID-19 Crash",
            start_price=100.0,
            end_price=80.0,
            return_pct=-20.0,
            position_value_start=10000.0,
            position_value_end=8000.0,
            loss_amount=-2000.0,
            max_drawdown=-25.0,
        )

        self.assertEqual(result.ticker, "AAPL")
        self.assertEqual(result.scenario, "COVID-19 Crash")
        self.assertEqual(result.start_price, 100.0)
        self.assertEqual(result.end_price, 80.0)
        self.assertEqual(result.return_pct, -20.0)
        self.assertEqual(result.position_value_start, 10000.0)
        self.assertEqual(result.position_value_end, 8000.0)
        self.assertEqual(result.loss_amount, -2000.0)
        self.assertEqual(result.max_drawdown, -25.0)


class TestMaxDrawdown(unittest.TestCase):
    """Test maximum drawdown calculation."""

    def test_max_drawdown_declining_prices(self):
        """Test max drawdown with steadily declining prices."""
        prices = pd.Series([100, 90, 80, 70, 60])
        max_dd = calculate_max_drawdown(prices)

        # From 100 to 60 is -40%
        self.assertAlmostEqual(max_dd, -40.0, places=1)

    def test_max_drawdown_with_recovery(self):
        """Test max drawdown with decline and partial recovery."""
        prices = pd.Series([100, 80, 90, 85, 95])
        max_dd = calculate_max_drawdown(prices)

        # Max drawdown is from 100 to 80 = -20%
        self.assertAlmostEqual(max_dd, -20.0, places=1)

    def test_max_drawdown_v_shaped_crash(self):
        """Test max drawdown with V-shaped crash and full recovery."""
        prices = pd.Series([100, 90, 70, 85, 100])
        max_dd = calculate_max_drawdown(prices)

        # Max drawdown is from 100 to 70 = -30%
        self.assertAlmostEqual(max_dd, -30.0, places=1)

    def test_max_drawdown_rising_prices(self):
        """Test max drawdown with only rising prices."""
        prices = pd.Series([100, 105, 110, 115, 120])
        max_dd = calculate_max_drawdown(prices)

        # No drawdown (should be 0 or very small)
        self.assertGreaterEqual(max_dd, -0.1)
        self.assertLessEqual(max_dd, 0.0)

    def test_max_drawdown_flat_prices(self):
        """Test max drawdown with flat prices."""
        prices = pd.Series([100, 100, 100, 100, 100])
        max_dd = calculate_max_drawdown(prices)

        # No drawdown
        self.assertEqual(max_dd, 0.0)

    def test_max_drawdown_multiple_peaks(self):
        """Test max drawdown with multiple peaks."""
        prices = pd.Series([100, 80, 120, 90, 110, 70, 100])
        max_dd = calculate_max_drawdown(prices)

        # Max drawdown is from 120 to 70 = -41.67%
        expected_dd = ((70 - 120) / 120) * 100
        self.assertAlmostEqual(max_dd, expected_dd, places=1)

    def test_max_drawdown_single_value(self):
        """Test max drawdown with single value."""
        prices = pd.Series([100])
        max_dd = calculate_max_drawdown(prices)

        # No drawdown with single value
        self.assertEqual(max_dd, 0.0)

    def test_max_drawdown_realistic_crash(self):
        """Test max drawdown with realistic crash scenario."""
        # Simulate a 50% crash
        prices = pd.Series([100, 95, 85, 70, 60, 50, 55, 65, 70])
        max_dd = calculate_max_drawdown(prices)

        # From 100 to 50 = -50%
        self.assertAlmostEqual(max_dd, -50.0, places=1)


class TestPortfolioStressCalculation(unittest.TestCase):
    """Test portfolio-level stress test calculation."""

    def setUp(self):
        """Set up test data."""
        # Create sample price data for two assets
        dates = pd.date_range(start="2020-01-01", periods=60, freq="D")

        # Asset 1: Declines 20%
        asset1_prices = pd.Series(
            np.linspace(100, 80, 60), index=dates, name="ASSET1"
        )

        # Asset 2: Declines 30%
        asset2_prices = pd.Series(
            np.linspace(100, 70, 60), index=dates, name="ASSET2"
        )

        self.prices = pd.DataFrame({"ASSET1": asset1_prices, "ASSET2": asset2_prices})
        self.weights = np.array([0.6, 0.4])
        self.portfolio_value = 100000

    def test_calculate_portfolio_stress_basic(self):
        """Test basic portfolio stress calculation."""
        result = calculate_portfolio_stress(
            self.prices, self.weights, self.portfolio_value, "Test Scenario"
        )

        self.assertIsInstance(result, StressTestResult)
        self.assertEqual(result.ticker, "PORTFOLIO")
        self.assertEqual(result.scenario, "Test Scenario")
        self.assertLess(result.return_pct, 0)  # Should be negative (loss)
        self.assertLess(result.loss_amount, 0)  # Should be negative (loss)

    def test_portfolio_stress_weighted_correctly(self):
        """Test that portfolio stress correctly weights assets."""
        result = calculate_portfolio_stress(
            self.prices, self.weights, self.portfolio_value, "Test"
        )

        # Portfolio return should be weighted average
        # Asset1: -20%, Asset2: -30%
        # Expected: 0.6*(-20) + 0.4*(-30) = -12 - 12 = -24%
        # Allow some tolerance due to compounding
        self.assertLess(result.return_pct, -20)
        self.assertGreater(result.return_pct, -30)

    def test_portfolio_stress_equal_weights(self):
        """Test portfolio stress with equal weights."""
        equal_weights = np.array([0.5, 0.5])
        result = calculate_portfolio_stress(
            self.prices, equal_weights, 100000, "Test"
        )

        # With equal weights: 0.5*(-20) + 0.5*(-30) = -25%
        self.assertLess(result.return_pct, -20)
        self.assertGreater(result.return_pct, -30)

    def test_portfolio_stress_single_asset(self):
        """Test portfolio stress with single asset (100% weight)."""
        single_prices = pd.DataFrame({"ASSET1": self.prices["ASSET1"]})
        single_weights = np.array([1.0])

        result = calculate_portfolio_stress(
            single_prices, single_weights, 100000, "Test"
        )

        # Should match asset1 return (-20%)
        self.assertAlmostEqual(result.return_pct, -20.0, places=0)

    def test_portfolio_stress_gain_scenario(self):
        """Test portfolio stress with positive returns (gain)."""
        # Create rising prices
        dates = pd.date_range(start="2020-01-01", periods=60, freq="D")
        rising_prices = pd.DataFrame(
            {
                "ASSET1": np.linspace(100, 120, 60),
                "ASSET2": np.linspace(100, 115, 60),
            },
            index=dates,
        )

        result = calculate_portfolio_stress(
            rising_prices, self.weights, 100000, "Bull Market"
        )

        # Should have positive returns
        self.assertGreater(result.return_pct, 0)
        self.assertGreater(result.loss_amount, 0)  # Positive = gain

    def test_portfolio_stress_values_consistency(self):
        """Test that portfolio stress values are internally consistent."""
        result = calculate_portfolio_stress(
            self.prices, self.weights, self.portfolio_value, "Test"
        )

        # Check consistency
        self.assertEqual(result.position_value_start, self.portfolio_value)

        # loss_amount = end_value - start_value
        expected_loss = result.position_value_end - result.position_value_start
        self.assertAlmostEqual(result.loss_amount, expected_loss, places=2)

        # return_pct = (end - start) / start * 100
        expected_return = (
            (result.position_value_end - result.position_value_start)
            / result.position_value_start
            * 100
        )
        self.assertAlmostEqual(result.return_pct, expected_return, places=1)

    def test_portfolio_stress_max_drawdown(self):
        """Test that portfolio max drawdown is calculated."""
        result = calculate_portfolio_stress(
            self.prices, self.weights, self.portfolio_value, "Test"
        )

        # Max drawdown should be negative
        self.assertLess(result.max_drawdown, 0)
        # Max drawdown should be at least as bad as final return
        self.assertLessEqual(result.max_drawdown, result.return_pct)


class TestPortfolioDiversification(unittest.TestCase):
    """Test diversification effects in portfolio stress testing."""

    def test_diversification_reduces_drawdown(self):
        """Test that diversification reduces maximum drawdown."""
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")

        # Asset 1: Large decline
        asset1 = pd.Series(np.linspace(100, 50, 100), index=dates)

        # Asset 2: Small decline
        asset2 = pd.Series(np.linspace(100, 90, 100), index=dates)

        prices = pd.DataFrame({"ASSET1": asset1, "ASSET2": asset2})

        # Portfolio with both assets
        diversified_result = calculate_portfolio_stress(
            prices, np.array([0.5, 0.5]), 100000, "Test"
        )

        # Portfolio with only asset 1
        concentrated_result = calculate_portfolio_stress(
            pd.DataFrame({"ASSET1": asset1}), np.array([1.0]), 100000, "Test"
        )

        # Diversified portfolio should have smaller loss
        self.assertGreater(diversified_result.return_pct, concentrated_result.return_pct)
        self.assertGreater(diversified_result.max_drawdown, concentrated_result.max_drawdown)

    def test_correlation_effect(self):
        """Test that correlation affects portfolio stress."""
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        np.random.seed(42)

        # Create two assets with different correlation patterns
        base_returns = np.linspace(-0.5, 0, 100)

        # Positively correlated: both decline together
        asset1_pos = pd.Series(100 * np.cumprod(1 + base_returns + np.random.normal(0, 0.01, 100)), index=dates)
        asset2_pos = pd.Series(100 * np.cumprod(1 + base_returns + np.random.normal(0, 0.01, 100)), index=dates)

        prices_correlated = pd.DataFrame({"ASSET1": asset1_pos, "ASSET2": asset2_pos})

        # Negatively correlated: one goes up when other goes down
        asset1_neg = pd.Series(100 * np.cumprod(1 + base_returns), index=dates)
        asset2_neg = pd.Series(100 * np.cumprod(1 - base_returns), index=dates)

        prices_uncorrelated = pd.DataFrame({"ASSET1": asset1_neg, "ASSET2": asset2_neg})

        result_correlated = calculate_portfolio_stress(
            prices_correlated, np.array([0.5, 0.5]), 100000, "Test"
        )

        result_uncorrelated = calculate_portfolio_stress(
            prices_uncorrelated, np.array([0.5, 0.5]), 100000, "Test"
        )

        # Uncorrelated portfolio should have better (less negative) returns
        self.assertGreater(result_uncorrelated.return_pct, result_correlated.return_pct)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_portfolio_stress_with_volatile_prices(self):
        """Test portfolio stress with highly volatile prices."""
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        np.random.seed(42)

        # Create very volatile prices
        volatile_returns = np.random.normal(0, 0.05, 100)
        volatile_prices = pd.Series(
            100 * np.cumprod(1 + volatile_returns), index=dates, name="VOLATILE"
        )

        prices = pd.DataFrame({"VOLATILE": volatile_prices})
        weights = np.array([1.0])

        result = calculate_portfolio_stress(prices, weights, 100000, "Volatility Test")

        # Should complete without errors
        self.assertIsInstance(result, StressTestResult)
        self.assertIsInstance(result.max_drawdown, float)

    def test_portfolio_stress_with_flat_prices(self):
        """Test portfolio stress with no price movement."""
        dates = pd.date_range(start="2020-01-01", periods=50, freq="D")
        flat_prices = pd.DataFrame(
            {"FLAT": pd.Series([100] * 50, index=dates)}, index=dates
        )

        result = calculate_portfolio_stress(
            flat_prices, np.array([1.0]), 100000, "Flat Test"
        )

        # Return should be close to 0
        self.assertAlmostEqual(result.return_pct, 0.0, places=1)
        # Loss should be close to 0
        self.assertAlmostEqual(result.loss_amount, 0.0, places=0)
        # Max drawdown should be 0
        self.assertEqual(result.max_drawdown, 0.0)

    def test_portfolio_stress_short_period(self):
        """Test portfolio stress with very short time period."""
        dates = pd.date_range(start="2020-01-01", periods=2, freq="D")
        short_prices = pd.DataFrame(
            {"ASSET": pd.Series([100, 90], index=dates)}, index=dates
        )

        result = calculate_portfolio_stress(
            short_prices, np.array([1.0]), 100000, "Short Test"
        )

        # Should still calculate correctly
        self.assertAlmostEqual(result.return_pct, -10.0, places=1)

    def test_max_drawdown_with_recovery_above_start(self):
        """Test max drawdown when prices recover above starting point."""
        dates = pd.date_range(start="2020-01-01", periods=50, freq="D")
        # Start at 100, drop to 80, recover to 120
        prices_list = (
            list(np.linspace(100, 80, 20))
            + list(np.linspace(80, 120, 30))
        )
        prices = pd.Series(prices_list, index=dates)

        max_dd = calculate_max_drawdown(prices)

        # Max drawdown should be from 100 to 80 = -20%
        self.assertAlmostEqual(max_dd, -20.0, places=1)

    def test_portfolio_stress_extreme_weights(self):
        """Test portfolio stress with extreme weight distribution."""
        dates = pd.date_range(start="2020-01-01", periods=60, freq="D")
        prices = pd.DataFrame(
            {
                "ASSET1": np.linspace(100, 80, 60),
                "ASSET2": np.linspace(100, 50, 60),
            },
            index=dates,
        )

        # 99% in asset1, 1% in asset2
        extreme_weights = np.array([0.99, 0.01])

        result = calculate_portfolio_stress(
            prices, extreme_weights, 100000, "Extreme Weights"
        )

        # Return should be very close to asset1's return (-20%)
        self.assertAlmostEqual(result.return_pct, -20.0, delta=1.0)


class TestStressTestIntegration(unittest.TestCase):
    """Test integration scenarios combining multiple components."""

    def test_multi_asset_portfolio_complete_workflow(self):
        """Test complete stress test workflow with multiple assets."""
        dates = pd.date_range(start="2020-02-01", periods=60, freq="D")
        np.random.seed(42)

        # Create realistic crisis scenario prices
        # Tech stock: -35%
        tech_prices = pd.Series(100 * np.cumprod(1 + np.linspace(-0.01, -0.005, 60)), index=dates)

        # Safe haven: +5%
        safe_prices = pd.Series(100 * np.cumprod(1 + np.linspace(0.001, 0.0008, 60)), index=dates)

        # Industrial: -25%
        industrial_prices = pd.Series(100 * np.cumprod(1 + np.linspace(-0.008, -0.004, 60)), index=dates)

        prices = pd.DataFrame({
            "TECH": tech_prices,
            "SAFE": safe_prices,
            "INDUSTRIAL": industrial_prices,
        })

        # Diversified portfolio
        weights = np.array([0.4, 0.3, 0.3])

        result = calculate_portfolio_stress(prices, weights, 100000, "COVID-19 Crash")

        # Check all components
        self.assertEqual(result.ticker, "PORTFOLIO")
        self.assertLess(result.return_pct, 0)  # Overall loss
        self.assertGreater(result.return_pct, -35)  # But not as bad as worst asset
        self.assertLess(result.max_drawdown, 0)
        self.assertLess(result.loss_amount, 0)
        self.assertEqual(result.position_value_start, 100000)


if __name__ == "__main__":
    unittest.main()
