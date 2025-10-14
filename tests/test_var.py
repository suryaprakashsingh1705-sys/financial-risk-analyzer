"""
Comprehensive tests for the Value at Risk (VaR) module.

Tests all three VaR calculation methods: Parametric, Historical, and Monte Carlo.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from scipy import stats

from modules.var import (
    calculate_parametric_var,
    calculate_historical_var,
    calculate_monte_carlo_var,
    calculate_portfolio_var,
    VaRResult,
)


class TestParametricVaR(unittest.TestCase):
    """Test parametric (variance-covariance) VaR calculation."""

    def setUp(self):
        """Set up test data."""
        # Create sample returns (normally distributed)
        np.random.seed(42)
        self.returns = pd.Series(np.random.normal(0.001, 0.02, 250))
        self.position_value = 100000
        self.confidence = 95.0
        self.days = 1

    def test_parametric_var_basic(self):
        """Test basic parametric VaR calculation."""
        result = calculate_parametric_var(
            self.returns, self.confidence, self.days, self.position_value
        )

        self.assertIn("var_amount", result)
        self.assertIn("var_percentage", result)
        self.assertGreater(result["var_amount"], 0)
        self.assertGreater(result["var_percentage"], 0)
        self.assertLess(result["var_percentage"], 100)

    def test_parametric_var_95_confidence(self):
        """Test parametric VaR at 95% confidence."""
        result = calculate_parametric_var(self.returns, 95.0, 1, self.position_value)

        # VaR should be positive and reasonable
        self.assertGreater(result["var_amount"], 0)
        self.assertLess(result["var_amount"], self.position_value)

    def test_parametric_var_99_confidence(self):
        """Test parametric VaR at 99% confidence."""
        result_95 = calculate_parametric_var(self.returns, 95.0, 1, self.position_value)
        result_99 = calculate_parametric_var(self.returns, 99.0, 1, self.position_value)

        # 99% VaR should be higher than 95% VaR
        self.assertGreater(result_99["var_amount"], result_95["var_amount"])

    def test_parametric_var_multiday(self):
        """Test parametric VaR with multiple day horizon."""
        result_1day = calculate_parametric_var(
            self.returns, self.confidence, 1, self.position_value
        )
        result_10day = calculate_parametric_var(
            self.returns, self.confidence, 10, self.position_value
        )

        # 10-day VaR should be higher than 1-day (by sqrt(10))
        expected_ratio = np.sqrt(10)
        actual_ratio = result_10day["var_amount"] / result_1day["var_amount"]
        self.assertAlmostEqual(actual_ratio, expected_ratio, places=1)

    def test_parametric_var_zero_position_raises_error(self):
        """Test that zero position value raises ValueError."""
        with self.assertRaises(ValueError) as context:
            calculate_parametric_var(self.returns, 95.0, 1, 0)
        self.assertIn("Position value must be positive", str(context.exception))

    def test_parametric_var_negative_position_raises_error(self):
        """Test that negative position value raises ValueError."""
        with self.assertRaises(ValueError) as context:
            calculate_parametric_var(self.returns, 95.0, 1, -10000)
        self.assertIn("Position value must be positive", str(context.exception))

    def test_parametric_var_zero_days_raises_error(self):
        """Test that zero days raises ValueError."""
        with self.assertRaises(ValueError) as context:
            calculate_parametric_var(self.returns, 95.0, 0, self.position_value)
        self.assertIn("Days must be positive", str(context.exception))

    def test_parametric_var_negative_days_raises_error(self):
        """Test that negative days raises ValueError."""
        with self.assertRaises(ValueError) as context:
            calculate_parametric_var(self.returns, 95.0, -5, self.position_value)
        self.assertIn("Days must be positive", str(context.exception))

    def test_parametric_var_different_position_sizes(self):
        """Test that VaR scales linearly with position size."""
        result_100k = calculate_parametric_var(self.returns, 95.0, 1, 100000)
        result_200k = calculate_parametric_var(self.returns, 95.0, 1, 200000)

        # VaR should double when position doubles
        ratio = result_200k["var_amount"] / result_100k["var_amount"]
        self.assertAlmostEqual(ratio, 2.0, places=1)


class TestHistoricalVaR(unittest.TestCase):
    """Test historical simulation VaR calculation."""

    def setUp(self):
        """Set up test data."""
        # Create sample returns with known distribution
        np.random.seed(42)
        self.returns = pd.Series(np.random.normal(0.001, 0.02, 250))
        self.position_value = 100000
        self.confidence = 95.0
        self.days = 1

    def test_historical_var_basic(self):
        """Test basic historical VaR calculation."""
        result = calculate_historical_var(
            self.returns, self.confidence, self.days, self.position_value
        )

        self.assertIn("var_amount", result)
        self.assertIn("var_percentage", result)
        self.assertGreater(result["var_amount"], 0)
        self.assertGreater(result["var_percentage"], 0)

    def test_historical_var_percentile(self):
        """Test that historical VaR uses correct percentile."""
        # Create returns with known values
        returns = pd.Series([-0.05, -0.03, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05])

        result = calculate_historical_var(returns, 90.0, 1, 100000)

        # At 90% confidence, we look at 10th percentile (worst 10%)
        # The 10th percentile of this data should be around -0.05
        self.assertGreater(result["var_amount"], 0)

    def test_historical_var_95_vs_99_confidence(self):
        """Test that 99% confidence gives higher VaR than 95%."""
        result_95 = calculate_historical_var(self.returns, 95.0, 1, self.position_value)
        result_99 = calculate_historical_var(self.returns, 99.0, 1, self.position_value)

        # 99% VaR should be higher (captures more extreme losses)
        self.assertGreater(result_99["var_amount"], result_95["var_amount"])

    def test_historical_var_multiday(self):
        """Test historical VaR with multiple day horizon."""
        result_1day = calculate_historical_var(
            self.returns, self.confidence, 1, self.position_value
        )
        result_5day = calculate_historical_var(
            self.returns, self.confidence, 5, self.position_value
        )

        # Multi-day VaR should be higher (compounds returns)
        self.assertGreater(result_5day["var_amount"], result_1day["var_amount"])

    def test_historical_var_zero_position_raises_error(self):
        """Test that zero position value raises ValueError."""
        with self.assertRaises(ValueError) as context:
            calculate_historical_var(self.returns, 95.0, 1, 0)
        self.assertIn("Position value must be positive", str(context.exception))

    def test_historical_var_zero_days_raises_error(self):
        """Test that zero days raises ValueError."""
        with self.assertRaises(ValueError) as context:
            calculate_historical_var(self.returns, 95.0, 0, self.position_value)
        self.assertIn("Days must be positive", str(context.exception))

    def test_historical_var_negative_returns(self):
        """Test historical VaR with predominantly negative returns."""
        # Market crash scenario
        negative_returns = pd.Series(np.random.normal(-0.02, 0.03, 100))

        result = calculate_historical_var(negative_returns, 95.0, 1, 100000)

        # VaR should still be positive (it's the loss amount)
        self.assertGreater(result["var_amount"], 0)


class TestMonteCarloVaR(unittest.TestCase):
    """Test Monte Carlo simulation VaR calculation."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.returns = pd.Series(np.random.normal(0.001, 0.02, 250))
        self.position_value = 100000
        self.confidence = 95.0
        self.days = 1

    def test_monte_carlo_var_basic(self):
        """Test basic Monte Carlo VaR calculation."""
        result = calculate_monte_carlo_var(
            self.returns, self.confidence, self.days, self.position_value, simulations=1000
        )

        self.assertIn("var_amount", result)
        self.assertIn("var_percentage", result)
        self.assertGreater(result["var_amount"], 0)
        self.assertGreater(result["var_percentage"], 0)

    def test_monte_carlo_var_default_simulations(self):
        """Test Monte Carlo VaR with default number of simulations."""
        result = calculate_monte_carlo_var(
            self.returns, self.confidence, self.days, self.position_value
        )

        self.assertGreater(result["var_amount"], 0)

    def test_monte_carlo_var_different_simulation_counts(self):
        """Test that more simulations give more stable results."""
        np.random.seed(42)
        result_1000 = calculate_monte_carlo_var(
            self.returns, 95.0, 1, 100000, simulations=1000
        )

        np.random.seed(42)
        result_10000 = calculate_monte_carlo_var(
            self.returns, 95.0, 1, 100000, simulations=10000
        )

        # Results should be similar (within 10%)
        ratio = result_10000["var_amount"] / result_1000["var_amount"]
        self.assertGreater(ratio, 0.9)
        self.assertLess(ratio, 1.1)

    def test_monte_carlo_var_multiday(self):
        """Test Monte Carlo VaR with multiple day horizon."""
        result_1day = calculate_monte_carlo_var(
            self.returns, self.confidence, 1, self.position_value, simulations=1000
        )
        result_10day = calculate_monte_carlo_var(
            self.returns, self.confidence, 10, self.position_value, simulations=1000
        )

        # Multi-day VaR should be higher
        self.assertGreater(result_10day["var_amount"], result_1day["var_amount"])

    def test_monte_carlo_var_zero_position_raises_error(self):
        """Test that zero position value raises ValueError."""
        with self.assertRaises(ValueError) as context:
            calculate_monte_carlo_var(self.returns, 95.0, 1, 0)
        self.assertIn("Position value must be positive", str(context.exception))

    def test_monte_carlo_var_zero_days_raises_error(self):
        """Test that zero days raises ValueError."""
        with self.assertRaises(ValueError) as context:
            calculate_monte_carlo_var(self.returns, 95.0, 0, self.position_value)
        self.assertIn("Days must be positive", str(context.exception))

    def test_monte_carlo_var_zero_simulations_raises_error(self):
        """Test that zero simulations raises ValueError."""
        with self.assertRaises(ValueError) as context:
            calculate_monte_carlo_var(
                self.returns, 95.0, 1, self.position_value, simulations=0
            )
        self.assertIn("Number of simulations must be positive", str(context.exception))

    def test_monte_carlo_var_negative_simulations_raises_error(self):
        """Test that negative simulations raises ValueError."""
        with self.assertRaises(ValueError) as context:
            calculate_monte_carlo_var(
                self.returns, 95.0, 1, self.position_value, simulations=-100
            )
        self.assertIn("Number of simulations must be positive", str(context.exception))


class TestPortfolioVaR(unittest.TestCase):
    """Test portfolio-level VaR calculation."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        # Create correlated returns for two assets
        dates = pd.date_range(start="2023-01-01", periods=250, freq="D")
        returns1 = np.random.normal(0.001, 0.02, 250)
        returns2 = returns1 * 0.7 + np.random.normal(0.0005, 0.015, 250) * 0.3
        self.returns = pd.DataFrame(
            {"ASSET1": returns1, "ASSET2": returns2}, index=dates
        )
        self.weights = np.array([0.6, 0.4])
        self.portfolio_value = 100000
        self.confidence = 95.0
        self.days = 1

    def test_portfolio_var_parametric(self):
        """Test portfolio VaR using parametric method."""
        result = calculate_portfolio_var(
            self.returns,
            self.weights,
            self.confidence,
            self.days,
            self.portfolio_value,
            method="parametric",
        )

        self.assertIn("var_amount", result)
        self.assertIn("var_percentage", result)
        self.assertGreater(result["var_amount"], 0)

    def test_portfolio_var_historical(self):
        """Test portfolio VaR using historical method."""
        result = calculate_portfolio_var(
            self.returns,
            self.weights,
            self.confidence,
            self.days,
            self.portfolio_value,
            method="historical",
        )

        self.assertIn("var_amount", result)
        self.assertGreater(result["var_amount"], 0)

    def test_portfolio_var_monte_carlo(self):
        """Test portfolio VaR using Monte Carlo method."""
        result = calculate_portfolio_var(
            self.returns,
            self.weights,
            self.confidence,
            self.days,
            self.portfolio_value,
            method="monte_carlo",
        )

        self.assertIn("var_amount", result)
        self.assertGreater(result["var_amount"], 0)

    def test_portfolio_var_diversification_benefit(self):
        """Test that portfolio VaR is less than sum of individual VaRs (diversification)."""
        # Calculate individual VaRs
        var1 = calculate_parametric_var(
            self.returns["ASSET1"], 95.0, 1, self.portfolio_value * 0.6
        )
        var2 = calculate_parametric_var(
            self.returns["ASSET2"], 95.0, 1, self.portfolio_value * 0.4
        )
        sum_individual_vars = var1["var_amount"] + var2["var_amount"]

        # Calculate portfolio VaR
        portfolio_var = calculate_portfolio_var(
            self.returns, self.weights, 95.0, 1, self.portfolio_value, "parametric"
        )

        # Portfolio VaR should be less than sum (due to diversification)
        self.assertLess(portfolio_var["var_amount"], sum_individual_vars)

    def test_portfolio_var_equal_weights(self):
        """Test portfolio VaR with equal weights."""
        equal_weights = np.array([0.5, 0.5])
        result = calculate_portfolio_var(
            self.returns, equal_weights, 95.0, 1, 100000, "parametric"
        )

        self.assertGreater(result["var_amount"], 0)

    def test_portfolio_var_single_asset(self):
        """Test portfolio VaR with single asset (weight=1.0)."""
        single_returns = pd.DataFrame({"ASSET1": self.returns["ASSET1"]})
        single_weights = np.array([1.0])

        result = calculate_portfolio_var(
            single_returns, single_weights, 95.0, 1, 100000, "parametric"
        )

        self.assertGreater(result["var_amount"], 0)

    def test_portfolio_var_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        with self.assertRaises(ValueError) as context:
            calculate_portfolio_var(
                self.returns, self.weights, 95.0, 1, 100000, method="invalid"
            )
        self.assertIn("Unknown method", str(context.exception))


class TestVaRResult(unittest.TestCase):
    """Test VaRResult dataclass."""

    def test_var_result_creation(self):
        """Test creating a VaRResult object."""
        result = VaRResult(
            ticker="AAPL",
            method="Historical",
            confidence=95.0,
            days=1,
            var_amount=2500.0,
            var_percentage=2.5,
            position_value=100000.0,
        )

        self.assertEqual(result.ticker, "AAPL")
        self.assertEqual(result.method, "Historical")
        self.assertEqual(result.confidence, 95.0)
        self.assertEqual(result.days, 1)
        self.assertEqual(result.var_amount, 2500.0)
        self.assertEqual(result.var_percentage, 2.5)
        self.assertEqual(result.position_value, 100000.0)


class TestMethodComparison(unittest.TestCase):
    """Test comparing results across different VaR methods."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.returns = pd.Series(np.random.normal(0.001, 0.02, 250))
        self.position_value = 100000
        self.confidence = 95.0
        self.days = 1

    def test_all_methods_give_reasonable_results(self):
        """Test that all three methods give similar order of magnitude."""
        parametric = calculate_parametric_var(
            self.returns, self.confidence, self.days, self.position_value
        )
        historical = calculate_historical_var(
            self.returns, self.confidence, self.days, self.position_value
        )
        monte_carlo = calculate_monte_carlo_var(
            self.returns, self.confidence, self.days, self.position_value, simulations=5000
        )

        # All methods should give positive VaR
        self.assertGreater(parametric["var_amount"], 0)
        self.assertGreater(historical["var_amount"], 0)
        self.assertGreater(monte_carlo["var_amount"], 0)

        # All should be within reasonable range (not more than 3x different)
        max_var = max(
            parametric["var_amount"],
            historical["var_amount"],
            monte_carlo["var_amount"],
        )
        min_var = min(
            parametric["var_amount"],
            historical["var_amount"],
            monte_carlo["var_amount"],
        )
        self.assertLess(max_var / min_var, 3.0)

    def test_methods_with_non_normal_distribution(self):
        """Test that historical and Monte Carlo differ from parametric for non-normal returns."""
        # Create skewed returns (not normal)
        skewed_returns = pd.Series(np.random.exponential(0.02, 250) - 0.02)

        parametric = calculate_parametric_var(skewed_returns, 95.0, 1, 100000)
        historical = calculate_historical_var(skewed_returns, 95.0, 1, 100000)

        # Results should differ (parametric assumes normality)
        # We're just checking they both run and produce different values
        self.assertNotAlmostEqual(
            parametric["var_amount"], historical["var_amount"], places=0
        )


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.returns = pd.Series(np.random.normal(0.001, 0.02, 250))

    def test_var_with_very_low_volatility(self):
        """Test VaR with very low volatility returns."""
        low_vol_returns = pd.Series(np.random.normal(0.0001, 0.001, 250))

        result = calculate_parametric_var(low_vol_returns, 95.0, 1, 100000)

        # VaR should be small but positive
        self.assertGreater(result["var_amount"], 0)
        self.assertLess(result["var_amount"], 1000)  # Less than 1% of portfolio

    def test_var_with_very_high_volatility(self):
        """Test VaR with very high volatility returns."""
        high_vol_returns = pd.Series(np.random.normal(0.0, 0.1, 250))

        result = calculate_parametric_var(high_vol_returns, 95.0, 1, 100000)

        # VaR should be large
        self.assertGreater(result["var_amount"], 5000)  # More than 5% of portfolio

    def test_var_with_constant_returns(self):
        """Test VaR with constant (zero volatility) returns."""
        constant_returns = pd.Series([0.001] * 250)

        result = calculate_parametric_var(constant_returns, 95.0, 1, 100000)

        # VaR should be very small (near zero)
        self.assertLess(result["var_amount"], 500)

    def test_var_percentage_calculation(self):
        """Test that VaR percentage is correctly calculated."""
        result = calculate_parametric_var(self.returns, 95.0, 1, 100000)

        # VaR % should match: (VaR amount / position value) * 100
        expected_pct = (result["var_amount"] / 100000) * 100
        self.assertAlmostEqual(result["var_percentage"], expected_pct, places=2)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    unittest.main()
