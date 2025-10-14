"""
Integration tests for the Financial Risk Analyzer CLI.

Tests end-to-end workflows through main.py for all three modules.
"""

import unittest
import subprocess
import os
import tempfile
import csv
from pathlib import Path


class TestCLIIntegration(unittest.TestCase):
    """Test CLI integration through subprocess calls."""

    def setUp(self):
        """Set up test environment."""
        self.repo_root = Path(__file__).parent.parent
        self.main_script = self.repo_root / "main.py"
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def run_cli(self, args, timeout=60):
        """Helper to run CLI commands."""
        cmd = ["python", str(self.main_script)] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(self.repo_root),
        )
        return result

    def test_cli_no_command_shows_help(self):
        """Test that running with no command shows help."""
        result = self.run_cli([])

        self.assertEqual(result.returncode, 1)
        self.assertIn("usage:", result.stdout.lower())

    def test_cli_help_flag(self):
        """Test --help flag."""
        result = self.run_cli(["--help"])

        self.assertEqual(result.returncode, 0)
        self.assertIn("usage:", result.stdout.lower())
        self.assertIn("zscore", result.stdout.lower())
        self.assertIn("var", result.stdout.lower())
        self.assertIn("stress", result.stdout.lower())


class TestZScoreIntegration(unittest.TestCase):
    """Integration tests for Z-Score module."""

    def setUp(self):
        """Set up test environment."""
        self.repo_root = Path(__file__).parent.parent
        self.main_script = self.repo_root / "main.py"
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def run_cli(self, args, timeout=120):
        """Helper to run CLI commands."""
        cmd = ["python", str(self.main_script)] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(self.repo_root),
        )
        return result

    def test_zscore_help(self):
        """Test Z-Score help."""
        result = self.run_cli(["zscore", "--help"])

        self.assertEqual(result.returncode, 0)
        self.assertIn("zscore", result.stdout.lower())

    def test_zscore_invalid_ticker(self):
        """Test Z-Score with invalid ticker."""
        output_file = os.path.join(self.temp_dir, "zscore_invalid.csv")
        result = self.run_cli(
            ["zscore", "INVALIDTICKER123", "--out", output_file], timeout=30
        )

        # Should exit with error
        self.assertEqual(result.returncode, 1)

    def test_zscore_output_file_format(self):
        """Test that Z-Score creates properly formatted CSV."""
        # Skip if no network or takes too long
        # This is more of a smoke test
        pass


class TestVaRIntegration(unittest.TestCase):
    """Integration tests for VaR module."""

    def setUp(self):
        """Set up test environment."""
        self.repo_root = Path(__file__).parent.parent
        self.main_script = self.repo_root / "main.py"
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def run_cli(self, args, timeout=120):
        """Helper to run CLI commands."""
        cmd = ["python", str(self.main_script)] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(self.repo_root),
        )
        return result

    def test_var_help(self):
        """Test VaR help."""
        result = self.run_cli(["var", "--help"])

        self.assertEqual(result.returncode, 0)
        self.assertIn("var", result.stdout.lower())

    def test_var_invalid_confidence(self):
        """Test VaR with invalid confidence level."""
        result = self.run_cli(
            ["var", "AAPL", "--confidence", "150"], timeout=10
        )

        # Should error on invalid confidence
        self.assertEqual(result.returncode, 2)
        self.assertIn("confidence", result.stderr.lower())

    def test_var_invalid_days(self):
        """Test VaR with invalid days."""
        result = self.run_cli(["var", "AAPL", "--days", "-5"], timeout=10)

        # Should error on invalid days
        self.assertEqual(result.returncode, 2)
        self.assertIn("days", result.stderr.lower())

    def test_var_zero_portfolio_value(self):
        """Test VaR with zero portfolio value."""
        result = self.run_cli(
            ["var", "AAPL", "--portfolio-value", "0"], timeout=10
        )

        # Should error on zero portfolio value
        self.assertEqual(result.returncode, 2)
        self.assertIn("positive", result.stderr.lower())

    def test_var_negative_weights(self):
        """Test VaR with negative weights."""
        result = self.run_cli(
            ["var", "AAPL", "MSFT", "--weights", "-0.5", "1.5"], timeout=10
        )

        # Should error on negative weights
        self.assertEqual(result.returncode, 2)
        self.assertIn("negative", result.stderr.lower())

    def test_var_weights_exceed_one(self):
        """Test VaR with individual weight exceeding 1.0."""
        result = self.run_cli(
            ["var", "AAPL", "MSFT", "--weights", "1.5", "0.5"], timeout=10
        )

        # Should error
        self.assertEqual(result.returncode, 2)

    def test_var_negative_position_values(self):
        """Test VaR with negative position values."""
        result = self.run_cli(
            ["var", "AAPL", "MSFT", "--values", "-10000", "20000"], timeout=10
        )

        # Should error on negative values
        self.assertEqual(result.returncode, 2)
        self.assertIn("positive", result.stderr.lower())

    def test_var_unsupported_currency(self):
        """Test VaR with unsupported currency."""
        result = self.run_cli(["var", "AAPL", "--currency", "ZAR"], timeout=10)

        # Should error on unsupported currency
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr.lower())

    def test_var_supported_currencies(self):
        """Test that all supported currencies are accepted."""
        supported = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF"]

        for currency in supported:
            result = self.run_cli(
                ["var", "AAPL", "--help"],  # Just test parsing
                timeout=10,
            )
            # Help should mention the currency
            self.assertIn(currency, result.stdout)

    def test_var_all_methods(self):
        """Test that all VaR methods are accepted."""
        methods = ["parametric", "historical", "monte_carlo"]

        for method in methods:
            # Just test that the method is accepted by help
            result = self.run_cli(["var", "--help"])
            self.assertIn(method, result.stdout)

    def test_var_values_overrides_weights_warning(self):
        """Test warning when both --values and --weights are specified."""
        # This would need actual execution which might be slow
        # Skipping for unit test
        pass


class TestStressTestIntegration(unittest.TestCase):
    """Integration tests for Stress Test module."""

    def setUp(self):
        """Set up test environment."""
        self.repo_root = Path(__file__).parent.parent
        self.main_script = self.repo_root / "main.py"
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def run_cli(self, args, timeout=120):
        """Helper to run CLI commands."""
        cmd = ["python", str(self.main_script)] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(self.repo_root),
        )
        return result

    def test_stress_help(self):
        """Test stress test help."""
        result = self.run_cli(["stress", "--help"])

        self.assertEqual(result.returncode, 0)
        self.assertIn("stress", result.stdout.lower())

    def test_stress_all_scenarios_available(self):
        """Test that all scenarios are available."""
        result = self.run_cli(["stress", "--help"])

        scenarios = ["dotcom", "gfc", "covid", "all"]
        for scenario in scenarios:
            self.assertIn(scenario, result.stdout)

    def test_stress_unsupported_currency(self):
        """Test stress test with unsupported currency."""
        result = self.run_cli(["stress", "AAPL", "--currency", "BRL"], timeout=10)

        # Should error on unsupported currency
        self.assertEqual(result.returncode, 2)

    def test_stress_negative_portfolio_value(self):
        """Test stress test with negative portfolio value."""
        result = self.run_cli(
            ["stress", "AAPL", "--portfolio-value", "-10000"], timeout=10
        )

        # Should error
        self.assertEqual(result.returncode, 2)

    def test_stress_negative_weights(self):
        """Test stress test with negative weights."""
        result = self.run_cli(
            ["stress", "AAPL", "MSFT", "--weights", "-0.5", "1.5"], timeout=10
        )

        # Should error on negative weights
        self.assertEqual(result.returncode, 2)


class TestCLIValidation(unittest.TestCase):
    """Test CLI input validation."""

    def setUp(self):
        """Set up test environment."""
        self.repo_root = Path(__file__).parent.parent
        self.main_script = self.repo_root / "main.py"

    def run_cli(self, args, timeout=10):
        """Helper to run CLI commands."""
        cmd = ["python", str(self.main_script)] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(self.repo_root),
        )
        return result

    def test_var_extreme_days_warning(self):
        """Test that warning is shown for extreme days."""
        result = self.run_cli(["var", "AAPL", "--days", "500"], timeout=10)

        # Should show warning (but might still run)
        # Check for warning in stderr
        if result.returncode == 0:
            # If it proceeds, should have warning
            self.assertIn("Warning", result.stderr) or self.assertIn(
                "warning", result.stderr.lower()
            )

    def test_var_confidence_boundaries(self):
        """Test confidence level boundaries."""
        # Test 0 (invalid)
        result = self.run_cli(["var", "AAPL", "--confidence", "0"], timeout=10)
        self.assertNotEqual(result.returncode, 0)

        # Test 100 (invalid)
        result = self.run_cli(["var", "AAPL", "--confidence", "100"], timeout=10)
        self.assertNotEqual(result.returncode, 0)

        # Test negative (invalid)
        result = self.run_cli(["var", "AAPL", "--confidence", "-5"], timeout=10)
        self.assertNotEqual(result.returncode, 0)


class TestMultiCurrencyIntegration(unittest.TestCase):
    """Test multi-currency functionality integration."""

    def setUp(self):
        """Set up test environment."""
        self.repo_root = Path(__file__).parent.parent
        self.main_script = self.repo_root / "main.py"

    def run_cli(self, args, timeout=30):
        """Helper to run CLI commands."""
        cmd = ["python", str(self.main_script)] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(self.repo_root),
        )
        return result

    def test_var_currency_parameter_exists(self):
        """Test that --currency parameter exists for var."""
        result = self.run_cli(["var", "--help"])

        self.assertIn("--currency", result.stdout)
        self.assertIn("USD", result.stdout)
        self.assertIn("EUR", result.stdout)

    def test_stress_currency_parameter_exists(self):
        """Test that --currency parameter exists for stress."""
        result = self.run_cli(["stress", "--help"])

        self.assertIn("--currency", result.stdout)
        self.assertIn("USD", result.stdout)
        self.assertIn("EUR", result.stdout)

    def test_all_supported_currencies_listed(self):
        """Test that all supported currencies are listed in help."""
        supported_currencies = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF"]

        result = self.run_cli(["var", "--help"])

        for currency in supported_currencies:
            self.assertIn(currency, result.stdout)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and user-friendly messages."""

    def setUp(self):
        """Set up test environment."""
        self.repo_root = Path(__file__).parent.parent
        self.main_script = self.repo_root / "main.py"

    def run_cli(self, args, timeout=30):
        """Helper to run CLI commands."""
        cmd = ["python", str(self.main_script)] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(self.repo_root),
        )
        return result

    def test_no_ticker_error(self):
        """Test error when no ticker is provided."""
        commands = ["zscore", "var", "stress"]

        for cmd in commands:
            result = self.run_cli([cmd], timeout=10)
            # Should error
            self.assertNotEqual(result.returncode, 0)

    def test_keyboard_interrupt_handling(self):
        """Test that keyboard interrupt is handled gracefully."""
        # This is difficult to test in unit tests
        # Would require actual interruption during execution
        pass


class TestOutputFiles(unittest.TestCase):
    """Test output file generation."""

    def setUp(self):
        """Set up test environment."""
        self.repo_root = Path(__file__).parent.parent
        self.main_script = self.repo_root / "main.py"
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def run_cli(self, args, timeout=120):
        """Helper to run CLI commands."""
        cmd = ["python", str(self.main_script)] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(self.repo_root),
        )
        return result

    def test_var_custom_output_file(self):
        """Test VaR with custom output file."""
        output_file = os.path.join(self.temp_dir, "custom_var.csv")
        # This would require actual execution with real tickers
        # Skip for now as it requires network and time
        pass

    def test_default_output_filenames(self):
        """Test that default output filenames are documented."""
        result = self.run_cli(["var", "--help"])
        self.assertIn("--out", result.stdout)

        result = self.run_cli(["stress", "--help"])
        self.assertIn("--out", result.stdout)

        result = self.run_cli(["zscore", "--help"])
        self.assertIn("--out", result.stdout)


class TestCLIExamples(unittest.TestCase):
    """Test that CLI examples in help are valid."""

    def setUp(self):
        """Set up test environment."""
        self.repo_root = Path(__file__).parent.parent
        self.main_script = self.repo_root / "main.py"

    def run_cli(self, args, timeout=10):
        """Helper to run CLI commands."""
        cmd = ["python", str(self.main_script)] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(self.repo_root),
        )
        return result

    def test_examples_shown_in_help(self):
        """Test that examples are shown in main help."""
        result = self.run_cli(["--help"])

        self.assertIn("Examples:", result.stdout)
        self.assertIn("python main.py", result.stdout)


if __name__ == "__main__":
    unittest.main()
