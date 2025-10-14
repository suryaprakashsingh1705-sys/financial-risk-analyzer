"""
Comprehensive tests for the Altman Z-Score module.

Tests all Z-Score calculation logic, company classification, and risk zone determination.
"""

import unittest
import sys
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from io import StringIO

from modules.zscore import (
    calculate_z_score_original,
    calculate_z_score_modified,
    get_zone,
    analyze_ticker,
    ZScoreResult,
)
from config import (
    ZSCORE_ORIGINAL_DISTRESS_THRESHOLD,
    ZSCORE_ORIGINAL_GREY_THRESHOLD,
    ZSCORE_MODIFIED_DISTRESS_THRESHOLD,
    ZSCORE_MODIFIED_GREY_THRESHOLD,
)


class TestZScoreCalculations(unittest.TestCase):
    """Test Z-Score calculation formulas."""

    def test_calculate_original_zscore(self):
        """Test original Z-Score calculation for manufacturing companies."""
        # Sample data for a healthy manufacturing company
        wc = 50000  # Working capital
        re = 100000  # Retained earnings
        ebit = 30000  # EBIT
        ta = 200000  # Total assets
        mve = 300000  # Market value of equity
        tl = 80000  # Total liabilities
        sales = 250000  # Sales

        z_score = calculate_z_score_original(wc, re, ebit, ta, mve, tl, sales)

        # Z = 1.2*(WC/TA) + 1.4*(RE/TA) + 3.3*(EBIT/TA) + 0.6*(MVE/TL) + 1.0*(Sales/TA)
        expected = (
            1.2 * (50000 / 200000)
            + 1.4 * (100000 / 200000)
            + 3.3 * (30000 / 200000)
            + 0.6 * (300000 / 80000)
            + 1.0 * (250000 / 200000)
        )

        self.assertAlmostEqual(z_score, expected, places=2)
        self.assertGreater(z_score, 0)

    def test_calculate_modified_zscore(self):
        """Test modified Z-Score calculation for non-manufacturing companies."""
        # Sample data for a healthy service company
        wc = 30000
        re = 80000
        ebit = 25000
        ta = 150000
        mve = 200000
        tl = 50000

        z_score = calculate_z_score_modified(wc, re, ebit, ta, mve, tl)

        # Z'' = 6.56*(WC/TA) + 3.26*(RE/TA) + 6.72*(EBIT/TA) + 1.05*(BVE/TL)
        bve = ta - tl
        expected = (
            6.56 * (30000 / 150000)
            + 3.26 * (80000 / 150000)
            + 6.72 * (25000 / 150000)
            + 1.05 * (bve / 50000)
        )

        self.assertAlmostEqual(z_score, expected, places=2)
        self.assertGreater(z_score, 0)

    def test_original_zscore_negative_values(self):
        """Test original Z-Score with negative working capital."""
        wc = -10000  # Negative working capital
        re = 50000
        ebit = 10000
        ta = 100000
        mve = 80000
        tl = 40000
        sales = 120000

        z_score = calculate_z_score_original(wc, re, ebit, ta, mve, tl, sales)

        # Should still calculate, but score will be lower
        self.assertIsInstance(z_score, float)

    def test_modified_zscore_negative_values(self):
        """Test modified Z-Score with negative earnings."""
        wc = 20000
        re = -5000  # Negative retained earnings
        ebit = -2000  # Negative EBIT
        ta = 80000
        mve = 60000
        tl = 30000

        z_score = calculate_z_score_modified(wc, re, ebit, ta, mve, tl)

        # Should still calculate
        self.assertIsInstance(z_score, float)

    def test_original_zscore_zero_assets_raises_error(self):
        """Test that zero total assets raises ValueError."""
        with self.assertRaises(ValueError) as context:
            calculate_z_score_original(10000, 10000, 5000, 0, 50000, 20000, 30000)
        self.assertIn("Total assets must be positive", str(context.exception))

    def test_original_zscore_negative_assets_raises_error(self):
        """Test that negative total assets raises ValueError."""
        with self.assertRaises(ValueError) as context:
            calculate_z_score_original(10000, 10000, 5000, -100000, 50000, 20000, 30000)
        self.assertIn("Total assets must be positive", str(context.exception))

    def test_original_zscore_zero_liabilities_raises_error(self):
        """Test that zero total liabilities raises ValueError."""
        with self.assertRaises(ValueError) as context:
            calculate_z_score_original(10000, 10000, 5000, 100000, 50000, 0, 30000)
        self.assertIn("Total liabilities must be positive", str(context.exception))

    def test_modified_zscore_zero_assets_raises_error(self):
        """Test that zero total assets raises ValueError for modified."""
        with self.assertRaises(ValueError) as context:
            calculate_z_score_modified(10000, 10000, 5000, 0, 50000, 20000)
        self.assertIn("Total assets must be positive", str(context.exception))

    def test_modified_zscore_zero_liabilities_raises_error(self):
        """Test that zero total liabilities raises ValueError for modified."""
        with self.assertRaises(ValueError) as context:
            calculate_z_score_modified(10000, 10000, 5000, 100000, 50000, 0)
        self.assertIn("Total liabilities must be positive", str(context.exception))


class TestRiskZones(unittest.TestCase):
    """Test risk zone classification."""

    def test_original_distress_zone(self):
        """Test distress zone for original model."""
        # Score below 1.81
        z_score = 1.5
        zone = get_zone(z_score, "Original")
        self.assertEqual(zone, "Distress")

    def test_original_grey_zone(self):
        """Test grey zone for original model."""
        # Score between 1.81 and 2.99
        z_score = 2.3
        zone = get_zone(z_score, "Original")
        self.assertEqual(zone, "Grey Zone")

    def test_original_safe_zone(self):
        """Test safe zone for original model."""
        # Score above 2.99
        z_score = 3.5
        zone = get_zone(z_score, "Original")
        self.assertEqual(zone, "Safe")

    def test_original_boundary_distress_to_grey(self):
        """Test boundary between distress and grey zone."""
        z_score = ZSCORE_ORIGINAL_DISTRESS_THRESHOLD
        zone = get_zone(z_score, "Original")
        self.assertEqual(zone, "Grey Zone")

    def test_original_boundary_grey_to_safe(self):
        """Test boundary between grey and safe zone."""
        z_score = ZSCORE_ORIGINAL_GREY_THRESHOLD
        zone = get_zone(z_score, "Original")
        self.assertEqual(zone, "Safe")

    def test_modified_distress_zone(self):
        """Test distress zone for modified model."""
        # Score below 1.1
        z_score = 0.8
        zone = get_zone(z_score, "Modified")
        self.assertEqual(zone, "Distress")

    def test_modified_grey_zone(self):
        """Test grey zone for modified model."""
        # Score between 1.1 and 2.6
        z_score = 1.9
        zone = get_zone(z_score, "Modified")
        self.assertEqual(zone, "Grey Zone")

    def test_modified_safe_zone(self):
        """Test safe zone for modified model."""
        # Score above 2.6
        z_score = 3.2
        zone = get_zone(z_score, "Modified")
        self.assertEqual(zone, "Safe")

    def test_modified_boundary_distress_to_grey(self):
        """Test boundary between distress and grey zone for modified."""
        z_score = ZSCORE_MODIFIED_DISTRESS_THRESHOLD
        zone = get_zone(z_score, "Modified")
        self.assertEqual(zone, "Grey Zone")

    def test_modified_boundary_grey_to_safe(self):
        """Test boundary between grey and safe zone for modified."""
        z_score = ZSCORE_MODIFIED_GREY_THRESHOLD
        zone = get_zone(z_score, "Modified")
        self.assertEqual(zone, "Safe")

    def test_negative_zscore_distress(self):
        """Test that negative Z-scores are in distress."""
        z_score = -0.5
        zone_original = get_zone(z_score, "Original")
        zone_modified = get_zone(z_score, "Modified")
        self.assertEqual(zone_original, "Distress")
        self.assertEqual(zone_modified, "Distress")


class TestZScoreResult(unittest.TestCase):
    """Test ZScoreResult dataclass."""

    def test_zscore_result_creation(self):
        """Test creating a ZScoreResult object."""
        result = ZScoreResult(
            ticker="AAPL",
            company="Apple Inc.",
            region="US",
            classification="Non-Mfg",
            model="Modified",
            z_score=3.45,
            zone="Safe",
            industry="Technology",
        )

        self.assertEqual(result.ticker, "AAPL")
        self.assertEqual(result.company, "Apple Inc.")
        self.assertEqual(result.region, "US")
        self.assertEqual(result.classification, "Non-Mfg")
        self.assertEqual(result.model, "Modified")
        self.assertEqual(result.z_score, 3.45)
        self.assertEqual(result.zone, "Safe")
        self.assertEqual(result.industry, "Technology")


class TestAnalyzeTicker(unittest.TestCase):
    """Test the analyze_ticker function with mocked data."""

    @patch("modules.zscore.fetch_company_info")
    @patch("modules.zscore.fetch_financial_components")
    @patch("modules.zscore.get_company_profile")
    def test_analyze_manufacturing_company(
        self, mock_profile, mock_components, mock_info
    ):
        """Test analyzing a manufacturing company."""
        # Mock company info
        mock_info.return_value = {
            "longName": "Test Manufacturing Inc.",
            "country": "United States",
            "industry": "Auto Manufacturers",
        }

        # Mock company profile
        mock_profile_obj = MagicMock()
        mock_profile_obj.is_manufacturing = True
        mock_profile_obj.region = "US"
        mock_profile_obj.industry = "Auto Manufacturers"
        mock_profile_obj.classification_reason = "Manufacturing company"
        mock_profile.return_value = mock_profile_obj

        # Mock financial components
        mock_components.return_value = {
            "wc": 50000,
            "re": 100000,
            "ebit": 30000,
            "ta": 200000,
            "mve": 300000,
            "tl": 80000,
            "sales": 250000,
        }

        result = analyze_ticker("TEST", verbose=False, debug=False)

        self.assertIsNotNone(result)
        self.assertEqual(result.ticker, "TEST")
        self.assertEqual(result.model, "Original")
        self.assertEqual(result.classification, "Mfg")
        self.assertGreater(result.z_score, 0)

    @patch("modules.zscore.fetch_company_info")
    @patch("modules.zscore.fetch_financial_components")
    @patch("modules.zscore.get_company_profile")
    def test_analyze_nonmanufacturing_company(
        self, mock_profile, mock_components, mock_info
    ):
        """Test analyzing a non-manufacturing company."""
        # Mock company info
        mock_info.return_value = {
            "longName": "Test Services Inc.",
            "country": "United States",
            "industry": "Software",
        }

        # Mock company profile
        mock_profile_obj = MagicMock()
        mock_profile_obj.is_manufacturing = False
        mock_profile_obj.region = "US"
        mock_profile_obj.industry = "Software"
        mock_profile_obj.classification_reason = "Non-manufacturing company"
        mock_profile.return_value = mock_profile_obj

        # Mock financial components
        mock_components.return_value = {
            "wc": 30000,
            "re": 80000,
            "ebit": 25000,
            "ta": 150000,
            "mve": 200000,
            "tl": 50000,
            "sales": 0,  # Not used in modified
        }

        result = analyze_ticker("TEST", verbose=False, debug=False)

        self.assertIsNotNone(result)
        self.assertEqual(result.ticker, "TEST")
        self.assertEqual(result.model, "Modified")
        self.assertEqual(result.classification, "Non-Mfg")
        self.assertGreater(result.z_score, 0)

    @patch("modules.zscore.fetch_company_info")
    def test_analyze_invalid_ticker(self, mock_info):
        """Test analyzing an invalid ticker."""
        mock_info.return_value = None

        result = analyze_ticker("INVALID", verbose=False, debug=False)

        self.assertIsNone(result)

    @patch("modules.zscore.fetch_company_info")
    @patch("modules.zscore.fetch_financial_components")
    @patch("modules.zscore.get_company_profile")
    def test_analyze_missing_financial_data(
        self, mock_profile, mock_components, mock_info
    ):
        """Test analyzing a company with missing financial data."""
        mock_info.return_value = {"longName": "Test Inc."}

        mock_profile_obj = MagicMock()
        mock_profile_obj.is_manufacturing = True
        mock_profile.return_value = mock_profile_obj

        # Return None for missing data
        mock_components.return_value = None

        result = analyze_ticker("TEST", verbose=False, debug=False)

        self.assertIsNone(result)

    @patch("modules.zscore.fetch_company_info")
    @patch("modules.zscore.fetch_financial_components")
    @patch("modules.zscore.get_company_profile")
    def test_analyze_invalid_financial_data(
        self, mock_profile, mock_components, mock_info
    ):
        """Test analyzing a company with invalid financial data (zero assets)."""
        mock_info.return_value = {"longName": "Test Inc."}

        mock_profile_obj = MagicMock()
        mock_profile_obj.is_manufacturing = True
        mock_profile.return_value = mock_profile_obj

        # Return data with zero total assets (invalid)
        mock_components.return_value = {
            "wc": 50000,
            "re": 100000,
            "ebit": 30000,
            "ta": 0,  # Invalid!
            "mve": 300000,
            "tl": 80000,
            "sales": 250000,
        }

        result = analyze_ticker("TEST", verbose=False, debug=False)

        self.assertIsNone(result)

    @patch("modules.zscore.fetch_company_info")
    @patch("modules.zscore.fetch_financial_components")
    @patch("modules.zscore.get_company_profile")
    def test_analyze_distress_zone_company(
        self, mock_profile, mock_components, mock_info
    ):
        """Test analyzing a company in distress zone."""
        mock_info.return_value = {"longName": "Distressed Inc."}

        mock_profile_obj = MagicMock()
        mock_profile_obj.is_manufacturing = False
        mock_profile_obj.region = "US"
        mock_profile_obj.industry = "Retail"
        mock_profile_obj.classification_reason = "Distressed company"
        mock_profile.return_value = mock_profile_obj

        # Financial data indicating distress
        mock_components.return_value = {
            "wc": -5000,  # Negative working capital
            "re": -10000,  # Negative retained earnings
            "ebit": -3000,  # Negative EBIT
            "ta": 50000,
            "mve": 20000,
            "tl": 45000,
        }

        result = analyze_ticker("DISTRESSED", verbose=False, debug=False)

        self.assertIsNotNone(result)
        self.assertEqual(result.zone, "Distress")

    @patch("modules.zscore.fetch_company_info")
    @patch("modules.zscore.fetch_financial_components")
    @patch("modules.zscore.get_company_profile")
    def test_analyze_with_verbose(self, mock_profile, mock_components, mock_info):
        """Test analyze_ticker with verbose mode."""
        mock_info.return_value = {"longName": "Test Inc."}

        mock_profile_obj = MagicMock()
        mock_profile_obj.is_manufacturing = True
        mock_profile_obj.region = "US"
        mock_profile_obj.industry = "Manufacturing"
        mock_profile_obj.classification_reason = "Manufacturing company"
        mock_profile.return_value = mock_profile_obj

        mock_components.return_value = {
            "wc": 50000,
            "re": 100000,
            "ebit": 30000,
            "ta": 200000,
            "mve": 300000,
            "tl": 80000,
            "sales": 250000,
        }

        # Capture stderr
        captured_output = StringIO()
        with patch("sys.stderr", new=captured_output):
            result = analyze_ticker("TEST", verbose=True, debug=False)

        self.assertIsNotNone(result)
        # Verbose mode should print classification reason
        output = captured_output.getvalue()
        self.assertIn("TEST", output)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_very_high_zscore(self):
        """Test very high Z-scores (super safe companies)."""
        # Extremely healthy financials
        wc = 500000
        re = 1000000
        ebit = 300000
        ta = 1000000
        mve = 2000000
        tl = 100000
        sales = 1500000

        z_score = calculate_z_score_original(wc, re, ebit, ta, mve, tl, sales)

        self.assertGreater(z_score, 10)
        self.assertEqual(get_zone(z_score, "Original"), "Safe")

    def test_very_low_zscore(self):
        """Test very low Z-scores (bankruptcy imminent)."""
        # Terrible financials
        wc = -50000
        re = -100000
        ebit = -30000
        ta = 100000
        mve = 10000
        tl = 95000
        sales = 20000

        z_score = calculate_z_score_original(wc, re, ebit, ta, mve, tl, sales)

        self.assertLess(z_score, 0)
        self.assertEqual(get_zone(z_score, "Original"), "Distress")

    def test_zero_sales_modified_model(self):
        """Test modified model with zero sales (not used in formula)."""
        wc = 30000
        re = 80000
        ebit = 25000
        ta = 150000
        mve = 200000
        tl = 50000

        # Modified model doesn't use sales
        z_score = calculate_z_score_modified(wc, re, ebit, ta, mve, tl)

        self.assertIsInstance(z_score, float)
        self.assertGreater(z_score, 0)


if __name__ == "__main__":
    unittest.main()
