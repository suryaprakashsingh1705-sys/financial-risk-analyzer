"""
Configuration and Constants for Financial Risk Analyzer

This module contains all configuration settings, thresholds, and magic numbers
used throughout the application.
"""

# =============================================================================
# ALTMAN Z-SCORE CONSTANTS
# =============================================================================

# Original Z-Score Coefficients (Manufacturing)
ZSCORE_ORIGINAL_WC_COEF = 1.2
ZSCORE_ORIGINAL_RE_COEF = 1.4
ZSCORE_ORIGINAL_EBIT_COEF = 3.3
ZSCORE_ORIGINAL_MVE_COEF = 0.6
ZSCORE_ORIGINAL_SALES_COEF = 1.0

# Modified Z''-Score Coefficients (Non-Manufacturing)
ZSCORE_MODIFIED_WC_COEF = 6.56
ZSCORE_MODIFIED_RE_COEF = 3.26
ZSCORE_MODIFIED_EBIT_COEF = 6.72
ZSCORE_MODIFIED_BVE_COEF = 1.05

# Z-Score Risk Zone Thresholds - Original Model
ZSCORE_ORIGINAL_DISTRESS_THRESHOLD = 1.81
ZSCORE_ORIGINAL_GREY_THRESHOLD = 2.99

# Z-Score Risk Zone Thresholds - Modified Model
ZSCORE_MODIFIED_DISTRESS_THRESHOLD = 1.1
ZSCORE_MODIFIED_GREY_THRESHOLD = 2.6


# =============================================================================
# VALUE AT RISK (VAR) CONSTANTS
# =============================================================================

# Default VaR Settings
VAR_DEFAULT_CONFIDENCE = 95.0
VAR_DEFAULT_DAYS = 1
VAR_DEFAULT_PORTFOLIO_VALUE = 100000.0

# Monte Carlo Simulation Settings
MONTE_CARLO_DEFAULT_SIMULATIONS = 10000

# Historical Data Period
VAR_HISTORICAL_DATA_PERIOD = "2y"

# Valid Confidence Levels
VAR_MIN_CONFIDENCE = 0.0
VAR_MAX_CONFIDENCE = 100.0

# Valid Common Confidence Levels
VAR_COMMON_CONFIDENCE_LEVELS = [90.0, 95.0, 99.0, 99.9]


# =============================================================================
# STRESS TEST CONSTANTS
# =============================================================================

# Default Stress Test Settings
STRESS_DEFAULT_PORTFOLIO_VALUE = 100000.0

# Crisis Scenario Definitions
CRISIS_DOTCOM_START = "2000-03-01"
CRISIS_DOTCOM_END = "2002-10-01"
CRISIS_DOTCOM_MARKET_DROP = -49.1  # NASDAQ drop percentage

CRISIS_GFC_START = "2007-10-01"
CRISIS_GFC_END = "2009-03-01"
CRISIS_GFC_MARKET_DROP = -56.8  # S&P 500 drop percentage

CRISIS_COVID_START = "2020-02-01"
CRISIS_COVID_END = "2020-03-31"
CRISIS_COVID_MARKET_DROP = -33.9  # S&P 500 drop percentage


# =============================================================================
# COMPANY CLASSIFICATION CONSTANTS
# =============================================================================

# PPE (Property, Plant & Equipment) Ratio Thresholds
# Used to distinguish fab-owning semiconductor manufacturers from fabless designers
PPE_RATIO_HIGH_THRESHOLD = 0.30  # >30% indicates manufacturing (owns fabs)
PPE_RATIO_LOW_THRESHOLD = 0.10  # <10% indicates fabless (design only)

# Classification Score Boosts
FAB_SCORE_BOOST_HIGH = 3  # Strong indicator of manufacturing
FAB_SCORE_BOOST_MEDIUM = 2
FABLESS_SCORE_BOOST = 2  # Strong indicator of fabless


# =============================================================================
# DATA FETCHING CONSTANTS
# =============================================================================

# yfinance Settings
YFINANCE_PROGRESS_BAR = False  # Disable progress bar for downloads
YFINANCE_AUTO_ADJUST = True  # Auto-adjust prices for splits/dividends


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

# Currency Format
CURRENCY_DECIMAL_PLACES = 2

# Percentage Format
PERCENTAGE_DECIMAL_PLACES = 2

# Default Output Filenames
DEFAULT_ZSCORE_OUTPUT = "zscore_results.csv"
DEFAULT_VAR_OUTPUT = "var_results.csv"
DEFAULT_STRESS_OUTPUT = "stress_test_results.csv"


# =============================================================================
# CURRENCY CONSTANTS
# =============================================================================

# Supported currencies for portfolio analysis
# These are currencies with reliable historical exchange rate data on Yahoo Finance
SUPPORTED_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF"]

# Default portfolio currency
DEFAULT_CURRENCY = "USD"


# =============================================================================
# VALIDATION CONSTANTS
# =============================================================================

# Minimum valid values
MIN_PORTFOLIO_VALUE = 0.01
MIN_POSITION_VALUE = 0.01
MIN_DAYS = 1
MAX_DAYS = 365

# Weight validation
WEIGHT_SUM_TOLERANCE = 0.0001  # Tolerance for floating point comparison
