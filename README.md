# Financial Risk Analyzer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

**Comprehensive financial risk assessment toolkit with three powerful modules:**

1. **Altman Z-Score** - Bankruptcy risk prediction
2. **Value at Risk (VaR)** - Portfolio risk quantification  
3. **Stress Testing** - Historical crisis scenario analysis

All powered by real-time data from Yahoo Finance, supporting both US and European markets.

---

## 🎯 Features

- **Multi-Module Architecture** - Independent, reusable analysis tools
- **Automated Classification** - Intelligent company type detection
- **Region-Aware** - Handles US, European, and Canadian stocks
- **Batch Processing** - Analyze multiple stocks simultaneously
- **Multiple VaR Methods** - Parametric, Historical, Monte Carlo
- **Historical Crises** - Test against Dot-Com, GFC, COVID-19 scenarios

---

## 📊 Example Outputs

### Altman Z-Score
```
Ticker  Company                  Region  Classification  Model     ZScore  Zone
NVDA    NVIDIA Corporation       US      Non-Mfg        Modified  13.29   Safe
TSLA    Tesla, Inc.              US      Mfg            Original  18.79   Safe
MSFT    Microsoft Corporation    US      Non-Mfg        Modified   4.46   Safe
```

### Value at Risk
```
Ticker      Position Value    VaR (95.0%)    VaR %     Days
AAPL        $50,000.00       $2,765.84      5.53%      5
MSFT        $50,000.00       $2,241.41      4.48%      5
PORTFOLIO   $100,000.00      $4,141.06      4.14%      5
```

### Stress Test
```
Scenario: COVID-19 Crash (2020-02-01 to 2020-03-31)
Ticker  Start Price  End Price  Return   Gain/Loss      Max Drawdown
NVDA    $5.98       $6.61      +10.58%  +$996.73       -37.55%
TSLA    $52.00      $33.48     -35.62%  -$1,550.69     -60.63%
QCOM    $76.10      $61.23     -19.54%  -$599.85       -33.06%
AMD     $48.02      $47.86     -0.33%   -$7.95         -34.28%

Portfolio Impact: -6.49% loss, -$2,413.99, Max Drawdown: -41.14%
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/vdamov/financial-risk-analyzer.git
cd financial-risk-analyzer

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Altman Z-Score
python main.py zscore TSLA AAPL MSFT

# Value at Risk
python main.py var TSLA AAPL --confidence 95 --days 1

# Stress Testing
python main.py stress TSLA AAPL --scenario covid
```

---

## 📖 Module Documentation

### 1. Altman Z-Score Module

Predicts bankruptcy risk using the Altman Z-Score model.

```bash
# Basic usage
python main.py zscore TSLA AAPL MSFT

# European stocks
python main.py zscore ASML.AS SOLB.BR SAP.DE

# With verbose classification reasoning
python main.py zscore NVDA AMD INTC --verbose

# Save to custom file
python main.py zscore TSLA --out my_results.csv
```

**Zone Interpretation:**

| Zone | Z-Score | Risk Level |
|------|---------|-----------|
| Safe | > 2.99 (Mfg) / > 2.6 (Non-Mfg) | Low bankruptcy risk |
| Grey Zone | 1.81-2.99 / 1.1-2.6 | Moderate risk |
| Distress | < 1.81 / < 1.1 | High bankruptcy risk |

**Formulas:**
- **Manufacturing:** `Z = 1.2×(WC/TA) + 1.4×(RE/TA) + 3.3×(EBIT/TA) + 0.6×(MVE/TL) + 1.0×(Sales/TA)`
- **Non-Manufacturing:** `Z = 6.56×(WC/TA) + 3.26×(RE/TA) + 6.72×(EBIT/TA) + 1.05×(BVE/TL)`

---

### 2. Value at Risk (VaR) Module

Quantifies maximum expected loss at a given confidence level.

```bash
# Single stock VaR
python main.py var TSLA --confidence 95 --days 1

# Portfolio VaR with equal weights
python main.py var TSLA AAPL MSFT --confidence 99 --days 10

# Custom portfolio weights (must sum to 1.0)
python main.py var TSLA AAPL MSFT --weights 0.5 0.3 0.2

# Use dollar values instead of weights
python main.py var TSLA AAPL MSFT --values 50000 30000 20000

# Different methods
python main.py var TSLA --method parametric
python main.py var TSLA --method historical
python main.py var TSLA --method monte_carlo

# Custom portfolio value
python main.py var TSLA AAPL --portfolio-value 500000
```

**Available Methods:**

| Method | Description | Best For |
|--------|-------------|----------|
| **Historical** | Uses actual historical returns | Most conservative, no assumptions |
| **Parametric** | Assumes normal distribution | Quick calculation, smooth data |
| **Monte Carlo** | Simulates 10,000 scenarios | Complex portfolios, stress scenarios |

**Common Confidence Levels:**
- **90%** - Less conservative, wider range
- **95%** - Standard for risk management (default)
- **99%** - Very conservative, regulatory compliance

---

### 3. Stress Test Module

Tests portfolio performance under historical crisis scenarios.

```bash
# Test against COVID-19 crash
python main.py stress TSLA AAPL --scenario covid

# Test against 2008 Financial Crisis
python main.py stress TSLA AAPL --scenario gfc

# Test against Dot-Com bubble
python main.py stress TSLA AAPL --scenario dotcom

# Run all crisis scenarios
python main.py stress TSLA AAPL --scenario all

# Custom portfolio using weights
python main.py stress TSLA AAPL MSFT --weights 0.4 0.4 0.2 --portfolio-value 250000

# Custom portfolio using dollar values
python main.py stress TSLA AAPL MSFT --values 100000 100000 50000
```

**Historical Crisis Scenarios:**

| Scenario | Period | Market Drop | Description |
|----------|--------|-------------|-------------|
| **Dot-Com** | 2000-2002 | -49.1% (NASDAQ) | Tech bubble burst & 9/11 |
| **GFC** | 2007-2009 | -56.8% (S&P 500) | Global Financial Crisis |
| **COVID** | Feb-Mar 2020 | -33.9% (S&P 500) | Pandemic crash |

---

## 📁 Project Structure

```
financial-risk-analyzer/
├── main.py                 # CLI entry point
├── modules/
│   ├── __init__.py
│   ├── zscore.py          # Altman Z-Score module
│   ├── var.py             # Value at Risk module
│   └── stress_test.py     # Stress testing module
├── utils/
│   ├── __init__.py
│   ├── data_fetcher.py    # Yahoo Finance data fetching
│   └── helpers.py         # Classification & utilities
├── config.py              # Configuration constants
├── requirements.txt
└── README.md
```

---

## 💡 Use Cases

### Risk Management
- Monitor bankruptcy risk across portfolio holdings
- Calculate maximum expected losses
- Test resilience against historical crises

### Investment Analysis
- Screen stocks for financial health
- Compare risk profiles across sectors
- Identify distressed companies

### Portfolio Construction
- Balance risk across multiple assets
- Stress test allocation strategies
- Optimize risk-adjusted returns

### Academic Research
- Financial modeling and analysis
- Crisis impact studies
- Risk assessment methodologies

---

## 🛠️ Advanced Examples

### Combined Analysis Workflow

```bash
# 1. Check financial health
python main.py zscore TSLA AAPL MSFT --verbose

# 2. Measure current risk with value-based allocation
python main.py var TSLA AAPL MSFT --values 40000 35000 25000

# 3. Test historical resilience with same allocation
python main.py stress TSLA AAPL MSFT --values 40000 35000 25000 --scenario all
```

**Example Output**:
```
Ticker               Company Region Classification    Model ZScore      Zone                  Industry
  TSLA           Tesla, Inc.     US            Mfg Original  18.79      Safe        Auto Manufacturers
  AAPL            Apple Inc.     US        Non-Mfg Modified   1.87 Grey Zone      Consumer Electronics
  MSFT Microsoft Corporation     US        Non-Mfg Modified   4.46      Safe Software - Infrastructure
```
```
================================================================================
   Ticker     Method Position Value VaR (95.0%) VaR %  Days
     TSLA Historical     $40,000.00   $2,221.27 5.55%     1
     AAPL Historical     $35,000.00     $946.98 2.71%     1
     MSFT Historical     $25,000.00     $539.14 2.16%     1
PORTFOLIO Historical    $100,000.00   $2,602.41 2.60%     1
================================================================================

💡 Interpretation:
   Portfolio VaR: With 95.0% confidence, you will not lose more than
   $2,602.41 (2.60%) over 1 day(s).
```
```
================================================================================
📉 COVID-19 Crash (2020-02-01 to 2020-03-31)
   Global pandemic market crash
   Historical Market Drop: -33.9%
================================================================================

Ticker Start Price End Price  Return   Gain/Loss Max Drawdown
  TSLA      $52.00    $33.48 -35.62% -$14,249.74      -60.63%
  AAPL      $74.55    $61.69 -17.25%  -$6,037.65      -31.43%
  MSFT     $165.88   $152.84  -7.86%  -$1,965.88      -28.04%

📊 Portfolio Impact:
   Portfolio Return (with correlations): -21.35%
   Portfolio Gain/Loss: -$21,354.64
   Portfolio Max Drawdown: -41.19%
   Final Portfolio Value: $78,645.36
```

### Batch Processing from File

Create `portfolio.txt`:
```
TSLA
AAPL
MSFT
GOOGL
AMZN
```

Run analysis:
```bash
# Z-Score for all stocks
python main.py zscore $(cat portfolio.txt)

# VaR for portfolio (equal weights)
python main.py var $(cat portfolio.txt) --weights 0.2 0.2 0.2 0.2 0.2

# Or use dollar amounts
python main.py var $(cat portfolio.txt) --values 20000 20000 20000 20000 20000

# Stress test
python main.py stress $(cat portfolio.txt) --scenario all
```

---

## 📋 Requirements

- Python 3.8+
- yfinance >= 0.2.0 (tested with latest versions)
- pandas >= 1.5.0
- numpy >= 1.23.0
- scipy >= 1.9.0

```bash
pip install -r requirements.txt
```

**Note**: This toolkit is compatible with the latest yfinance API changes and automatically handles MultiIndex DataFrame structures.

---

## ⚠️ Limitations

1. **Data Availability**
   - Requires complete financial data from Yahoo Finance
   - Some European stocks may have limited historical data
   - Stress tests require data during crisis periods

2. **Classification Accuracy**
   - Automated classification is highly accurate but not perfect
   - Use `--verbose` flag to review reasoning
   - Edge cases exist for complex business models

3. **VaR Assumptions**
   - Parametric method assumes normal distribution
   - Historical method limited by available data
   - Past performance doesn't guarantee future results

4. **Not Financial Advice**
   - Educational and research purposes only
   - Always conduct thorough due diligence
   - Consider multiple risk factors

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add more crisis scenarios (1987 crash, 2011 European debt crisis)
- [ ] Implement Conditional VaR (CVaR/Expected Shortfall)
- [ ] Add more bankruptcy prediction models (Merton, KMV)
- [ ] Create visualization dashboard
- [ ] Add API for programmatic access
- [ ] Machine learning classification improvements
- [ ] Unit tests and integration tests

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📚 Background

### Altman Z-Score
Developed by Edward Altman in 1968, the Z-Score predicts corporate bankruptcy with ~80-90% accuracy up to two years before failure.

**Key Papers:**
- Altman, E. I. (1968). "Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy"
- Altman, E. I. (2000). "Predicting Financial Distress of Companies: Revisiting the Z-Score and ZETA Models"

### Value at Risk (VaR)
VaR quantifies the maximum expected loss at a given confidence level over a specified time horizon. Widely used by financial institutions for risk management.

### Stress Testing
Stress testing evaluates portfolio performance under extreme but plausible adverse scenarios. Required by regulators for financial institutions post-2008 crisis.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Edward Altman for developing the Z-Score model
- Yahoo Finance for providing free financial data
- The quantitative finance community for VaR methodologies
- The open-source community

---

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/vdamov/financial-risk-analyzer/issues)
- **Discussions:** [GitHub Discussions](https://github.com/vdamov/financial-risk-analyzer/discussions)
- **Documentation:** See module-specific help with `python main.py [module] --help`

---

## ⭐ Star History

If you find this tool useful, please consider giving it a star! ⭐

---

**Made with ❤️ for the quantitative finance community**