#!/usr/bin/env python3
"""
Financial Risk Analyzer - Main Entry Point

Provides multiple financial analysis tools:
- Altman Z-Score: Bankruptcy risk assessment
- Value at Risk (VaR): Portfolio risk measurement
- Stress Testing: Historical crisis scenario analysis
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Financial Risk Analyzer - Comprehensive risk assessment toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Altman Z-Score
  python main.py zscore TSLA AAPL MSFT
  python main.py zscore ASML.AS --verbose
  
  # Value at Risk
  python main.py var TSLA AAPL MSFT --confidence 95 --days 10
  python main.py var TSLA --method historical
  
  # Stress Testing
  python main.py stress TSLA AAPL --scenario covid
  python main.py stress AAPL --scenario all
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Analysis module to run")

    # ============================================================
    # Z-Score Command
    # ============================================================
    zscore_parser = subparsers.add_parser(
        "zscore", help="Calculate Altman Z-Score for bankruptcy risk assessment"
    )
    zscore_parser.add_argument("tickers", nargs="+", help="Stock tickers to analyze")
    zscore_parser.add_argument(
        "--out", default="zscore_results.csv", help="Output CSV file"
    )
    zscore_parser.add_argument(
        "--verbose", action="store_true", help="Show classification reasoning"
    )
    zscore_parser.add_argument(
        "--debug", action="store_true", help="Show detailed financial data"
    )

    # ============================================================
    # VaR Command
    # ============================================================
    var_parser = subparsers.add_parser(
        "var", help="Calculate Value at Risk for portfolio risk measurement"
    )
    var_parser.add_argument("tickers", nargs="+", help="Stock tickers to analyze")
    var_parser.add_argument(
        "--confidence",
        type=float,
        default=95.0,
        help="Confidence level (90, 95, 99). Default: 95",
    )
    var_parser.add_argument(
        "--days", type=int, default=1, help="Time horizon in days. Default: 1"
    )
    var_parser.add_argument(
        "--method",
        choices=["parametric", "historical", "monte_carlo"],
        default="historical",
        help="VaR calculation method. Default: historical",
    )
    var_parser.add_argument(
        "--portfolio-value",
        type=float,
        default=100000,
        help="Portfolio value in USD. Default: 100000",
    )
    var_parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        help="Portfolio weights (must sum to 1). Equal weights if not specified",
    )
    var_parser.add_argument(
        "--values",
        nargs="+",
        type=float,
        help="Position values in USD for each ticker. Overrides --portfolio-value and --weights",
    )
    var_parser.add_argument("--out", default="var_results.csv", help="Output CSV file")

    # ============================================================
    # Stress Test Command
    # ============================================================
    stress_parser = subparsers.add_parser(
        "stress", help="Run historical crisis stress tests"
    )
    stress_parser.add_argument("tickers", nargs="+", help="Stock tickers to analyze")
    stress_parser.add_argument(
        "--scenario",
        choices=["dotcom", "gfc", "covid", "all"],
        default="all",
        help="Crisis scenario to test. Default: all",
    )
    stress_parser.add_argument(
        "--portfolio-value",
        type=float,
        default=100000,
        help="Portfolio value in USD. Default: 100000",
    )
    stress_parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        help="Portfolio weights (must sum to 1). Equal weights if not specified",
    )
    stress_parser.add_argument(
        "--values",
        nargs="+",
        type=float,
        help="Position values in USD for each ticker. Overrides --portfolio-value and --weights",
    )
    stress_parser.add_argument(
        "--out", default="stress_test_results.csv", help="Output CSV file"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Input validation
    if args.command in ["var", "stress"]:
        # Validate confidence level for VaR
        if args.command == "var":
            if args.confidence <= 0 or args.confidence >= 100:
                parser.error(
                    f"Confidence level must be between 0 and 100 (got {args.confidence})"
                )

            if args.days <= 0:
                parser.error(f"Days must be positive (got {args.days})")

            if args.days > 365:
                print(
                    f"⚠️  Warning: {args.days} days is a very long time horizon. Results may be unreliable.",
                    file=sys.stderr,
                )

        # Validate portfolio value
        if hasattr(args, "portfolio_value") and args.portfolio_value <= 0:
            parser.error(
                f"Portfolio value must be positive (got {args.portfolio_value})"
            )

        # Validate weights
        if hasattr(args, "weights") and args.weights:
            if any(w < 0 for w in args.weights):
                parser.error("Weights cannot be negative")
            if any(w > 1 for w in args.weights):
                parser.error("Individual weights cannot exceed 1.0")

        # Validate values
        if hasattr(args, "values") and args.values:
            if any(v <= 0 for v in args.values):
                parser.error("Position values must be positive")

            # Check for conflicting options
            if hasattr(args, "weights") and args.weights:
                print(
                    "⚠️  Warning: Both --values and --weights specified. Using --values.",
                    file=sys.stderr,
                )

    # Validate tickers
    if args.command in ["zscore", "var", "stress"]:
        if not args.tickers:
            parser.error("At least one ticker is required")

    # Route to appropriate module
    try:
        if args.command == "zscore":
            from modules.zscore import run_zscore

            run_zscore(args)
        elif args.command == "var":
            from modules.var import run_var

            run_var(args)
        elif args.command == "stress":
            from modules.stress_test import run_stress_test

            run_stress_test(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}", file=sys.stderr)
        if hasattr(args, "debug") and args.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
