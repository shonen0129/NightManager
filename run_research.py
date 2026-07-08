#!/usr/bin/env python3
"""Research script execution alias.

Usage:
  python run_research.py macro analyze_gold_correlation
  python run_research.py blpx experiment_bayesian_blpx
  python run_research.py sprint run_sprint0_diagnostics
  python run_research.py backtest run_overnight_holding_backtest
"""

import argparse
import sys
from pathlib import Path

# Add src/ to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Execute research scripts from src/research/scripts/",
        epilog="Examples:\n"
               "  python run_research.py macro analyze_gold_correlation\n"
               "  python run_research.py blpx experiment_bayesian_blpx\n"
               "  python run_research.py sprint run_sprint0_diagnostics\n"
               "  python run_research.py backtest run_overnight_holding_backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("category", choices=["macro", "blpx", "sprint", "backtest"],
                        help="Research script category")
    parser.add_argument("script", help="Script name (without .py extension)")
    parser.add_argument("args", nargs=argparse.REMAINDER,
                        help="Arguments to pass to the research script")

    args = parser.parse_args()

    # Build script path
    script_path = ROOT / "src" / "research" / "scripts" / args.category / f"{args.script}.py"

    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        print(f"Available scripts in {args.category}/:")
        script_dir = script_path.parent
        for py_file in sorted(script_dir.glob("*.py")):
            print(f"  - {py_file.stem}")
        sys.exit(1)

    # Execute the script
    import runpy
    sys.argv = [str(script_path)] + args.args
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
