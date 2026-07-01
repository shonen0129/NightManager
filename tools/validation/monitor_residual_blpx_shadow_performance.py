#!/usr/bin/env python
"""Production Shadow-Run Performance Monitor for Residual-BLPX.

Evaluates daily portfolio weights from shadow run directories against realized
returns to track ex-post performance, realized Sharpe, drawdowns, and costs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src/ to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.tickers import JP_TICKERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ShadowMonitor")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Residual-BLPX Shadow Run Performance Monitor")
    parser.add_argument("--shadow-root", default="shadow_runs/residual_blpx", help="Root directory of shadow runs")
    parser.add_argument("--gap-input-dir", default=None, help="Gap-adjusted distribution directory (realized returns source)")
    parser.add_argument("--output-dir", default="results/production_shadow_monitoring", help="Output directory for monitoring results")
    parser.add_argument("--cost-bps-per-gross", type=float, default=10.0, help="Cost in bps per unit gross")
    parser.add_argument("--self-test", default="false", choices=["true", "false"], help="Run self-tests and exit")
    return parser.parse_args()


def scan_shadow_portfolios(shadow_root: Path) -> pd.DataFrame:
    """Scan all daily folders in shadow root and load shadow_portfolios.csv files."""
    records = []
    
    if not shadow_root.exists():
        logger.warning(f"Shadow root directory does not exist: {shadow_root}")
        return pd.DataFrame()
        
    for path in shadow_root.iterdir():
        if not path.is_dir():
            continue
        
        name = path.name
        is_date = False
        trade_date = None
        
        if len(name) == 8 and name.isdigit():
            is_date = True
            trade_date = f"{name[:4]}-{name[4:6]}-{name[6:]}"
        elif len(name) == 10 and name[4] == '-' and name[7] == '-':
            is_date = True
            trade_date = name
            
        if not is_date or not trade_date:
            continue
            
        port_file = path / "shadow_portfolios.csv"
        if not port_file.exists():
            port_file = path / "shadow_portfolios.parquet"
            
        if not port_file.exists():
            continue
            
        try:
            if port_file.suffix == ".parquet":
                df_day = pd.read_parquet(port_file)
            else:
                df_day = pd.read_csv(port_file)
                
            df_day["trade_date"] = trade_date
            records.append(df_day)
        except Exception as e:
            logger.error(f"Error reading {port_file}: {e}")
            
    if not records:
        return pd.DataFrame()
        
    return pd.concat(records, ignore_index=True)


def calculate_expost_metrics(
    rets: np.ndarray,
    weights_matrix: np.ndarray,
    cost_bps_per_gross: float = 10.0
) -> dict:
    """Calculate ex-post return, risk, Sharpe, drawdowns, and turnover."""
    T = len(rets)
    if T == 0:
        return {}
        
    ann_ret = np.mean(rets) * 252.0
    ann_vol = np.std(rets, ddof=1) * np.sqrt(252.0) if T > 1 else 0.0
    sharpe = ann_ret / ann_vol if ann_vol > 0.0 else 0.0
    
    # Sortino
    down_rets = rets[rets < 0.0]
    down_vol = np.std(down_rets, ddof=1) * np.sqrt(252.0) if len(down_rets) > 1 else 0.0
    sortino = ann_ret / down_vol if down_vol > 0.0 else 0.0
    
    # Drawdown
    W = np.cumprod(1.0 + rets)
    running_max = np.maximum.accumulate(W)
    drawdowns = (W / running_max) - 1.0
    mdd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    calmar = ann_ret / abs(mdd) if abs(mdd) > 0.0 else 0.0
    
    # CVaR 95% / 99%
    cvar95 = float(np.mean(np.percentile(rets, 5.0)))
    cvar99 = float(np.mean(np.percentile(rets, 1.0)))
    
    # Max weight
    max_w = float(np.max(np.abs(weights_matrix)))
    
    # Turnover
    w_prev = np.vstack([np.zeros(weights_matrix.shape[1]), weights_matrix[:-1]])
    turns = np.sum(np.abs(weights_matrix - w_prev), axis=1)
    avg_turn = float(np.mean(turns))
    
    # Hit rate
    hit_rate = float(np.sum(rets > 0) / T)
    
    return {
        "annualized_net_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": mdd,
        "calmar_ratio": calmar,
        "cvar_95_pct": cvar95,
        "cvar_99_pct": cvar99,
        "average_max_abs_weight": max_w,
        "average_turnover": avg_turn,
        "hit_rate": hit_rate
    }


def run_self_tests() -> int:
    """Run verification self-tests on temporary simulated directory."""
    logger.info("=== Running Performance Monitor Self-Tests ===")
    
    import tempfile
    import shutil
    
    temp_dir = Path(tempfile.mkdtemp(prefix="monitor_self_test_"))
    
    try:
        # Create mock daily shadow run directories
        shadow_root = temp_dir / "shadow_runs"
        shadow_root.mkdir()
        
        dates = ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05"]
        
        for dt in dates:
            dt_num = dt.replace("-", "")
            day_dir = shadow_root / dt_num
            day_dir.mkdir()
            
            # Create mock portfolio weights
            recs = []
            for cand in ["baseline", "primary_ruleD"]:
                for idx, tk in enumerate(JP_TICKERS):
                    w = 0.2 if idx < 5 else (-0.2 if idx >= len(JP_TICKERS)-5 else 0.0)
                    recs.append({
                        "trade_date": dt,
                        "candidate": cand,
                        "ticker": tk,
                        "weight_final": w,
                        "expected_cost_bps": abs(w) * 10.0
                    })
            pd.DataFrame(recs).to_csv(day_dir / "shadow_portfolios.csv", index=False)
            
        # Create mock realized returns
        gap_dir = temp_dir / "gap_distribution"
        gap_dir.mkdir()
        
        long_recs = []
        for dt in dates:
            for tk in JP_TICKERS:
                long_recs.append({
                    "trade_date": dt,
                    "ticker": tk,
                    "realized_target_return": np.random.normal(0.0005, 0.01)
                })
        pd.DataFrame(long_recs).to_csv(gap_dir / "gap_adjusted_distribution_long.csv", index=False)
        
        # Run monitor pipeline
        output_dir = temp_dir / "monitoring_output"
        
        df_ports = scan_shadow_portfolios(shadow_root)
        assert len(df_ports) > 0, "Failed to scan shadow portfolios"
        
        # Load realized returns
        df_realized = pd.read_csv(gap_dir / "gap_adjusted_distribution_long.csv")
        df_realized["trade_date"] = pd.to_datetime(df_realized["trade_date"]).dt.strftime("%Y-%m-%d")
        
        # Merge
        df_merged = pd.merge(df_ports, df_realized, on=["trade_date", "ticker"], how="inner")
        assert len(df_merged) > 0, "Merge with realized returns failed"
        
        # Generate scorecard
        candidates = df_merged["candidate"].unique()
        assert "baseline" in candidates
        assert "primary_ruleD" in candidates
        
        logger.info("Performance Monitor Self-Tests PASSED.")
        return 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    args = parse_arguments()
    
    if args.self_test == "true":
        sys.exit(run_self_tests())
        
    shadow_root = Path(args.shadow_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    if not args.gap_input_dir:
        logger.error("Missing required --gap-input-dir pointing to realized returns folder.")
        sys.exit(1)
        
    gap_input_dir = Path(args.gap_input_dir)
    
    logger.info(f"Scanning shadow runs from: {shadow_root}")
    df_ports = scan_shadow_portfolios(shadow_root)
    if df_ports.empty:
        logger.error(f"No shadow run portfolios found under {shadow_root}. Exiting.")
        sys.exit(1)
        
    logger.info(f"Loaded {len(df_ports)} daily shadow portfolio rows.")
    
    # Load realized returns
    logger.info(f"Loading realized returns from: {gap_input_dir}")
    df_realized = pd.read_csv(gap_input_dir / "gap_adjusted_distribution_long.csv")
    df_realized["trade_date"] = pd.to_datetime(df_realized["trade_date"]).dt.strftime("%Y-%m-%d")
    
    # Slice common dates
    dates_in_ports = set(df_ports["trade_date"].unique())
    dates_in_realized = set(df_realized["trade_date"].unique())
    common_dates = sorted(list(dates_in_ports & dates_in_realized))
    
    if not common_dates:
        logger.error("No common dates between daily shadow run portfolios and realized returns file.")
        sys.exit(1)
        
    logger.info(f"Monitoring shadow-runs over {len(common_dates)} overlapping trading days ({common_dates[0]} to {common_dates[-1]}).")
    
    df_ports = df_ports[df_ports["trade_date"].isin(common_dates)]
    df_realized = df_realized[df_realized["trade_date"].isin(common_dates)]
    
    # Merge on date & ticker
    df_merged = pd.merge(df_ports, df_realized, on=["trade_date", "ticker"], how="inner")
    
    # Compute daily net returns
    df_merged["gross_return_contribution"] = df_merged["weight_final"] * df_merged["realized_target_return"]
    if "expected_cost_bps" in df_merged.columns:
        df_merged["cost_bps"] = df_merged["expected_cost_bps"]
    else:
        df_merged["cost_bps"] = np.abs(df_merged["weight_final"]) * args.cost_bps_per_gross
    df_merged["cost_rate"] = df_merged["cost_bps"] / 10000.0
    
    # Group by candidate and date
    df_daily = df_merged.groupby(["candidate", "trade_date"]).agg({
        "gross_return_contribution": "sum",
        "cost_rate": "sum",
        "weight_final": lambda x: np.sum(np.abs(x))
    }).reset_index()
    df_daily.rename(columns={"gross_return_contribution": "gross_return", "weight_final": "realized_gross"}, inplace=True)
    df_daily["net_return"] = df_daily["gross_return"] - df_daily["cost_rate"]
    
    # Save daily net returns to csv
    df_daily.to_csv(output_dir / "candidate_daily_returns.csv", index=False)
    
    # Calculate ex-post metrics for each candidate
    candidates = df_daily["candidate"].unique()
    metrics_records = []
    
    for cand in candidates:
        df_cand_daily = df_daily[df_daily["candidate"] == cand].sort_values("trade_date")
        
        # Construct weight matrix to evaluate max weight and turnover
        df_cand_port = df_ports[df_ports["candidate"] == cand]
        df_w_pivot = df_cand_port.pivot(index="trade_date", columns="ticker", values="weight_final")
        df_w_pivot = df_w_pivot.reindex(index=common_dates, columns=JP_TICKERS).fillna(0.0)
        
        met = calculate_expost_metrics(
            df_cand_daily["net_return"].values,
            df_w_pivot.values,
            args.cost_bps_per_gross
        )
        met["candidate"] = cand
        met["days_count"] = len(df_cand_daily)
        metrics_records.append(met)
        
    df_metrics = pd.DataFrame(metrics_records)
    df_metrics.to_csv(output_dir / "monitoring_scorecard.csv", index=False)
    
    # Plot 1: Cumulative Net Returns
    plt.figure(figsize=(10, 6))
    for cand in candidates:
        df_cand = df_daily[df_daily["candidate"] == cand].sort_values("trade_date")
        dates = pd.to_datetime(df_cand["trade_date"])
        cum_ret = np.cumprod(1.0 + df_cand["net_return"].values) - 1.0
        plt.plot(dates, cum_ret * 100.0, label=cand)
    plt.title("Live Shadow-Run: Cumulative Net Return Comparison")
    plt.ylabel("Cumulative Net Return (%)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / "cumulative_net_return.png", bbox_inches="tight")
    plt.close()
    
    # Plot 2: Drawdowns
    plt.figure(figsize=(10, 5))
    for cand in candidates:
        df_cand = df_daily[df_daily["candidate"] == cand].sort_values("trade_date")
        dates = pd.to_datetime(df_cand["trade_date"])
        W = np.cumprod(1.0 + df_cand["net_return"].values)
        rm = np.maximum.accumulate(W)
        dd = (W / rm) - 1.0
        plt.plot(dates, dd * 100.0, label=cand)
    plt.title("Live Shadow-Run: Drawdown Curves Comparison")
    plt.ylabel("Drawdown (%)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / "drawdowns.png", bbox_inches="tight")
    plt.close()
    
    # Render monitoring_report.md
    rep_text = f"""# Production Shadow-Run Live Monitoring Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. Summary

- **Overlapping Dates Evaluated**: `{common_dates[0]}` to `{common_dates[-1]}` ({len(common_dates)} trading days)
- **Shadow Root**: `{args.shadow_root}`
- **Realized Return Source**: `{args.gap_input_dir}`

## 2. Ex-Post Performance Scorecard

| Candidate | Annualized Net Return | Annualized Volatility | Sharpe Ratio | Sortino Ratio | Max Drawdown | Calmar Ratio | Average Turnover | Hit Rate | Days |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    for _, row in df_metrics.iterrows():
        rep_text += (
            f"| {row['candidate']} | {row['annualized_net_return']*100.0:.2f}% | "
            f"{row['annualized_volatility']*100.0:.2f}% | {row['sharpe_ratio']:.4f} | "
            f"{row['sortino_ratio']:.4f} | {row['max_drawdown']*100.0:.2f}% | "
            f"{row['calmar_ratio']:.4f} | {row['average_turnover']:.4f} | "
            f"{row['hit_rate']*100.0:.1f}% | {int(row['days_count'])} |\n"
        )
        
    rep_text += """
## 3. Cumulative Performance Visualization

![Cumulative Net Return](plots/cumulative_net_return.png)

![Drawdown Comparison](plots/drawdowns.png)

## 4. Key Observations

- Realized returns represent actual market movement from 9:10 TO CLOSE.
- Expected daily transaction cost drag is set to 10 bps per unit gross.
- Check validation outputs and baseline overlap counts under the daily shadow run directory to monitor consistency.
"""
    
    with open(output_dir / "monitoring_report.md", "w") as f:
        f.write(rep_text)
        
    logger.info(f"Live performance monitoring report generated: {output_dir / 'monitoring_report.md'}")


if __name__ == "__main__":
    main()
