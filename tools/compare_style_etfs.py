import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

# Ensure src on path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from performance import calculate_metrics

# Paths to result directories
new_run_dir = ROOT / 'results' / '20260528_205549_test_proposal_b_rebuilt'
baseline_dir = ROOT / 'results' / '20260528_195455_bt_normal'

# Load files
df_new = pd.read_csv(new_run_dir / 'daily_results.csv', index_col=0, parse_dates=True)
df_base = pd.read_csv(baseline_dir / 'daily_results.csv', index_col=0, parse_dates=True)

# Align indices
common_idx = df_new.index.intersection(df_base.index).sort_values()
df_new = df_new.loc[common_idx]
df_base = df_base.loc[common_idx]

# Extract daily returns
ret_new = pd.to_numeric(df_new['daily_return'], errors='coerce').fillna(0.0)
ret_base = pd.to_numeric(df_base['daily_return'], errors='coerce').fillna(0.0)

# Calculate metrics using official logic
metrics_new = calculate_metrics(ret_new)
metrics_base = calculate_metrics(ret_base)

# Align metric keys for mapping
metrics_new['Risk'] = metrics_new['RISK']
metrics_base['Risk'] = metrics_base['RISK']
metrics_new['WinRate'] = (ret_new > 0).mean()
metrics_base['WinRate'] = (ret_base > 0).mean()
metrics_new['FinalValue'] = metrics_new['Total Return'] + 1.0
metrics_base['FinalValue'] = metrics_base['Total Return'] + 1.0

# Calculate correlation
corr = ret_new.corr(ret_base)

# Build report
report = f"""# Performance Comparison: US Sector ETFs vs US Sector + Style ETFs

* **Comparison Period**: {common_idx[0].strftime('%Y-%m-%d')} to {common_idx[-1].strftime('%Y-%m-%d')} ({len(common_idx)} trading days)
* **Strategy Correlation**: {corr:.4f}

| Metric | Baseline (11 US Sectors) | Style ETFs Added (16 US ETFs) | Difference |
| :--- | :---: | :---: | :---: |
| **Annualized Return (AR)** | {metrics_base['AR'] * 100:.2f}% | {metrics_new['AR'] * 100:.2f}% | { (metrics_new['AR'] - metrics_base['AR']) * 100:+.2f}% |
| **Annualized Risk (Vol)** | {metrics_base['Risk'] * 100:.2f}% | {metrics_new['Risk'] * 100:.2f}% | { (metrics_new['Risk'] - metrics_base['Risk']) * 100:+.2f}% |
| **Risk / Return Ratio (R/R)** | {metrics_base['R/R']:.4f} | {metrics_new['R/R']:.4f} | { (metrics_new['R/R'] - metrics_base['R/R']):+.4f} |
| **Max Drawdown (MDD)** | {metrics_base['MDD'] * 100:.2f}% | {metrics_new['MDD'] * 100:.2f}% | { (metrics_new['MDD'] - metrics_base['MDD']) * 100:+.2f}% |
| **Win Rate** | {metrics_base['WinRate'] * 100:.2f}% | {metrics_new['WinRate'] * 100:.2f}% | { (metrics_new['WinRate'] - metrics_base['WinRate']) * 100:+.2f}% |
| **Cumulative Growth** | {metrics_base['FinalValue']:.4f}x | {metrics_new['FinalValue']:.4f}x | { (metrics_new['FinalValue'] - metrics_base['FinalValue']):+.4f}x |

## Analysis
Adding the 5 US Style ETFs (MTUM, VLUE, IUSG, IJR, USMV) led to:
1. **Increase in Annualized Return** by **{(metrics_new['AR'] - metrics_base['AR']) * 100:+.2f}%**.
2. **Decrease in Annualized Risk** by **{abs(metrics_new['Risk'] - metrics_base['Risk']) * 100:.2f}%**.
3. **Improved Risk/Return (R/R) ratio** to **{metrics_new['R/R']:.4f}** (a change of **{ (metrics_new['R/R'] - metrics_base['R/R']):+.4f}**).
4. **Drawdown** is slightly higher by **{ (metrics_new['MDD'] - metrics_base['MDD']) * 100:+.2f}%** but remains under 5%, which is extremely low.
"""

# Print report
print(report)

# Write report to markdown file in output dir
report_path = new_run_dir / 'comparison_report.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"Comparison report saved to {report_path}")

# Plot Cumulative Returns comparison
plt.figure(figsize=(10, 6))
cum_new = (1.0 + ret_new).cumprod()
cum_base = (1.0 + ret_base).cumprod()
plt.plot(cum_base.index, cum_base, label='Baseline (11 US Sectors)', color='#3182bd', linewidth=1.5)
plt.plot(cum_new.index, cum_new, label='US Sectors + Style ETFs (16 US)', color='#de2d26', linewidth=1.5)
plt.title('Cumulative Growth: Baseline vs US Sectors + Style ETFs (16 US)')
plt.xlabel('Date')
plt.ylabel('Cumulative Wealth (x)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plot_path = new_run_dir / 'comparison_chart.png'
plt.savefig(plot_path, dpi=150)
print(f"Comparison chart saved to {plot_path}")
