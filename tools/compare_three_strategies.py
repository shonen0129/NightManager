import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Candidate result dirs
revenue_dir = Path('results') / '20260527_011325_fine_ar_center_bc0_45_bw75'
compromise_dir = Path('results') / '20260527_011501_fine_ar_center_bc0_8_bw75'
baseline_dir = Path('results') / '20260527_001024_baseline_no_topix'

out_dir = Path('results') / 'fine_grid_local'
out_dir.mkdir(parents=True, exist_ok=True)

labels = {
    'revenue': revenue_dir,
    'compromise': compromise_dir,
    'baseline': baseline_dir,
}

series = {}
for name, d in labels.items():
    f = d / 'daily_results.csv'
    if not f.exists():
        raise FileNotFoundError(f"Missing {f}")
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    dr = pd.to_numeric(df['daily_return'], errors='coerce').fillna(0.0)
    series[name] = (1.0 + dr).cumprod()

# Align indices (intersection)
common = None
for s in series.values():
    common = s.index if common is None else common.intersection(s.index)
common = common.sort_values()

for k in list(series.keys()):
    series[k] = series[k].reindex(common)

# Combine and save series CSV
combined = pd.DataFrame({k: v for k, v in series.items()})
combined.to_csv(out_dir / 'compare_three_series.csv', index_label='date')

# Plot
plt.figure(figsize=(10,6))
plt.plot(combined.index, combined['revenue'], label='Revenue-max (beta=0.45,bw=75)')
plt.plot(combined.index, combined['compromise'], label='Compromise (beta=0.8,bw=75)')
plt.plot(combined.index, combined['baseline'], label='Baseline (beta=0)')
plt.legend()
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('Cumulative Wealth')
plt.title('Strategy comparison: Revenue-max vs Compromise vs Baseline')
plt.tight_layout()
plot_path = out_dir / 'compare_three.png'
plt.savefig(plot_path, dpi=150)
print('Saved', plot_path, out_dir / 'compare_three_series.csv')
