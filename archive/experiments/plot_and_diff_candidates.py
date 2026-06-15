import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pairs = [
    # AR_center: best by AR vs best by R/R
    (
        Path(r"results/20260527_011325_fine_ar_center_bc0_45_bw75"),
        Path(r"results/20260527_011524_fine_ar_center_bc0_9_bw60"),
        'AR_center_AR_vs_RR',
    ),
    # RR_center: best by AR vs best by R/R
    (
        Path(r"results/20260527_011501_fine_ar_center_bc0_8_bw75"),
        Path(r"results/20260527_011729_fine_rr_center_bc1_2_bw60"),
        'RR_center_AR_vs_RR',
    ),
]

out_root = Path('results') / 'fine_grid_local'
out_root.mkdir(parents=True, exist_ok=True)

for a_dir, b_dir, tag in pairs:
    fa = a_dir / 'daily_results.csv'
    fb = b_dir / 'daily_results.csv'
    if not fa.exists() or not fb.exists():
        print('Missing daily_results for', a_dir, b_dir)
        continue
    ra = pd.read_csv(fa, index_col=0, parse_dates=True)
    rb = pd.read_csv(fb, index_col=0, parse_dates=True)
    common = ra.index.intersection(rb.index).sort_values()
    ra = ra.reindex(common)
    rb = rb.reindex(common)

    wa = (1.0 + ra['daily_return']).cumprod()
    wb = (1.0 + rb['daily_return']).cumprod()

    # Plot cumulative wealth
    plt.figure(figsize=(10,6))
    plt.plot(wa.index, wa.values, label=f'A: {a_dir.name}')
    plt.plot(wb.index, wb.values, label=f'B: {b_dir.name}')
    plt.legend()
    plt.title(f'Cumulative wealth: {tag}')
    plt.ylabel('Wealth (relative)')
    plt.xlabel('Date')
    plt.grid(True)
    plot_path = out_root / f'cumwealth_{tag}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Compute daily diffs
    diff = ra.copy()
    # Only compute differences for numeric columns
    numeric_cols = [c for c in ra.columns if pd.api.types.is_numeric_dtype(ra[c])]
    numeric_cols = [c for c in numeric_cols if c in rb.columns and pd.api.types.is_numeric_dtype(rb[c])]
    for col in numeric_cols:
        diff[col] = ra[col] - rb[col]
    diff['abs_daily_return_diff'] = (ra['daily_return'] - rb['daily_return']).abs()
    diff['cumwealth_a'] = wa
    diff['cumwealth_b'] = wb
    diff['cumwealth_diff'] = (wa - wb).abs()

    # Save top differing days by cumwealth_diff
    top = diff.sort_values('cumwealth_diff', ascending=False).head(50)
    top_path = out_root / f'diff_top50_{tag}.csv'
    top.to_csv(top_path, index_label='date')

    # Save full diff series
    series_path = out_root / f'diff_series_{tag}.csv'
    diff.to_csv(series_path, index_label='date')

    print('Saved', plot_path, top_path, series_path)
