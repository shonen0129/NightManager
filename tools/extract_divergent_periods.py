import pandas as pd
import sys
from pathlib import Path

# Usage: python extract_divergent_periods.py <run_a_dir> <run_b_dir> [threshold]
if len(sys.argv) < 3:
    print('Usage: extract_divergent_periods.py <run_a_dir> <run_b_dir> [threshold_rel]')
    sys.exit(2)

pa = Path(sys.argv[1]) / 'daily_results.csv'
pb = Path(sys.argv[2]) / 'daily_results.csv'
threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01

ra = pd.read_csv(pa, index_col=0, parse_dates=True)
rb = pd.read_csv(pb, index_col=0, parse_dates=True)

# align
common = ra.index.intersection(rb.index).sort_values()
ra = ra.reindex(common)
rb = rb.reindex(common)

wa = (1.0 + ra['daily_return']).cumprod()
wb = (1.0 + rb['daily_return']).cumprod()

rel_diff = (wa - wb) / wa
abs_rel = rel_diff.abs()

# find contiguous periods where abs_rel > threshold
mask = abs_rel > threshold
periods = []
start = None
for dt, val in mask.items():
    if val and start is None:
        start = dt
    if not val and start is not None:
        end = prev_dt
        seg = abs_rel.loc[start:end]
        max_idx = seg.idxmax()
        periods.append({
            'start': start.date(),
            'end': end.date(),
            'days': len(seg),
            'max_rel_diff': float(seg.max()),
            'date_max': max_idx.date(),
        })
        start = None
    prev_dt = dt
# handle tail
if start is not None:
    end = prev_dt
    seg = abs_rel.loc[start:end]
    max_idx = seg.idxmax()
    periods.append({
        'start': start.date(),
        'end': end.date(),
        'days': len(seg),
        'max_rel_diff': float(seg.max()),
        'date_max': max_idx.date(),
    })

out = pd.DataFrame(periods)
print('Threshold (relative):', threshold)
print('Number of divergent periods:', len(out))
if len(out) > 0:
    print(out.to_string(index=False))

# Save full diff series
out_dir = Path('results') / f'divergence_{pa.parent.name}_vs_{pb.parent.name}'
out_dir.mkdir(parents=True, exist_ok=True)
series_df = pd.DataFrame({
    'wealth_a': wa,
    'wealth_b': wb,
    'rel_diff': rel_diff,
    'abs_rel_diff': abs_rel,
})
series_df.to_csv(out_dir / 'divergence_series.csv', index_label='date')
print('Saved divergence series to', out_dir / 'divergence_series.csv')
