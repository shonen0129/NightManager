import pandas as pd
import sys
from pathlib import Path

if len(sys.argv) < 3:
    print("Usage: analyze_total_return_diff.py <run_a> <run_b>")
    sys.exit(2)

pa = Path(sys.argv[1]) / 'daily_results.csv'
pb = Path(sys.argv[2]) / 'daily_results.csv'
ra = pd.read_csv(pa, index_col=0, parse_dates=True)
rb = pd.read_csv(pb, index_col=0, parse_dates=True)

def stats(df):
    dr = df['daily_return']
    wealth = (1.0 + dr).cumprod()
    return {
        'rows': len(df),
        'final_wealth': float(wealth.iloc[-1]),
        'mean_daily_return': float(dr.mean()),
        'std_daily_return': float(dr.std()),
        'nonzero_active_days': int((df.get('active_count', 0) > 0).sum()),
    }

sa = stats(ra)
sb = stats(rb)
print('Run A:', pa.parent)
print(sa)
print('\nRun B:', pb.parent)
print(sb)

# Show first differing date where positions differ
common_index = ra.index.intersection(rb.index)

# compute cumulative wealth series
wa = (1.0 + ra['daily_return']).cumprod()
wb = (1.0 + rb['daily_return']).cumprod()

# difference series
diff = (wa.reindex(common_index) - wb.reindex(common_index)).abs()
if diff.max() == 0:
    print('\nNo cumulative-wealth difference detected')
else:
    first_diff = diff[diff > 1e-12].index[0]
    print('\nFirst significant wealth divergence at', first_diff.date())
    print('Wealth A at that date:', wa.loc[first_diff])
    print('Wealth B at that date:', wb.loc[first_diff])

# show sample rows where active_count differs
if 'active_count' in ra.columns and 'active_count' in rb.columns:
    idx = common_index
n_diff_active = 0
rows = []
for dt in common_index[:300]:
    arow = ra.loc[dt]
    brow = rb.loc[dt]
    if any((arow.fillna(0) - brow.fillna(0)).abs() > 1e-12):
        n_diff_active += 1
        rows.append(dt)
        if len(rows) >= 5:
            break
print('\nSample differing rows count (first 5 shown):', n_diff_active)
for d in rows:
    print('\nDate', d.date())
    print('A:', ra.loc[d].to_dict())
    print('B:', rb.loc[d].to_dict())
