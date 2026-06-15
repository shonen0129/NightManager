import sys
from pathlib import Path
import functools
import csv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import production
import pandas as pd

out_dir = ROOT / 'results' / 'fine_grid_local'
out_dir.mkdir(parents=True, exist_ok=True)
summary_csv = out_dir / 'fine_grid_results.csv'

# Define fine grids
ar_coefs = np.arange(0.3, 0.9, 0.05).round(2)  # around 0.6
rr_coefs = np.arange(0.8, 1.6, 0.05).round(2)  # around 1.2
beta_windows = [45, 60, 75]

grids = [
    ('AR_center', list(ar_coefs)),
    ('RR_center', list(rr_coefs)),
]

rows = []
orig_Prod = production.ProductionConfig
count = 0
total = sum(len(coefs) * len(beta_windows) for _, coefs in grids)
for label, coefs in grids:
    for bc in coefs:
        for bw in beta_windows:
            count += 1
            tag = f"fine_{label}_bc{bc}_bw{bw}"
            print(f"[{count}/{total}] {tag}")
            production.ProductionConfig = functools.partial(orig_Prod, topix_beta_coef=float(bc), beta_window=int(bw))
            try:
                out = production.run_production(
                    start_date='2015-01-01',
                    output_root=production.get_default_results_root(),
                    run_tag=tag,
                    skip_chart=True,
                )
                m = pd.read_csv(Path(out) / 'metrics.csv')
                met = m.iloc[0].to_dict()
                met.update({'label': label, 'topix_beta_coef': float(bc), 'beta_window': int(bw), 'out_dir': str(out)})
                rows.append(met)
            except Exception as e:
                print('Error for', tag, e)
                rows.append({'label': label, 'topix_beta_coef': float(bc), 'beta_window': int(bw), 'error': str(e)})

# restore
production.ProductionConfig = orig_Prod

# write CSV
keys = set()
for r in rows:
    keys.update(r.keys())
keys = sorted(keys)
with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

# Summarize best by AR and by R/R for each label
df = pd.read_csv(summary_csv)
for label in df['label'].unique():
    sub = df[df['label'] == label].copy()
    sub = sub[pd.isna(sub.get('error'))]
    sub[['AR','R/R']] = sub[['AR','R/R']].apply(pd.to_numeric, errors='coerce')
    best_ar = sub.loc[sub['AR'].idxmax()]
    best_rr = sub.loc[sub['R/R'].idxmax()]
    print('\nSummary for', label)
    print('Best by AR:', best_ar[['topix_beta_coef','beta_window','AR','R/R','MDD','out_dir']].to_string())
    print('Best by R/R:', best_rr[['topix_beta_coef','beta_window','AR','R/R','MDD','out_dir']].to_string())

print('Fine grid search completed. Results:', summary_csv)
