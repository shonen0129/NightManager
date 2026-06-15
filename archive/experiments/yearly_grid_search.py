import sys
from pathlib import Path
import functools
import csv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import production
import pandas as pd

# Grid settings
years = list(range(2015, 2026))  # 2015..2025
beta_coefs = [0.0, 0.6, 1.2, 1.8]
beta_windows = [30, 60]

out_dir = ROOT / 'results' / 'grid_search_yearly'
out_dir.mkdir(parents=True, exist_ok=True)
summary_csv = out_dir / 'grid_results.csv'

rows = []
orig_Prod = production.ProductionConfig

total = len(years) * len(beta_coefs) * len(beta_windows)
count = 0
for year in years:
    start_date = f"{year:04d}-01-01"
    for bc in beta_coefs:
        for bw in beta_windows:
            count += 1
            print(f"[{count}/{total}] Year={year}, topix_beta_coef={bc}, beta_window={bw}")
            production.ProductionConfig = functools.partial(orig_Prod, topix_beta_coef=bc, beta_window=bw)
            try:
                out = production.run_production(
                    start_date=start_date,
                    output_root=production.get_default_results_root(),
                    run_tag=f"grid_{year}_bc{bc}_bw{bw}",
                    skip_chart=True,
                )
                m = pd.read_csv(Path(out) / 'metrics.csv')
                met = m.iloc[0].to_dict()
                met.update({'year': year, 'topix_beta_coef': bc, 'beta_window': bw, 'out_dir': str(out)})
                rows.append(met)
            except Exception as e:
                print('Error for', year, bc, bw, e)
                rows.append({'year': year, 'topix_beta_coef': bc, 'beta_window': bw, 'error': str(e)})

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
print('Grid search completed. Summary saved to', summary_csv)
