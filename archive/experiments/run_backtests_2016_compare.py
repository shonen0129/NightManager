import sys
from pathlib import Path
import functools

# ensure src
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))

import production

orig_Prod = production.ProductionConfig

out_dirs = {}

# Baseline: topix_beta_coef = 0
production.ProductionConfig = functools.partial(orig_Prod, topix_beta_coef=0.0)
print('Running baseline (topix_beta_coef=0) start_date=2016-01-01')
out_dirs['baseline'] = production.run_production(
    start_date='2016-01-01',
    output_root=production.get_default_results_root(),
    run_tag='baseline_2016',
    skip_chart=True,
)
print('Baseline done:', out_dirs['baseline'])

# Proposal B: topix_beta_coef = default (1.2)
production.ProductionConfig = functools.partial(orig_Prod, topix_beta_coef=1.2)
print('Running Proposal B (topix_beta_coef=1.2) start_date=2016-01-01')
out_dirs['proposal_b'] = production.run_production(
    start_date='2016-01-01',
    output_root=production.get_default_results_root(),
    run_tag='proposal_b_2016',
    skip_chart=True,
)
print('Proposal B done:', out_dirs['proposal_b'])

# Restore
production.ProductionConfig = orig_Prod

# Print metrics for both
import pandas as pd
for k, d in out_dirs.items():
    m = pd.read_csv(Path(d) / 'metrics.csv')
    print(f"\n{k}: {d}")
    print(m.to_string(index=False))
