import sys
import os
from pathlib import Path

# Ensure src on path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import data_loader
import production

DATA_DIR = ROOT / "data"
ETF_PKL = DATA_DIR / "etf_data.pkl"
DECISION_CACHE = DATA_DIR / "decision_cache.npz"

print('Touching etf_data.pkl to avoid re-downloading...')
try:
    if ETF_PKL.exists():
        os.utime(ETF_PKL, None)
        print(f'Touched: {ETF_PKL}')
except Exception as e:
    print('Failed to touch', ETF_PKL, e)

print('Removing existing decision cache if present...')
try:
    if DECISION_CACHE.exists():
        DECISION_CACHE.unlink()
        print(f'Removed: {DECISION_CACHE}')
except Exception as e:
    print('Failed to remove', DECISION_CACHE, e)

# Rebuild etf_data.pkl and decision cache
print('Downloading and rebuilding market data (this may take a few minutes)...')
data = data_loader.download_data(start_date='2009-01-01', beta_window=60)
print('Preprocessing execution dataframe...')
df_exec = data_loader.preprocess_data(data, beta_window=60)
print('Saving decision cache...')
path = data_loader.save_decision_cache(df_exec)
print('Decision cache saved at', path)

# Run production backtest with default ProductionConfig (topix_beta_coef should be non-zero)
print('Running production backtest with Proposal B enabled...')
out_dir = production.run_production(
    start_date='2015-01-01',
    output_root=production.get_default_results_root(),
    run_tag='test_proposal_b_rebuilt',
    skip_chart=True,
)
print('Backtest output:', out_dir)
