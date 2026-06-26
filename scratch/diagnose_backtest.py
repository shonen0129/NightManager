import sys, time
sys.path.insert(0, 'src')
import yaml
from leadlag.data.fetcher import download_data
from leadlag.data.preprocessor import preprocess_data
from leadlag.models.sector_relative_ensemble_blp_enhanced import SectorRelativeEnsembleBLPEnhancedModel
from leadlag.execution.backtester import BacktestEngine
from leadlag.reporting.metrics import calculate_metrics


def log(msg):
    print(f"[{time.time()-t0:.1f}s] {msg}", flush=True)


t0 = time.time()

with open('configs/production.yaml') as f:
    cfg = yaml.safe_load(f)
cfg.setdefault('blpx', {})['sector_eta'] = 0.5
cfg['blpx']['sector_gamma'] = 2.0

log('downloading/preprocessing data')
data = download_data(beta_window=60)
df_exec = preprocess_data(data, beta_window=60)
log(f'df_exec shape={df_exec.shape}')

log('initializing model')
model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
log(f'weights: raw_pca={model.raw_pca_weight}, res_pca={model.residual_pca_weight}, '
    f'raw_blpx={model.raw_blpx_weight}, res_blpx={model.residual_blpx_weight}')
log(f'exec_adjustment={model.exec_adjustment}')

log('running predict_signals')
pred = model.predict_signals(df_exec)
log('predict_signals done')

log('running backtest')
results = BacktestEngine.run_backtest(
    model, df_exec=df_exec, start_date='2015-01-05',
    overnight_alpha_long=0.0, overnight_alpha_short=0.0,
    buy_interest_annual=0.025, borrow_fee_annual=0.0115, reverse_fee_bps=2.0,
)
log('backtest done')

metrics = calculate_metrics(results['daily_returns'])
metrics['avg_turnover'] = float(results['daily_turnover'].mean())
metrics['avg_cost'] = float(results['daily_costs'].mean())

print()
print('=== Tikhonov + Continuous M_sector (production, eta=0.5, gamma=2.0) ===')
for k, v in metrics.items():
    if k in ['AR', 'RISK', 'MDD', 'Total Return']:
        print(f'  {k}: {v*100:.2f}%')
    elif k == 'Sharpe':
        print(f'  {k}: {v:.4f}')
    else:
        print(f'  {k}: {v:.4f}')

log('finished')
