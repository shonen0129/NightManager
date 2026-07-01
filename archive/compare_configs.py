import sys, time
sys.path.insert(0, 'src')
import yaml
import pandas as pd
from leadlag.data.fetcher import download_data
from leadlag.data.preprocessor import preprocess_data
from leadlag.models.sector_relative_ensemble_blp_enhanced import SectorRelativeEnsembleBLPEnhancedModel
from leadlag.execution.backtester import BacktestEngine
from leadlag.reporting.metrics import calculate_metrics


def run_case(cfg, label, overnight_alpha_long=0.0, overnight_alpha_short=0.0):
    model = SectorRelativeEnsembleBLPEnhancedModel(cfg)
    results = BacktestEngine.run_backtest(
        model, df_exec=df_exec, start_date='2015-01-05',
        overnight_alpha_long=overnight_alpha_long,
        overnight_alpha_short=overnight_alpha_short,
        buy_interest_annual=0.025, borrow_fee_annual=0.0115, reverse_fee_bps=2.0,
    )
    metrics = calculate_metrics(results['daily_returns'])
    metrics['avg_turnover'] = float(results['daily_turnover'].mean())
    metrics['avg_cost'] = float(results['daily_costs'].mean())
    return {
        'label': label,
        'AR': metrics['AR'],
        'RISK': metrics['RISK'],
        'Sharpe': metrics['Sharpe'],
        'MDD': metrics['MDD'],
        'turnover': metrics['avg_turnover'],
        'cost': metrics['avg_cost'],
    }


if __name__ == '__main__':
    data = download_data(beta_window=60)
    global df_exec
    df_exec = preprocess_data(data, beta_window=60)

    with open('configs/production.yaml') as f:
        prod_cfg = yaml.safe_load(f)

    results = []

    # Case 1: Production config as baseline (residual_blpx only, no continuous M)
    cfg = prod_cfg.copy()
    cfg.setdefault('blpx', {})['sector_eta'] = 0.0
    results.append(run_case(cfg, 'prod_fixedM_residBLPX'))

    # Case 2: Production config + continuous M (eta=0.5)
    cfg = prod_cfg.copy()
    cfg.setdefault('blpx', {})['sector_eta'] = 0.5
    cfg['blpx']['sector_gamma'] = 2.0
    results.append(run_case(cfg, 'prod_continuousM_residBLPX'))

    # Case 3: Add raw_blpx 0.5 + residual_blpx 0.5
    cfg = prod_cfg.copy()
    cfg.setdefault('blpx', {})['sector_eta'] = 0.5
    cfg['blpx']['sector_gamma'] = 2.0
    cfg['signal_components'] = {
        'residual_blpx': {'enabled': True, 'weight': 0.5},
        'raw_blpx': {'enabled': True, 'weight': 0.5},
        'raw_pca': {'enabled': False, 'weight': 0.0},
        'residual_pca': {'enabled': False, 'weight': 0.0},
    }
    results.append(run_case(cfg, 'prod_continuousM_50raw_50resid'))

    # Case 4: Only raw_blpx
    cfg = prod_cfg.copy()
    cfg.setdefault('blpx', {})['sector_eta'] = 0.5
    cfg['blpx']['sector_gamma'] = 2.0
    cfg['signal_components'] = {
        'raw_blpx': {'enabled': True, 'weight': 1.0},
        'residual_blpx': {'enabled': False, 'weight': 0.0},
        'raw_pca': {'enabled': False, 'weight': 0.0},
        'residual_pca': {'enabled': False, 'weight': 0.0},
    }
    results.append(run_case(cfg, 'prod_continuousM_onlyRawBLPX'))

    # Case 5: Production config with overnight holding from YAML
    cfg = prod_cfg.copy()
    cfg.setdefault('blpx', {})['sector_eta'] = 0.5
    cfg['blpx']['sector_gamma'] = 2.0
    results.append(run_case(cfg, 'prod_continuousM_overnight75_50', overnight_alpha_long=0.75, overnight_alpha_short=0.5))

    # Case 6: Production with eta=0.0, overnight holding
    cfg = prod_cfg.copy()
    cfg.setdefault('blpx', {})['sector_eta'] = 0.0
    results.append(run_case(cfg, 'prod_fixedM_overnight75_50', overnight_alpha_long=0.75, overnight_alpha_short=0.5))

    print()
    df = pd.DataFrame(results)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print()
    print('Best Sharpe:', df.loc[df['Sharpe'].idxmax(), 'label'])
    print('Best MDD:', df.loc[df['MDD'].idxmax(), 'label'])  # MDD is negative, max = least negative
