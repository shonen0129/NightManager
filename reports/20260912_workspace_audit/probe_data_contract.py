"""Offline configuration and price-contract checks."""
from pathlib import Path
import json
import logging
import sys
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from leadlag.data.preprocessor import preprocess_data
from leadlag.data.tickers import US_TICKERS,JP_TICKERS,TOPIX_TICKER
from leadlag.execution.config import load_config_from_yaml
from leadlag.models.production_v2 import parse_run_config
from leadlag.execution.v2_bridge import _resolve_current_prices
logging.basicConfig(level=logging.ERROR)
cfg=load_config_from_yaml(ROOT/'configs/production/production.yaml',strict=True)
dates=pd.bdate_range('2026-06-01',periods=80)
us=pd.DataFrame({tk:100+np.arange(80) for tk in US_TICKERS},index=dates)
jp=pd.DataFrame({tk:1000+np.arange(80) for tk in JP_TICKERS+[TOPIX_TICKER]},index=dates)
try:
    preprocess_data({'us_close':us,'jp_close':jp,'jp_open':jp-.5},beta_window=2,strict_validation=True)
    strict={'raised':False}
except Exception as exc:strict={'raised':True,'error':str(exc),'all_raw_prices_finite_positive':True}
class PriceBroker:
    def __init__(self):self.calls=[]
    def fetch_open_prices(self,tickers,**kwargs):self.calls.append('open');return {tk:1000. for tk in tickers}
    def fetch_current_prices(self,tickers,**kwargs):self.calls.append('current');return {tk:1050. for tk in tickers}
broker=PriceBroker()
prices=_resolve_current_prices(cfg,broker,None,False)
raw=yaml.safe_load((ROOT/'configs/production/production.yaml').read_text())
upstream=parse_run_config(raw)
params=['rho','alpha_xx','alpha_yx','lambda_pca','lambda_sector','beta_conf','winsor_sigma','blp_window','asymmetry_delta','frac_diff_enabled','copula_enabled']
config_diff={k:{'production':getattr(cfg.v2.blpx,k),'step2':getattr(upstream.blpx,k)} for k in params if getattr(cfg.v2.blpx,k)!=getattr(upstream.blpx,k)}
print(json.dumps({'strict_clean_data_rejected':strict,'decision_price':{'methods_called':broker.calls,'resolved_price':prices[JP_TICKERS[0]],'current_price':1050.,'open_price':1000.},'upstream_config_differences':config_diff,'step2_baseline_ir':{'minvar_enabled':raw.get('portfolio',{}).get('minvar_enabled',False),'minvar_alpha':raw.get('portfolio',{}).get('minvar_alpha',.5),'mh_enabled':raw.get('multi_horizon_blend',{}).get('enabled',False),'cs_enabled':raw.get('cs_feature_overlay',{}).get('enabled',False)}},ensure_ascii=False,indent=2))
