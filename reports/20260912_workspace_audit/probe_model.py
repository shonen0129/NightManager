"""Offline model checks using synthetic history, fixed input snapshots, and fresh instances."""
import json
import logging
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from leadlag.execution.config import load_config_from_yaml
from leadlag.models.blpx import ProductionBLPXModel
from leadlag.models.production_v2 import ProductionV2Model
from leadlag.data.pit_lake import PITDataLake
from leadlag.data.tickers import JP_TICKERS,US_TICKERS
logging.basicConfig(level=logging.ERROR)
cfg=load_config_from_yaml(ROOT/'configs/production/production.yaml',strict=True)
rng=np.random.default_rng(20260912)
dates=pd.bdate_range('2009-01-05','2017-01-31')
n=len(dates)
cols={'sig_date':dates-pd.Timedelta(days=1),'is_provisional':np.zeros(n),'topix_oc_return':rng.normal(0,.01,n),'topix_night_return':rng.normal(0,.005,n)}
for tk in US_TICKERS:cols[f'us_cc_{tk}']=rng.normal(0,.01,n)
for tk in JP_TICKERS:
    oc=rng.normal(0,.01,n);gap=rng.normal(0,.004,n)
    close=1000*np.cumprod((1+oc)*(1+gap));prev=np.r_[1000,close[:-1]]
    cols[f'jp_oc_{tk}']=oc;cols[f'jp_open_trade_{tk}']=prev*(1+gap)
    cols[f'jp_close_sig_{tk}']=prev;cols[f'jp_gap_{tk}']=gap;cols[f'jp_beta_{tk}']=np.ones(n)
df=pd.DataFrame(cols,index=dates)
date='2016-08-15'
i=df.index.get_loc(date)
# Disable macro only in this isolated numerical probe; the production setting is unchanged.
rc=cfg.v2.model_copy(deep=True,update={'macro_kappa_enabled':False,'macro_direction_enabled':False,'cs_overlay_enabled':False,'ml_overlay_enabled':False,'gap_input_dir':None})
def model():return ProductionV2Model(rc,blpx_model=ProductionBLPXModel(rc.blpx))
def dist(m,frame,h):
    lake=PITDataLake(frame);snap=lake.get_snapshot(date)
    return m.compute_distribution(date,frame,snap.current_prices,horizon=h,use_file_cache=False,snapshot=snap)
with patch('leadlag.data.cache.load_intraday_cache',return_value=None):
    base=model();mu1,om1=dist(base,df,1)
    mu3_reused,om3_reused=dist(base,df,3)
    mu3_fresh,om3_fresh=dist(model(),df,3)
    altered=df.copy()
    for tk in JP_TICKERS:
        altered.loc[date:,f'jp_oc_{tk}']+=.25
    # Keep today's known gap and US input intact; alter strictly future known-input rows as well.
    for tk in US_TICKERS:altered.iloc[i+1:,altered.columns.get_loc(f'us_cc_{tk}')]+=.2
    ma,oa=dist(model(),altered,1)
    result={'h3_reused_vs_fresh':{'max_abs_mu_difference':float(np.max(np.abs(mu3_reused-mu3_fresh))),'relative_covariance_difference':float(np.linalg.norm(om3_reused-om3_fresh)/np.linalg.norm(om3_fresh))},'future_target_perturbation_h1':{'max_abs_mu_difference':float(np.max(np.abs(ma-mu1))),'max_abs_covariance_difference':float(np.max(np.abs(oa-om1))),'dates':n,'as_of':date}}
    for horizon in (3,5):
        original_mu,original_omega=dist(model(),df,horizon)
        changed_mu,changed_omega=dist(model(),altered,horizon)
        result[f'future_target_perturbation_h{horizon}']={
            'max_abs_mu_difference':float(np.max(np.abs(changed_mu-original_mu))),
            'max_abs_covariance_difference':float(np.max(np.abs(changed_omega-original_omega))),
            'dates':n,'as_of':date,
        }
print(json.dumps(result,indent=2))
