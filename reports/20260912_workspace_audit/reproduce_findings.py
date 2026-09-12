"""Offline, isolated minimal reproductions; does not place orders or edit runtime data."""
from pathlib import Path
import json
import logging
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from leadlag.execution.config import load_config_from_yaml
from leadlag.execution.backtester import BacktestEngine
from leadlag.models.v2.audit_comparator import _run_safety_audits
from leadlag.models.blpx import ProductionBLPXModel
from leadlag.reporting.metrics import calculate_metrics
from leadlag.experiment_registry import compute_deflated_sharpe
from leadlag.data.gap_store import GapStore
from leadlag.data.tickers import JP_TICKERS
from research.experiment_utils import _extract_metrics

logging.basicConfig(level=logging.ERROR)
cfg = load_config_from_yaml(ROOT/'configs/production/production.yaml', strict=True)
out = {}

def probe(name, fn):
    try: out[name] = fn()
    except Exception as exc: out[name] = {'probe_error':type(exc).__name__, 'message':str(exc)}

def leakage_failure():
    w=np.zeros(17); w[0]=1.; w[1]=-1.
    r=_run_safety_audits(w,np.ones(17),np.ones(17)*.01,np.eye(17)*.001,np.ones(17)*.03,None,'2026-08-17','2026-08-17',cfg.v2,{'gap_data_missing':False},{'multiplier':1.,'assigned_bin':'Medium','threshold_low':0.,'threshold_high':1.},[],None,'probe','probe')
    return {'leakage':r['leakage'],'numerical':r['numerical']['status'],'gross':float(np.abs(r['w_final']).sum()),'fallback':r['fallback'],'alerts':r['alerts']}
probe('leakage_failure_keeps_weights', leakage_failure)

def pnl_cost():
    def case(w):
        w=np.array(w,dtype=float)
        r=BacktestEngine._simulate_daily_pnl(w,np.zeros_like(w),np.zeros_like(w),pd.bdate_range('2026-08-13',periods=len(w)),.0005,0.,0.,0.,.75,.5,1.5)
        held=np.zeros(w.shape[1]); correct=[]
        for row in w:
            a=np.where(row>0,.75,np.where(row<0,.5,0.))
            h=a*row
            correct.append(float(.0005*1.5*(np.abs(row-held).sum()+np.abs(row-h).sum())))
            held=h
        return {'reported_slip_bps':(np.array(r['slip_costs'])*10000).tolist(),'inventory_flow_slip_bps':(np.array(correct)*10000).tolist()}
    return {'flat':case([[1,-1],[0,0]]),'sign_flip':case([[1,-1],[-1,1]]),'unchanged':case([[1,-1],[1,-1]])}
probe('slippage_inventory_flow',pnl_cost)
probe('initial_loss_mdd',lambda:{'reported':calculate_metrics(pd.Series([-.1,0.],index=pd.bdate_range('2026-08-13',periods=2)))['MDD'],'expected':-.1})

def dsr():
    m={'net_sharpe':1.,'trials':1,'n_observations':252}
    daily_sr=1/np.sqrt(252)
    return {'inputs':m,'reported':compute_deflated_sharpe(m),'same_frequency_reference':float(norm.cdf(daily_sr*np.sqrt(251)/np.sqrt(1+.5*daily_sr**2)))}
probe('dsr_annualization',dsr)
probe('registry_metrics',lambda:_extract_metrics({'daily_returns':pd.Series([-.1,0.,.02,0.]),'daily_fallback':np.array([False,True,False,True])}))

def corr_cache():
    bc=cfg.v2.blpx.model_copy(deep=True,update={'copula_enabled':False})
    a=np.random.default_rng(7).normal(0,.01,(504,32)); b=a.copy(); b[:,15:]*=3
    shared=ProductionBLPXModel(bc)
    x=shared._estimate_correlation(a,504,True)
    stale=shared._estimate_correlation(b,504,True)
    fresh=ProductionBLPXModel(bc)._estimate_correlation(b,504,True)
    return {'cached_object_reused':stale is x,'reported_jp_sigma':float(stale[1][15]),'expected_jp_sigma':float(fresh[1][15]),'ratio':float(stale[1][15]/fresh[1][15])}
probe('correlation_cache_wrong_input',corr_cache)

def common_cache():
    model=ProductionBLPXModel(cfg.v2.blpx)
    a=pd.DataFrame({'a':[1.,2.]},index=pd.to_datetime(['2020-01-06','2020-01-07']))
    b=a.copy(); b.iloc[-1,0]=100.
    def build(df,target,**kwargs):return SimpleNamespace(to_dict=lambda:{'observed_last':float(df.iloc[-1,0])})
    with patch('leadlag.core.pipeline.build_common_inputs',side_effect=build) as mock:
        old=model._prepare_common_inputs(a,y_jp_target=np.zeros((2,17)))
        new=model._prepare_common_inputs(b,y_jp_target=np.ones((2,17)))
    return {'build_calls':mock.call_count,'same_object':old is new,'expected_last':100.,'observed_last':new['observed_last'],'target_remains_zero':bool(np.all(new['y_jp_target']==0))}
probe('common_inputs_cache_wrong_input',common_cache)

def gap_pair():
    with tempfile.TemporaryDirectory(prefix='leadlag-audit-gap-') as tmp:
        store=GapStore(Path(tmp)/'test.sqlite')
        store.save('2026-08-17',np.ones(17),np.eye(17),{'version':'old'})
        original=store.put
        def fail(date,typ,*args,**kwargs):
            if typ=='omega':raise RuntimeError('injected write failure')
            return original(date,typ,*args,**kwargs)
        try:
            with patch.object(store,'put',side_effect=fail):store.save('2026-08-17',np.ones(17)*2,np.eye(17)*2,{'version':'new'})
        except RuntimeError:pass
        mu,omega,meta=store.load('2026-08-17')
        return {'mu':float(mu[0]),'omega':float(omega[0,0]),'metadata':meta,'mixed_version':bool(mu[0]!=omega[0,0])}
probe('gap_pair_partial_commit',gap_pair)

def pit_multiplier():
    from leadlag.models.v2.fallback import _apply_pit_ruleD
    custom=cfg.v2.model_copy(deep=True,update={'fallback_multiplier':.25})
    w=np.zeros(17);w[:2]=[1,-1]
    _,binning,_,_=_apply_pit_ruleD(w,np.ones(17),np.eye(17),None,'2026-08-17',custom,[])
    return {'configured_fallback_multiplier':.25,'actual_multiplier':binning['multiplier'],'fallback_flag':binning['fallback_flag']}
probe('pit_fallback_multiplier_ignored',pit_multiplier)
print(json.dumps(out,ensure_ascii=False,indent=2))
