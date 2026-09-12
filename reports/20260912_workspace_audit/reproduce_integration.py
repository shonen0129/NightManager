"""Offline boundary probes with fake brokers and temporary output stores."""
from pathlib import Path
import json
import logging
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from leadlag.execution.config import load_config_from_yaml,build_app_config_from_dict
from leadlag.execution import broker_ops,post_decision,v2_bridge,var_history
from leadlag.execution.backtester import BacktestEngine
from leadlag.core.types import OrderResult,OrderStatus
from leadlag.data.tickers import JP_TICKERS,US_TICKERS,TOPIX_TICKER
from leadlag.data.preprocessor import preprocess_data
from leadlag.data.cache_store import SqliteCacheStore
logging.basicConfig(level=logging.ERROR)
cfg=load_config_from_yaml(ROOT/'configs/production/production.yaml',strict=True)
out={}
def probe(name,fn):
    try:out[name]=fn()
    except Exception as exc:out[name]={'probe_error':type(exc).__name__,'message':str(exc)}

def inherited():
    raw=yaml.safe_load((ROOT/'configs/production/production.yaml').read_text())
    risk_cfg=build_app_config_from_dict(raw)
    keys=['mh_blend_enabled','ml_overlay_enabled','macro_kappa_enabled','frac_diff_enabled','gap_input_dir']
    return {k:{'production':getattr(cfg.v2,k),'risk_rebuild':getattr(risk_cfg.v2,k)} for k in keys}
probe('var_history_unresolved_base',inherited)

class RejectingBroker:
    def submit_orders_batch(self,orders,**kwargs):
        return [OrderResult(order_id='',status=OrderStatus.FAILED,ticker=o.ticker,side=o.side,quantity=o.quantity,message='fake rejection') for o in orders]

def failed_orders():
    orders=pd.DataFrame({'ticker':['1617.T','1618.T'],'quantity':[1,1],'action':['BUY','SELL']})
    with tempfile.TemporaryDirectory(prefix='leadlag-audit-orders-') as tmp:
        summary=broker_ops.submit_orders_via_api(orders,RejectingBroker(),tmp,{})
        return {'function_returned_normally':True,**{k:summary[k] for k in ['expected_orders_count','submitted_orders_count','failed_orders_count','first_batch_failed']},'statuses':[r['status'] for r in summary['buy_results']+summary['sell_results']]}
probe('all_orders_rejected_reported_success',failed_orders)

def risk_flat():
    d={'trade_date':pd.Timestamp('2026-08-17'),'tickers':JP_TICKERS,'weight':np.zeros(17),'signal':np.zeros(17),'action':['HOLD']*17}
    h=pd.Series([-.10],index=pd.to_datetime(['2026-08-14']))
    with tempfile.TemporaryDirectory(prefix='leadlag-audit-flat-') as tmp:
        with patch.object(post_decision,'_write_decision_output_and_submit') as submit:
            try:post_decision.execute_post_decision_flow(d,cfg.strategy,{tk:1000. for tk in JP_TICKERS},300000,h,tmp,RejectingBroker(),current_positions={'1617.T':10})
            except RuntimeError as exc:return {'blocked':True,'submit_called':submit.called,'error':str(exc)}
    return {'blocked':False}
probe('risk_stop_blocks_position_reduction',risk_flat)

def stale_day():
    dates=pd.to_datetime(['2026-08-13','2026-08-14'])
    columns={}
    for tk in US_TICKERS:columns[f'us_cc_{tk}']=[.001,.001]
    for tk in JP_TICKERS:
        columns[f'jp_open_trade_{tk}']=[1000.,1000.]
        columns[f'jp_close_sig_{tk}']=[1000.,1000.]
        columns[f'jp_beta_{tk}']=[1.,1.]
        columns[f'jp_gap_{tk}']=[0.,0.]
    columns['topix_night_return']=[0.,0.]
    df=pd.DataFrame(columns,index=dates)
    class Runner:
        def __init__(self,*args):pass
        def run(self,inputs):
            out['stale_date_selected']={'requested':'2026-08-17','runner_trade_date':inputs.trade_date,'snapshot_date':inputs.snapshot.trade_date,'gap_dir':str(inputs.gap_input_dir)}
            return {'fallback':{'gap_data_missing':True}}
    with tempfile.TemporaryDirectory(prefix='leadlag-audit-stale-') as tmp:
        with patch.object(v2_bridge,'_load_df_exec',return_value=df),patch.object(v2_bridge,'ProductionRunner',Runner),patch.object(v2_bridge,'write_production_files'):
            v2_bridge.run_v2_decision(ROOT/'configs/production/production.yaml',trade_date='2026-08-17',live_dir=tmp,dry_run=True,api_enable=False)
    return out['stale_date_selected']
probe('stale_date_replay',stale_day)

def holiday_alignment():
    dates=pd.to_datetime(['2026-08-07','2026-08-10','2026-08-11','2026-08-12','2026-08-13','2026-08-14'])
    jpdates=dates[dates!=pd.Timestamp('2026-08-11')]
    us=pd.DataFrame({tk:[100.,100.,110.,110.,110.,110.] for tk in US_TICKERS},index=dates)
    jp=pd.DataFrame(1000.,index=jpdates,columns=JP_TICKERS+[TOPIX_TICKER])
    df=preprocess_data({'us_close':us,'jp_close':jp,'jp_open':jp},beta_window=2)
    return {str(d.date()):{'sig_date':str(df.loc[d,'sig_date'].date()),'us_return':float(df.loc[d,f'us_cc_{US_TICKERS[0]}'])} for d in pd.to_datetime(['2026-08-12','2026-08-13'])}
probe('japan_holiday_lags_us_signal',holiday_alignment)

# Capture which overlay object a normal engine call forwards; no calculation or I/O.
def overlay_default():
    sentinel=RuntimeError('captured')
    with patch.object(BacktestEngine,'_resolve_v2_backtest_cost_params',return_value={'slip_bps':5.,'alpha_long':.75,'alpha_short':.5,'fin_annual':.025,'borrow_annual':.0115,'rev_bps':2.,'side_leverage':1.5}),patch.object(BacktestEngine,'_resolve_sim_dates',return_value=(pd.bdate_range('2026-08-13',periods=2),0,1)),patch.object(BacktestEngine,'_compute_target_and_gap_returns',return_value=(np.zeros((2,17)),np.zeros((2,17)))),patch.object(BacktestEngine,'_generate_v2_weights',side_effect=sentinel) as gen:
        try:BacktestEngine.run_v2_backtest(cfg,None,pd.DataFrame(index=pd.bdate_range('2026-08-13',periods=2)))
        except RuntimeError as exc:
            if exc is not sentinel:raise
        return {'ml_overlay_enabled':cfg.v2.ml_overlay_enabled,'model_dir':cfg.v2.ml_overlay_model_dir,'overlay_model_forwarded':str(gen.call_args.args[5])}
probe('default_backtest_omits_overlay',overlay_default)
print(json.dumps(out,ensure_ascii=False,indent=2,default=str))
