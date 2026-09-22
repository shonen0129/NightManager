"""Offline follow-up probes for A-C. Real brokers and live stores are not used."""
from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from leadlag.broker.tachibana.client import TachibanaBrokerClient
from leadlag.core.types import OrderResult, OrderStatus
from leadlag.data.gap_store import GapStore
from leadlag.data.preprocessor import preprocess_data
from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER, US_TICKERS
from leadlag.execution import broker_ops, close, post_decision, pricing
from leadlag.execution.config import load_config_from_yaml
from leadlag.models.ml_order_overlay import (
    MLOrderOverlayModel,
    load_overlay_model,
    save_overlay_model,
)
from leadlag.models.production_v2 import ProductionV2Model
from leadlag.runner.production import ProductionRunner
from leadlag.utils.gap_matrix_io import load_gap_bundle

logging.basicConfig(level=logging.ERROR)
out = {}
cfg = load_config_from_yaml(ROOT / 'configs/production/production.yaml', strict=True)

def probe(name, fn):
    try:
        out[name] = fn()
    except Exception as exc:
        out[name] = {'probe_error': str(exc), 'type': type(exc).__name__}

def horizon_metadata():
    with tempfile.TemporaryDirectory(prefix='abc-rereview-meta-') as tmp:
        path = Path(tmp) / 'gap.sqlite'
        store = GapStore(path)
        rc = cfg.v2.model_copy(deep=True, update={
            'macro_kappa_enabled': False, 'macro_direction_enabled': False,
            'cs_overlay_enabled': False, 'ml_overlay_enabled': False,
            'gap_input_dir': str(path),
        })
        values = []
        for sign in (1, -1):
            for h in (None, 3, 5):
                mu = np.linspace(-0.03, 0.03, len(JP_TICKERS))
                if h == 3:
                    mu = sign * mu[::-1] * 10
                store.save_horizon('2026-08-14', mu, np.eye(len(JP_TICKERS)) * .001,
                    metadata={'sig_date': '2026-08-17' if h == 3 else '2026-08-13'}, horizon=h)
            model = ProductionV2Model(rc, blpx_model=SimpleNamespace())
            df = pd.DataFrame(index=pd.to_datetime(['2026-08-14']))
            result = model.decide('2026-08-14', gap_input_dir=path, df_exec=df,
                current_prices={tk:1000. for tk in JP_TICKERS})
            values.append({'audit':result['leakage'], 'fallback':result['fallback'],
                'diagnostics':result.get('diagnostics'),
                'gross':float(np.abs(result['w_final']).sum()), 'scores':result['scores'].tolist()})
        return {'h3_sig_date':'2026-08-17', 'trade_date':'2026-08-14',
                'case_1':values[0], 'case_2':values[1],
                'max_score_change':float(np.max(np.abs(np.array(values[0]['scores'])-values[1]['scores'])))}
probe('future_h3_metadata_flats', horizon_metadata)

def atomic_reader():
    results={}
    for h in (None,3,5):
        with tempfile.TemporaryDirectory(prefix='abc-rereview-atomic-') as tmp:
            path=Path(tmp)/'gap.sqlite'
            writer=GapStore(path)
            writer.save_horizon('2026-08-14',np.ones(17),np.eye(17),{'version':'old'},horizon=h)
            from leadlag.data.cache_store import SqliteCacheStore
            original=SqliteCacheStore._get_with_conn
            swapped=[]
            def read_then_update(self,conn,key,default=None):
                value=original(self,conn,key,default)
                if not swapped:
                    swapped.append(True)
                    writer.save_horizon('2026-08-14',np.ones(17)*2,np.eye(17)*2,{'version':'new'},horizon=h)
                return value
            kwargs={} if h is None else {'mu_pattern':'matrices/mu_gap_h{h}_{date}.npy',
                'omega_pattern':'matrices/omega_gap_h{h}_{date}.npy','pattern_kwargs':{'h':h}}
            with patch.object(SqliteCacheStore,'_get_with_conn',read_then_update):
                mu,omega,meta,alerts=load_gap_bundle(path,'2026-08-14',strict=True,**kwargs)
            results[str(h)]={'interleaved_commit':bool(swapped),'mu':float(mu[0]),
                'omega':float(omega[0,0]),'meta':meta,'latest_mu':float(writer.load_horizon('2026-08-14',horizon=h)[0][0])}
    return results
probe('atomic_reader_passes_all_horizons',atomic_reader)

def padded_holiday():
    dates = pd.bdate_range('2026-08-03', '2026-08-14')
    us = pd.DataFrame(100., index=dates, columns=US_TICKERS)
    jp = pd.DataFrame(100., index=dates, columns=JP_TICKERS+[TOPIX_TICKER])
    jp.loc['2026-08-11'] = np.nan
    raw = {'us_close':us, 'jp_close':jp, 'jp_open':jp.copy()}
    cases = {}
    for strict in (False, True):
        try:
            value = preprocess_data(raw, beta_window=2, strict_validation=strict)
            cases[str(strict)] = {'dates':[str(d.date()) for d in value.index],
                'contains_20260813':pd.Timestamp('2026-08-13') in value.index}
        except ValueError as exc:
            cases[str(strict)] = {'error':str(exc)}
    clean = {key:frame.dropna(how='all') for key,frame in raw.items()}
    value = preprocess_data(clean, beta_window=2, strict_validation=True)
    cases['market_sessions_only'] = {'dates':[str(d.date()) for d in value.index],
        'contains_20260813':pd.Timestamp('2026-08-13') in value.index}
    return cases
probe('nontrading_nan_row_loses_next_return', padded_holiday)

def failure_journal():
    called = []
    class PartialBroker:
        def submit_orders_batch(self, orders, **kwargs):
            return [OrderResult(order_id=f'fake-{i}', status=OrderStatus.PARTIALLY_FILLED,
                ticker=o.ticker, side=o.side, quantity=o.quantity) for i,o in enumerate(orders)]
    decision = {'trade_date':pd.Timestamp('2026-08-14')}
    with tempfile.TemporaryDirectory(prefix='abc-rereview-journal-') as tmp:
        with patch.object(post_decision, 'save_decision_output',return_value='fake.csv'), \
             patch.object(broker_ops, 'split_large_orders',side_effect=lambda orders:(orders,[])), \
             patch.object(broker_ops, '_wait_for_fills_sync',side_effect=lambda api,results,**kwargs:(results,True)), \
             patch.object(post_decision, 'fetch_fill_prices', side_effect=lambda *a,**kw:called.append('fills')), \
             patch.object(post_decision, 'save_position_snapshot', side_effect=lambda *a,**kw:called.append('positions')), \
             patch.object(post_decision, 'save_wallet_snapshot', side_effect=lambda *a,**kw:called.append('wallet')), \
             patch.object(post_decision, 'save_daily_journal', side_effect=lambda *a,**kw:called.append('journal')):
            try:
                frame=pd.DataFrame({'ticker':['1617.T','1618.T'],'quantity':[100,100],'action':['BUY','SELL']})
                post_decision._write_decision_output_and_submit(frame,decision,tmp,False,PartialBroker(),{})
            except RuntimeError as exc:
                summary=json.loads((Path(tmp)/'api_execution_log.json').read_text())
                rows=summary['buy_results']+summary['sell_results']
                return {'raised':str(exc),'post_failure_collection':called,
                    'persisted_statuses':[r['status'] for r in rows],
                    'persisted_fill_quantities':[r.get('fill_quantity') for r in rows]}
probe('partial_failure_skips_reconciliation', failure_journal)

def cancelled_fill_collection():
    calls=[]
    api=object.__new__(TachibanaBrokerClient)
    api._client=SimpleNamespace(get_order_detail=lambda *a: calls.append(a) or {'sYakuzyouPrice':'1000','sYakuzyouSuryou':'30'})
    rows=[{'order_id':'fake','status':'CANCELLED','ticker':'1617.T','eigyou_day':'20260814'}]
    pricing.fetch_fill_prices(api,rows,wait_seconds=0.)
    return {'calls':len(calls),'result':rows[0]}
probe('cancelled_partial_not_collected',cancelled_fill_collection)

def rejected_close_exit():
    from leadlag.cli import main
    data={'close_incomplete':True,'filled_orders_count':0,'pending_orders_count':0,
        'failed_orders_count':1,'close_results':[{'status':'FAILED','ticker':'1617.T','quantity':100}]}
    with tempfile.TemporaryDirectory(prefix='abc-rereview-close-') as tmp:
        fake=SimpleNamespace(close=lambda:None)
        with patch('leadlag.core.market_calendar.is_market_closed',return_value=False), \
             patch.object(close,'build_api_client',return_value=fake), \
             patch.object(close,'build_output_dir',return_value=tmp), \
             patch.object(close,'close_all_positions',return_value=data) as close_call, \
             patch.object(close,'save_position_snapshot',return_value=None), \
             patch.object(close,'save_wallet_snapshot',return_value=None), \
             patch.object(close,'save_daily_journal',return_value=None):
            code=main(['close'])
            assert close_call.call_count == 1, 'CLI must reach the close execution boundary'
    return {'close_incomplete':True,'all_orders_rejected':True,'cli_return_code':code,
        'close_execution_calls':close_call.call_count,'holiday_skip_disabled':True}
probe('all_rejected_close_cli_success',rejected_close_exit)

def metadata_commit():
    model=MLOrderOverlayModel(lgbm=SimpleNamespace(marker='trained-through-2026'),cont_cols=[],target_std=1.,
        use_ticker=False,use_classification=False,per_ticker_interactions=False)
    def metadata(end):return {'train_start':'2015-01-05','train_end':end,'data_hash':'fake-'+end,'config_hash':'fake','metadata_status':'verified'}
    with tempfile.TemporaryDirectory(prefix='abc-rereview-ml-') as tmp:
        path=Path(tmp)
        save_overlay_model(model,path,training_metadata=metadata('2026-08-14'))
        import leadlag.models.ml_order_overlay as overlay
        previous=(path/'CURRENT').read_text(encoding='utf-8').strip()
        new_model=MLOrderOverlayModel(lgbm=SimpleNamespace(marker='trained-through-2020'),cont_cols=[],target_std=1.,
            use_ticker=False,use_classification=False,per_ticker_interactions=False)
        original_replace=overlay.os.replace
        def fail_current(source,destination):
            if Path(destination)==path/'CURRENT':
                raise OSError('injected publication failure')
            return original_replace(source,destination)
        try:
            with patch.object(overlay.os,'replace',fail_current):
                save_overlay_model(new_model,path,training_metadata=metadata('2020-12-31'))
        except OSError:
            pass
        loaded=load_overlay_model(path)
        active=(path/'CURRENT').read_text(encoding='utf-8').strip()
        return {'publication_failed':True,'loaded_marker':loaded.lgbm.marker,
                'loaded_train_end':loaded.metadata['train_end'],
                'active_version_unchanged':active==previous,'load_accepted':True}
probe('overlay_atomic_publish_failure',metadata_commit)

def runner_ready():
    try:
        ProductionRunner(cfg)
    except Exception as exc:
        return {'constructs':False,'error':str(exc),'type':type(exc).__name__}
    return {'constructs':True}
probe('configured_runner',runner_ready)

print(json.dumps(out,ensure_ascii=False,indent=2,default=str))
