"""Offline review probes; production data is read through read-only SQLite."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import sqlite3

import numpy as np
import pandas as pd

from leadlag.config.schemas import ProductionV2RunConfig
from leadlag.data.cache_store import SqliteCacheStore
from leadlag.data.intraday_inputs import build_open_910_returns, has_valid_open_910_returns
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.state_store import ExecutionStateStore
from leadlag.models.v2.distribution_source import FileCacheDistributionSource, OnDemandDistributionSource
from leadlag.utils.gap_matrix_io import save_gap_matrices
from leadlag.utils.gap_provenance import bundle_identity


def read_local(path, key):
    with sqlite3.connect(f'file:{path.resolve()}?mode=ro', uri=True) as conn:
        blob = conn.execute('SELECT value FROM cache_store WHERE key = ?', (key,)).fetchone()[0]
    return SqliteCacheStore._restore_from_storage_static(json.loads(blob))


def main():
    output = {}
    frame = read_local(Path('var/market_data/df_exec.sqlite'), 'df_exec')
    bars = read_local(Path('var/market_data/etf_prices.sqlite'), 'intraday_5m')
    open_910 = build_open_910_returns(frame, JP_TICKERS, df_5m=bars)
    complete = np.isfinite(open_910.to_numpy()).all(axis=1)
    model = SimpleNamespace(run_config=ProductionV2RunConfig(), n_j=len(JP_TICKERS), _blpx_model=object())
    output['actual_910_coverage'] = {
        'history_rows': len(frame), 'history_start': str(frame.index.min()),
        'history_end': str(frame.index.max()), 'complete_910_rows': int(complete.sum()),
        'missing_cells': int(open_910.isna().sum().sum()),
        'valid_strict_input': has_valid_open_910_returns(open_910, frame),
        'latest_complete_date': str(frame.index[complete][-1]) if complete.any() else None,
    }
    # Use a day with complete current observations, so rejection is solely historical.
    if complete.any():
        date = frame.index[complete][-1]
        history = frame.loc[:date]
        args = dict(trade_date=str(date.date()), df_exec=history, current_prices={},
                    open_910_returns=open_910, allow_implicit_io=False)
        file_model = SimpleNamespace(run_config=ProductionV2RunConfig(gap_input_dir=Path('/tmp/probe-unused-gap')), n_j=len(JP_TICKERS), _blpx_model=object())
        output['actual_910_coverage']['source_reasons_on_complete_date'] = {
            'file_cache': FileCacheDistributionSource(file_model).resolve(**args).reason.value,
            'on_demand': OnDemandDistributionSource(file_model).resolve(**args).reason.value,
        }
    with TemporaryDirectory(prefix='leadlag-review-state-') as folder:
        store = ExecutionStateStore(Path(folder) / 'state.sqlite')
        common = dict(account_key='test-account', strategy_key='production_v2', trade_date='2026-09-18')
        prepared = store.prepare_run(**common, job_type='close')
        other = store.prepare_run(**common, job_type='decision')
        store.mark_submission_started(other.run_id)
        store.mark_reconciliation_required(other.run_id, error='unknown broker response')
        resumed = store.prepare_run(**common, job_type='close')
        store.mark_submission_started(resumed.run_id)
        output['prepared_resume_bypass'] = {
            'resumed_same_run': resumed.run_id == prepared.run_id,
            'unresolved_count': len(store.list_recovery_candidates()),
            'statuses': sorted(run.status for run in store.list_recovery_candidates()),
        }
    with TemporaryDirectory(prefix='leadlag-review-bundle-') as folder:
        root = Path(folder)
        cfg = ProductionV2RunConfig(gap_input_dir=root)
        model = SimpleNamespace(run_config=cfg, n_j=len(JP_TICKERS), _blpx_model=None)
        date = '2026-09-18'
        sample = pd.DataFrame({'sig_date': ['2026-09-17'], **{f'jp_gap_{tk}': [0.01] for tk in JP_TICKERS}}, index=pd.DatetimeIndex([date]))
        sample_910 = pd.DataFrame(0.0, index=sample.index, columns=JP_TICKERS)
        metadata = dict(sig_date='2026-09-17', trade_date=date, horizon=1,
                        **bundle_identity(sample, date, config=cfg, open_910_returns=sample_910))
        save_gap_matrices(root, date, np.ones(len(JP_TICKERS))*0.01, np.eye(len(JP_TICKERS))*0.001, metadata=metadata)
        result = FileCacheDistributionSource(model).resolve(date, sample, {},
            snapshot=SimpleNamespace(jp_gap_returns=np.full(len(JP_TICKERS), 0.20), jp_betas=np.ones(len(JP_TICKERS)), topix_night_return=0.0),
            open_910_returns=sample_910, allow_implicit_io=False)
        output['changed_snapshot_cache'] = {'cached_gap': 0.01, 'decision_gap': 0.20,
                                           'source_status': result.status.value, 'source_reason': result.reason.value}
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
