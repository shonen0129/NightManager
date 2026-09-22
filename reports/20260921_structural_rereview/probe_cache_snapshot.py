"""Compare a saved distribution and on-demand calculation after a known-input correction.

Historical 09:10 returns are explicitly zero in this synthetic comparison to
isolate snapshot provenance from the separate missing-history finding.
"""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import json

import numpy as np
import pandas as pd

from probe_inputs_and_state import read_local
from leadlag.data.pit_lake import PITDataLake
from leadlag.data.tickers import JP_TICKERS
from leadlag.domain.inputs import HistoricalInputs
from leadlag.execution.config import load_config_from_yaml
from leadlag.models.production_v2 import ProductionV2Model
from leadlag.models.v2.distribution_source import FileCacheDistributionSource, OnDemandDistributionSource
from leadlag.runner.model_factory import build_blpx_model
from leadlag.utils.gap_matrix_io import save_gap_matrices
from leadlag.utils.gap_provenance import bundle_identity


def main():
    config = load_config_from_yaml(Path('configs/production/production.yaml'), strict=True)
    frame = read_local(Path('var/market_data/df_exec.sqlite'), 'df_exec')
    date = frame.index[-1]
    datestr = date.strftime('%Y-%m-%d')
    snapshot = PITDataLake(frame).get_snapshot(date + pd.Timedelta(hours=9, minutes=10))
    open_910 = pd.DataFrame(0.0, index=frame.index, columns=JP_TICKERS)
    history = HistoricalInputs(frame, open_910_returns=open_910).calculation_frame(snapshot.as_of)
    model = ProductionV2Model(config.v2, blpx_model=build_blpx_model(config, clear_cache=True))
    on_demand = OnDemandDistributionSource(model)
    args = dict(trade_date=datestr, df_exec=history, current_prices=dict(snapshot.current_prices),
                open_910_returns=open_910, allow_implicit_io=False)
    base = on_demand.resolve(**args, snapshot=snapshot)
    assert base.is_available, base.alerts
    changed_gap = snapshot.jp_gap_returns.copy()
    changed_gap[0] += 0.01
    changed_prices = dict(snapshot.current_prices)
    changed_prices[JP_TICKERS[0]] = snapshot.prev_closes[JP_TICKERS[0]] * (1 + changed_gap[0])
    changed = replace(snapshot, jp_gap_returns=changed_gap, current_prices=changed_prices)
    with TemporaryDirectory(prefix='leadlag-review-corrected-gap-') as folder:
        path = Path(folder)
        metadata = {**base.metadata, **bundle_identity(history, datestr, config=config.v2, open_910_returns=open_910)}
        save_gap_matrices(path, datestr, base.mu_gap, base.Omega_gap, metadata=metadata)
        cached = FileCacheDistributionSource(model, gap_input_dir=path).resolve(**args, snapshot=changed)
        fresh = on_demand.resolve(**args, snapshot=changed)
        assert cached.is_available and fresh.is_available
        assert not np.allclose(cached.mu_gap, fresh.mu_gap)
        print(json.dumps({'date': datestr, 'gap_delta_first_ticker': 0.01,
                          'cache_status': cached.status.value, 'on_demand_status': fresh.status.value,
                          'max_mu_difference': float(np.max(np.abs(cached.mu_gap-fresh.mu_gap))),
                          'max_omega_difference': float(np.max(np.abs(cached.Omega_gap-fresh.Omega_gap))),
                          'cache_change_from_old': float(np.max(np.abs(cached.mu_gap-base.mu_gap))),
                          'model_gross_limit': config.v2.baseline_gross,
                          'side_leverage': config.v2.costs.side_leverage,
                          'risk_limits': config.risk.model_dump(mode='json')}, indent=2))


if __name__ == '__main__':
    main()
