"""Capture synthetic decisions for independent code-version comparison.

The --old switch selects the saved pre-cleanup API only in this verifier.
No application module supplies a compatibility path.
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from leadlag.config.schemas import ProductionV2RunConfig
from leadlag.data.pit_lake import PITDataLake
from leadlag.data.tickers import JP_TICKERS
from leadlag.models.ml_order_overlay import MLOrderOverlayModel
from leadlag.models.ml_overlay_features import overlay_continuous_columns
from leadlag.models.ml_overlay_inference import apply_overlay
from leadlag.models.production_v2 import ProductionV2Model
from leadlag.models.v2.decision_engine import generate_v2_production_portfolio_from_distribution
from leadlag.utils.gap_matrix_io import save_gap_matrices
from leadlag.utils.gap_provenance import bundle_identity


class Predictor:
    def predict(self, features):
        return features['score'].to_numpy() * 0.3


def capture(result):
    return {
        'w_final': result.w_final.tolist(), 'scores': result.scores.tolist(),
        'mu_gap': result.mu_gap.tolist(), 'sigma_gap': result.sigma_gap.tolist(),
        'Omega_gap': result.Omega_gap.tolist(), 'fallback': result.fallback,
        'pit_binning': result.pit_binning, 'leakage': result.leakage,
        'numerical': result.numerical,
        'summary': {k: v for k, v in result.summary.items() if not k.startswith('p_trade_')},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = {}
    date = '2026-10-02'
    index = pd.date_range('2026-09-01', periods=24, freq='B')
    date = index[-1].strftime('%Y-%m-%d')
    frame = pd.DataFrame({
        'sig_date': index - pd.Timedelta(days=1),
        'topix_night_return': np.zeros(len(index)),
        **{f'jp_open_trade_{ticker}': np.full(len(index), 1000.0) for ticker in JP_TICKERS},
        **{f'jp_gap_{ticker}': np.zeros(len(index)) for ticker in JP_TICKERS},
        **{f'jp_beta_{ticker}': np.ones(len(index)) for ticker in JP_TICKERS},
        **{f'jp_oc_{ticker}': np.zeros(len(index)) for ticker in JP_TICKERS},
    }, index=index)
    observed = pd.DataFrame(0.0, index=index, columns=JP_TICKERS)
    market_vol = pd.DataFrame(0.0, index=index, columns=JP_TICKERS)
    adr = pd.DataFrame(0.0, index=index, columns=[f"adr_{ticker}" for ticker in JP_TICKERS])
    for seed in range(4):
        rng = np.random.default_rng(seed)
        mu = rng.normal(0, 0.01, len(JP_TICKERS))
        a = rng.normal(size=(len(JP_TICKERS), len(JP_TICKERS)))
        omega = a @ a.T * 0.0001 + np.eye(len(JP_TICKERS)) * 0.001
        for minvar in (False, True):
            cfg = ProductionV2RunConfig(
                macro_kappa_enabled=False, macro_direction_enabled=False,
                cs_overlay_enabled=False, ml_overlay_enabled=False,
                minvar_enabled=minvar,
            )
            key = f'{seed}:{minvar}'
            history = rng.normal(0.3, 0.4, 100)
            history_dates = pd.date_range(end=pd.Timestamp(date) - pd.Timedelta(days=1), periods=100, freq='B')
            direct = generate_v2_production_portfolio_from_distribution(
                mu_gap=mu, omega_gap=omega, trade_date=date, run_config=cfg,
                gap_input_dir=None, df_exec=frame, pit_ir_history=history,
                pit_history_trade_dates=history_dates.to_numpy(), allow_implicit_io=False,
            )
            records[key + ':kernel'] = capture(direct)
            overlay = MLOrderOverlayModel(
                Predictor(), overlay_continuous_columns(), 1.0, False, False, False,
                metadata={
                    'metadata_version': 2, 'metadata_status': 'verified',
                    'train_start': '2015-01-05', 'train_end': '2020-12-31',
                    'data_hash': 'synthetic', 'config_hash': 'synthetic',
                },
            )
            overlaid = apply_overlay(
                direct, frame, overlay, date, market_vol_frame=market_vol,
                allow_implicit_io=False, adr_features=adr,
            )
            assert overlaid.summary.get("overlay_applied") == 1
            records[key + ":overlay"] = capture(overlaid)
            with tempfile.TemporaryDirectory(prefix='leadlag-kiss-version-') as temporary:
                path = Path(temporary)
                lake = PITDataLake(frame)
                inputs = lake.build_decision_inputs(
                    date, current_prices={ticker: 1000.0 for ticker in JP_TICKERS},
                    gap_input_dir=path, open_910_returns=observed, source='comparison',
                )
                metadata = {
                    'sig_date': inputs.known.sig_date.strftime('%Y-%m-%d'),
                    'trade_date': date, 'horizon': 1,
                    **bundle_identity(
                        inputs.historical.calculation_frame(inputs.known.as_of), date,
                        config=cfg, open_910_returns=observed,
                        gap_inputs=(inputs.known.jp_gap_returns, inputs.known.jp_betas, inputs.known.topix_night_return),
                    ),
                }
                assert save_gap_matrices(path, date, mu, omega, metadata=metadata)
                typed = ProductionV2Model(cfg.model_copy(deep=True)).decide(inputs=inputs, overlay_enabled=False)
                records[key + ':typed'] = capture(typed)
                replay_model = ProductionV2Model(cfg.model_copy(deep=True))
                replay = replay_model.decide(trade_date=date, gap_input_dir=path, overlay_enabled=False) if args.old else replay_model.decide_from_cache(date, path)
                records[key + ':cache'] = capture(replay)
    args.output.write_text(json.dumps(records, sort_keys=True, default=str, allow_nan=True))
    print(f'{len(records)} numerical cases captured')


if __name__ == '__main__':
    main()
