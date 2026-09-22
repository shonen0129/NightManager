"""Offline accounting and ML training-cutoff review probes."""
from dataclasses import replace
import json
import runpy
import numpy as np
import pandas as pd

from leadlag.data.tickers import JP_TICKERS
from leadlag.models.ml_order_overlay import MLOrderOverlayModel, apply_overlay
from leadlag.models.ml_overlay_artifact import _validate_overlay_provenance
from leadlag.reporting.daily_pnl_report import compute_realized_pnl


def main():
    row = {'status': 'CANCELLED', 'ticker': '1305.T', 'side': 'SELL', 'original_side': 'BUY',
           'original_price': 100.0, 'quantity': 10, 'fill_quantity': 4, 'fill_price': 110.0,
           'trade_date': '2026-09-18', 'fill_detail': {'sBaiBaiTesuryo': 2.0}}
    cancelled = compute_realized_pnl([row])
    filled = compute_realized_pnl([{**row, 'status': 'FILLED', 'quantity': 4}])
    unknown_fee = compute_realized_pnl([{**row, 'status': 'FILLED', 'quantity': 4,
                                       'fill_detail': {'sBaiBaiTesuryo': None}}])
    missing_fee = compute_realized_pnl([{k: v for k, v in {**row, 'status': 'FILLED', 'quantity': 4}.items() if k != 'fill_detail'}])
    helpers = runpy.run_path('tests/unit/test_ml_order_overlay.py')
    # These timestamps refer to the Japanese session whose close is in the model.
    train_end = '2026-09-17T15:00:00+00:00'
    metadata = helpers['_verified_overlay_metadata'](train_end)
    normalized = _validate_overlay_provenance(metadata, 'review probe')
    date = pd.Timestamp('2026-09-18')
    scores = np.linspace(-1, 1, len(JP_TICKERS))
    weights = np.zeros(len(JP_TICKERS))
    weights[:5], weights[-5:] = -0.2, 0.2
    base = replace(helpers['_base_result'](scores), w_final=weights)
    model = MLOrderOverlayModel(helpers['_DummyLGBM'](), ['score'], 1.0, False, False, False, metadata=metadata)
    result = apply_overlay(base, helpers['_make_df_exec'](date), model, str(date.date()), allow_implicit_io=False)
    print(json.dumps({
        'cancelled_partial_fill': {'report_records': len(cancelled), 'realized_pnl': sum(r.realized_pnl for r in cancelled),
                                  'expected_pnl': 38.0, 'filled_control_pnl': sum(r.realized_pnl for r in filled)},
        'null_fee': {'report_records': len(unknown_fee), 'certified_fee': sum(r.fee for r in unknown_fee),
                     'reported_pnl': sum(r.realized_pnl for r in unknown_fee)},
        'missing_fee': {'report_records': len(missing_fee), 'certified_fee': sum(r.fee for r in missing_fee)},
        'ml_timezone_cutoff': {'raw_train_end': train_end, 'jst_train_end': '2026-09-18',
                              'normalized_train_end': normalized['train_end'],
                              'trade_date': str(date.date()), 'overlay_applied': result.summary.get('overlay_applied', 0)},
    }, indent=2))


if __name__ == '__main__':
    main()
