"""Offline completion probe; broker/API and output boundaries are stubbed."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import json
import pandas as pd

from leadlag.core.types import OrderRequest, OrderSide
from leadlag.execution.contracts import ExecutionPlan
from leadlag.execution.state_store import ExecutionStateStore
from leadlag.execution import post_decision


def main():
    output = []
    for missing in ('fee', 'fill_price'):
        with TemporaryDirectory(prefix='leadlag-review-fill-') as folder:
            root = Path(folder)
            store = ExecutionStateStore(root / 'state.sqlite')
            plan = ExecutionPlan(decision_id='d', trade_date='2026-09-18', close_orders=(),
                                 new_orders=(OrderRequest('1305.T', OrderSide.BUY, 10),))
            run = store.prepare_run(account_key='test', strategy_key='production_v2', trade_date=plan.trade_date,
                                    job_type='decision', decision_id=plan.decision_id)
            store.record_plan(run.run_id, plan)
            store.mark_submission_started(run.run_id)
            frame = pd.DataFrame({'ticker': ['1305.T'], 'action': ['BUY'], 'quantity': [10]})
            frame.attrs['trade_date'] = plan.trade_date
            summary = {'run_id': run.run_id, 'expected_orders_count': 1, 'buy_results': [
                {'ticker': '1305.T', 'side': 'BUY', 'quantity': 10, 'order_id': 'a', 'status': 'FILLED'}]}
            positions = root / 'positions.json'
            positions.write_text(json.dumps({'positions': [{'ticker': '1305.T', 'side': 'BUY', 'quantity': 10}]}))
            def fills(_broker, records):
                for record in records:
                    record.update(fill_quantity=10, fill_price=100.0, fee=1.0)
                    record[missing] = None
            with patch.multiple(post_decision,
                save_decision_output=lambda *a, **kw: 'decision.csv',
                submit_orders_via_api=lambda **kw: summary,
                save_position_snapshot=lambda *a, **kw: str(positions),
                save_wallet_snapshot=lambda *a, **kw: 'wallet.json',
                save_daily_journal=lambda **kw: 'journal.json',
                fetch_fill_prices=fills):
                post_decision._write_decision_output_and_submit(
                    frame, {'trade_date': plan.trade_date}, root, False, SimpleNamespace(), {}, store)
            output.append({'missing': missing, 'run_status': store.get_run(run.run_id).status,
                           'recovery_candidates': len(store.list_recovery_candidates()),
                           'accounting_fills': len(store.list_observed_fills(run.run_id))})
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
