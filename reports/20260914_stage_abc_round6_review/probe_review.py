"""Comprehensive offline A-C regression probes; no live stores or brokers."""
from __future__ import annotations
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def definitions(relative, name, split=None):
    path = ROOT / relative
    source = path.read_text()
    if split:
        source = source.partition(split)[0]
    tree = ast.parse(source, str(path))
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name) and node.value.func.id in {"probe", "run", "print"}
    )]
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


r4 = definitions("reports/20260914_stage_abc_round4_review/probe_review.py", "r5_previous", '\nrun("T01_artifact_boundaries"')
core = definitions("reports/20260912_workspace_audit/reproduce_findings.py", "r5_core")
integration = definitions("reports/20260912_workspace_audit/reproduce_integration.py", "r5_integration")
cfg = r4.old.cfg
from leadlag.utils import gap_matrix_io as gio
from leadlag.models.production_v2 import ProductionV2Model
from leadlag.models.signal_enhancement import apply_multi_horizon_blend
from leadlag.models.v2 import gap_io
from leadlag.broker.tachibana.client import TachibanaBrokerClient
from leadlag.execution import broker_ops, close, pricing, post_decision, v2_bridge
from leadlag.execution.backtester import BacktestEngine
from leadlag.core.types import OrderStatus, OrderResult, OrderSide
from leadlag.data.gap_store import GapStore
from leadlag.data.tickers import JP_TICKERS, US_TICKERS, TOPIX_TICKER
from leadlag.data.preprocessor import preprocess_data

results = {}


def run(name, function):
    try:
        results[name] = function()
    except Exception as exc:
        results[name] = {"probe_error": type(exc).__name__, "message": str(exc)}


def bundle_boundaries():
    result = {}
    with tempfile.TemporaryDirectory(prefix="abc-r5-npy-") as tmp:
        for horizon in (1, 3, 5):
            patterns = {} if horizon == 1 else {
                "mu_pattern": "matrices/mu_gap_h{h}_{date}.npy",
                "omega_pattern": "matrices/omega_gap_h{h}_{date}.npy",
                "pattern_kwargs": {"h": horizon},
            }
            mu_old = np.linspace(-.02, .02, 17)
            mu_new = mu_old[::-1] + np.sin(np.arange(17)) * .015
            omega_old, omega_new = np.eye(17) * .001, np.eye(17) * .002
            date = "2026-08-14"
            metadata = {"sig_date": "2026-08-13", "trade_date": date, "horizon": horizon}
            cases = {}
            for mode in ("normal", "no_metadata", "omega_failure", "metadata_failure", "manifest_failure", "missing_manifest", "interleaved_read"):
                directory = Path(tmp) / str(horizon) / mode
                assert gio.save_gap_matrices(directory, date, mu_old, omega_old, metadata=metadata, **patterns)
                saved = True
                fault_count = 0
                real_write = gio._atomic_write_bytes
                def faulty_write(path, value):
                    nonlocal fault_count
                    fail = ((mode == "omega_failure" and path.name.startswith("omega_gap")) or
                            (mode == "metadata_failure" and path.name.startswith("gap_metadata")) or
                            (mode == "manifest_failure" and path.name.endswith(".bundle.json")))
                    if fail:
                        fault_count += 1
                        raise OSError("injected " + mode)
                    return real_write(path, value)
                if mode == "no_metadata":
                    saved = gio.save_gap_matrices(directory, date, mu_new, omega_new, **patterns)
                elif mode.endswith("failure"):
                    with patch.object(gio, "_atomic_write_bytes", side_effect=faulty_write):
                        saved = gio.save_gap_matrices(directory, date, mu_new, omega_new, metadata={**metadata, "new": True}, **patterns)
                elif mode == "missing_manifest":
                    next((directory / "matrices").glob("*.bundle.json")).unlink()
                if mode == "interleaved_read":
                    original_read = Path.read_bytes
                    switched = []
                    def interleaved_read(path):
                        value = original_read(path)
                        if path.name.startswith("mu_gap") and not switched:
                            switched.append(True)
                            gio.save_gap_matrices(directory, date, mu_new, omega_new, metadata={**metadata, "new": True}, **patterns)
                        return value
                    with patch.object(Path, "read_bytes", new=interleaved_read):
                        mu, omega, meta, alerts = gio.load_gap_bundle(directory, date, **patterns)
                    cases[mode] = {"interleaved_commit": bool(switched), "metadata": meta, "alerts": alerts}
                    continue
                mu, omega, meta, alerts = gio.load_gap_bundle(directory, date, **patterns)
                rc = r4.old.config_for_probe().model_copy(deep=True, update={"gap_input_dir": str(directory)})
                model = ProductionV2Model(rc, blpx_model=SimpleNamespace())
                model._current_gap_input_dir = directory
                with patch.object(gap_io, "_compute_ondemand", return_value=(mu_old, omega_old)) as demand:
                    actual_mu, actual_omega = model.compute_distribution(
                        date, pd.DataFrame(index=[pd.Timestamp(date)]), {tk: 1000. for tk in JP_TICKERS}, horizon=horizon)
                entry = {
                    "save_returned": saved, "write_fault_count": fault_count,
                    "metadata": meta, "alerts": alerts,
                    "mu_is_new": bool(np.array_equal(mu, mu_new)),
                    "omega_is_new": bool(np.array_equal(omega, omega_new)),
                    "compute_distribution_ondemand_calls": demand.call_count,
                    "compute_distribution_used_new_mu": bool(np.array_equal(actual_mu, mu_new)),
                    "compute_distribution_used_old_omega": bool(np.array_equal(actual_omega, omega_old)),
                }
                if horizon == 1:
                    decision = r4.old.generate_v2_production_portfolio(date, directory, r4.old.config_for_probe())
                    entry["decision"] = {"gross": float(np.abs(decision["w_final"]).sum()), "leakage": decision["leakage"]["status"], "fallback": decision["fallback"]}
                else:
                    scores = np.linspace(-2., 2., 17)
                    blend, blend_alerts = apply_multi_horizon_blend(scores, directory, date, (1, horizon), (.5, .5))
                    entry["blend_max_change"] = float(np.max(np.abs(blend - scores)))
                    entry["blend_alerts"] = blend_alerts
                cases[mode] = entry
            result[str(horizon)] = cases
    return result


def broker_boundaries():
    api = object.__new__(TachibanaBrokerClient)
    cases = {}
    for name, code, qty in (("filled", "10", "100"), ("partial", "9", "30"), ("cancelled_partial", "7", "30"), ("cancelled", "7", "0"), ("rejected", "2", "0"), ("expired_partial", "11", "30"), ("cancel_failed_unfilled", "8", "0"), ("amend_failed_unfilled", "5", "0")):
        response = {"sOrderOrderSuryou": "100", "sYakuzyouSuryou": qty, "sOrderStatusCode": code}
        api._client = SimpleNamespace(get_order_detail=lambda *a, row=response: row)
        cases[name] = api.get_order_status("fake").value
    queries = []
    api._client = SimpleNamespace(get_order_detail=lambda *a: queries.append(a) or {"sYakuzyouPrice": "1000", "sYakuzyouSuryou": "30", "sOrderStatus": "取消完了"})
    rows = [{"order_id": "fake-1", "status": "FILLED"}, {"order_id": "fake-2", "status": "CANCELLED"}]
    pricing.fetch_fill_prices(api, rows, wait_seconds=0)
    cases["fill_collection"] = {"queries": len(queries), "quantities": [row["fill_quantity"] for row in rows]}
    with tempfile.TemporaryDirectory(prefix="abc-r5-reject-") as tmp:
        frame = pd.DataFrame({"ticker": JP_TICKERS[:2], "quantity": [100, 100], "action": ["BUY", "SELL"]})
        try:
            broker_ops.submit_orders_via_api(frame, integration.RejectingBroker(), tmp, {})
            cases["all_rejected"] = {"returned_normally": True}
        except broker_ops.OrderExecutionIncomplete as exc:
            cases["all_rejected"] = {"returned_normally": False, "accepted": exc.summary["accepted_orders_count"], "failed": exc.summary["failed_orders_count"]}
    for module in (broker_ops, close):
        sequence = iter([OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED])
        calls = []
        fake = SimpleNamespace(get_order_status=lambda *a: calls.append(a) or next(sequence))
        row = {"order_id": "fake", "ticker": "1617.T", "status": "SUBMITTED"}
        if module is broker_ops:
            order = OrderResult(order_id="fake", status=OrderStatus.SUBMITTED, ticker="1617.T", side=OrderSide.BUY, quantity=100)
            polled, completed = module._wait_for_fills_sync(fake, [order], timeout_seconds=2., poll_interval=0.)
            row["status"] = polled[0].status.value
        else:
            module._wait_for_close_fills_sync(fake, [row], timeout_seconds=2., poll_interval=0.)
            completed = row["status"] == "FILLED"
        cases[module.__name__ + "_poll"] = {"calls": len(calls), "status": row["status"], "completed": completed}
    return cases


def risk_boundaries():
    result = {"actual_stop_flat_flow": integration.risk_flat()}
    for name, action, target, current in (("flat", "HOLD", 0, 10), ("reduce", "BUY", 5, 10), ("increase", "BUY", 11, 10), ("reverse", "SELL", 5, 10), ("new", "BUY", 5, 0)):
        frame = pd.DataFrame({"ticker": ["1617.T"], "action": [action], "quantity": [target], "etf_amount": [1000. * target]})
        with patch.object(post_decision, "run_risk_checks", return_value={"is_blocked": True}), patch.object(post_decision, "_print_risk_report"):
            try:
                post_decision._run_risk_check_and_print({}, frame, 300000, pd.Series(dtype=float), cfg.strategy, {"1617.T": current})
                result[name] = {"allowed": True}
            except RuntimeError:
                result[name] = {"allowed": False}
    return result


def data_boundaries():
    result = {"holiday": integration.holiday_alignment(), "stale_requested_day": r4.outcome(integration.stale_day)}
    dates = pd.bdate_range("2026-06-01", periods=80)
    us = pd.DataFrame({tk: 100 + np.arange(80) for tk in US_TICKERS}, index=dates)
    jp = pd.DataFrame({tk: 1000 + np.arange(80) for tk in JP_TICKERS + [TOPIX_TICKER]}, index=dates)
    processed = preprocess_data({"us_close": us, "jp_close": jp, "jp_open": jp - .5}, beta_window=2, strict_validation=True)
    result["strict_clean_data"] = {"rows": len(processed), "signals_strictly_prior": bool((pd.to_datetime(processed["sig_date"]) < processed.index).all())}
    class PriceBroker:
        def __init__(self): self.calls = []
        def fetch_open_prices(self, tickers, **kwargs): self.calls.append("open"); return {tk: 1000. for tk in tickers}
        def fetch_current_prices(self, tickers, **kwargs): self.calls.append("current"); return {tk: 1050. for tk in tickers}
    broker = PriceBroker()
    prices = v2_bridge._resolve_current_prices(cfg, broker, None, False)
    result["current_prices"] = {"calls": broker.calls, "price": prices[JP_TICKERS[0]]}
    return result


def atomic_sqlite_failure():
    output = {}
    with tempfile.TemporaryDirectory(prefix="abc-r5-sqlite-") as tmp:
        for horizon in (None, 3, 5):
            store = GapStore(Path(tmp) / f"{horizon}.sqlite")
            store.save_horizon("2026-08-14", np.ones(17), np.eye(17), {"version": "old"}, horizon=horizon)
            original = store._cache._set_with_conn
            calls = []
            def fail(conn, key, value, *a, **kw):
                calls.append(key)
                if len(calls) == 2:
                    raise OSError("injected second cache write")
                return original(conn, key, value, *a, **kw)
            error = None
            try:
                with patch.object(store._cache, "_set_with_conn", side_effect=fail):
                    store.save_horizon("2026-08-14", np.ones(17) * 2, np.eye(17) * 2, {"version": "new"}, horizon=horizon)
            except Exception as exc:
                error = type(exc).__name__
            mu, omega, meta = store.load_horizon("2026-08-14", horizon=horizon)
            output[str(horizon)] = {"fault_reached": len(calls) == 2, "error": error, "mu": float(mu[0]), "omega": float(omega[0, 0]), "metadata": meta}
    return output


def default_backtest_overlay():
    with tempfile.TemporaryDirectory(prefix="abc-r5-bt-") as tmp:
        root = Path(tmp)
        r4.old.overlay.save_overlay_model(r4.old.dummy_overlay(), root, r4.old.train_meta("2020-12-31"))
        v2 = cfg.v2.model_copy(deep=True, update={"ml_overlay_model_dir": str(root)})
        test_cfg = cfg.model_copy(deep=True, update={"v2": v2})
        with patch.object(integration, "cfg", test_cfg):
            capture = integration.overlay_default()
        return capture


def common_schema_cache():
    a = pd.DataFrame({f"us_cc_{US_TICKERS[0]}": [.01, .02], f"us_cc_{US_TICKERS[1]}": [.03, .04]}, index=pd.to_datetime(["2020-01-06", "2020-01-07"]))
    first, second = a.columns
    b = a.rename(columns={first: second, second: first})
    model = core.ProductionBLPXModel(cfg.v2.blpx)
    def build(frame, target, **kwargs):
        return SimpleNamespace(to_dict=lambda: {"first_us_last": float(frame[first].iloc[-1]), "column_order": list(frame.columns)})
    with patch("leadlag.core.pipeline.build_common_inputs", side_effect=build) as factory:
        initial = model._prepare_common_inputs(a, y_jp_target=np.zeros((2, 17)))
        reused = model._prepare_common_inputs(b, y_jp_target=np.zeros((2, 17)))
        same_instance_calls = factory.call_count
        fresh = core.ProductionBLPXModel(cfg.v2.blpx)._prepare_common_inputs(b, y_jp_target=np.zeros((2, 17)))
    return {"old_columns": list(a.columns), "new_columns": list(b.columns), "same_numeric_buffer": bool(np.array_equal(a.to_numpy(), b.to_numpy())), "same_instance_build_calls": same_instance_calls, "cache_reused": initial is reused, "cached_value": reused["first_us_last"], "fresh_value": fresh["first_us_last"]}


def dsr_reference():
    output = {}
    for annual_factor in (245, 252):
        metrics = {"net_sharpe": 1., "trials": 1, "n_observations": 252, "trading_days_per_year": annual_factor}
        daily_sr = 1 / np.sqrt(annual_factor)
        output[str(annual_factor)] = {"reported": core.compute_deflated_sharpe(metrics), "reference": float(core.norm.cdf(daily_sr * np.sqrt(251) / np.sqrt(1 + .5 * daily_sr ** 2)))}
    return output


run("U01_bundle_boundaries", bundle_boundaries)
run("U02_var_version_race", r4.var_version_race)
run("U03_verification", r4.verification_dangling_pointer)
run("U04_migration", r4.migration_report)
run("T01_artifact_dates", r4.artifact_boundaries)
run("T02_legacy", r4.legacy_rejection)
run("T03_missing_metadata", r4.old.missing_distribution_provenance)
run("T04_fill_failure", r4.old.fill_failure_status)
run("T05_initial_log", r4.old.failed_initial_log)
run("F01_F02_F20_brokers", broker_boundaries)
run("F03_risk_reduction", risk_boundaries)
run("F04_audit_flat", core.leakage_failure)
run("F05_F07_F08_F09_data", data_boundaries)
run("F10_correlation_cache", core.corr_cache)
run("F11_common_inputs_cache", core.common_cache)
run("F11_column_identity", common_schema_cache)
run("F12_atomic_sqlite_write", atomic_sqlite_failure)
run("F13_default_backtest_overlay", default_backtest_overlay)
run("F14_inventory_slippage", core.pnl_cost)
run("F15_initial_drawdown", lambda: core.calculate_metrics(pd.Series([-.1, 0.], index=pd.bdate_range("2026-08-13", periods=2))))
run("F16_all_days_metrics", lambda: core._extract_metrics({"daily_returns": pd.Series([-.1, 0., .02, 0.]), "daily_fallback": np.array([False, True, False, True])}))
run("F17_DSR", dsr_reference)
run("F21_PIT_multiplier", core.pit_multiplier)
run("production_artifact_readiness", r4.live_artifact_readiness)
(OUT / "reproductions.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str) + "\n")
print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
if any("probe_error" in value for value in results.values()):
    raise SystemExit(1)
