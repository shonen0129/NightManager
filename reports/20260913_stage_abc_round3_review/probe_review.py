"""Offline review probes: temporary artifacts and fake brokers only."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import pickle
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from leadlag.data.gap_store import GapStore
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.config import load_config_from_yaml
from leadlag.models.production_v2 import ProductionV2Model, generate_v2_production_portfolio
from leadlag.models import ml_order_overlay as overlay
from leadlag.execution import post_decision, pricing, broker_ops, close, var_history
from leadlag.broker.tachibana.client import TachibanaBrokerClient
from leadlag.core.types import OrderResult, OrderStatus
from leadlag.cli import main

logging.basicConfig(level=logging.ERROR)
out = {}
cfg = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
spec = importlib.util.spec_from_file_location("review_overlay_helpers", ROOT / "tests/unit/test_ml_order_overlay.py")
helpers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helpers
spec.loader.exec_module(helpers)


def probe(name, fn):
    try:
        out[name] = fn()
    except Exception as exc:
        out[name] = {"probe_error": repr(exc), "type": type(exc).__name__}


def config_for_probe():
    return cfg.v2.model_copy(deep=True, update={
        "macro_kappa_enabled": False, "macro_direction_enabled": False,
        "cs_overlay_enabled": False, "ml_overlay_enabled": False,
    })


def missing_distribution_provenance():
    with tempfile.TemporaryDirectory(prefix="abc-r3-gap-") as tmp:
        path = Path(tmp) / "gap.sqlite"
        store = GapStore(path)
        mu, cov = np.linspace(-.03, .03, len(JP_TICKERS)), np.eye(len(JP_TICKERS)) * .001
        # First prove future metadata is rejected, then remove only metadata.
        cases = {}
        for name, metadata in (("future", {"sig_date": "2026-08-17"}), ("missing", None)):
            store.save_horizon("2026-08-14", mu, cov, metadata=metadata)
            result = generate_v2_production_portfolio("2026-08-14", path, config_for_probe())
            cases[name] = {"audit": result["leakage"], "gross": float(np.abs(result["w_final"]).sum()),
                           "fallback": result["fallback"], "diagnostics": result["diagnostics"]}
        npy_dir = Path(tmp) / "legacy"
        (npy_dir / "matrices").mkdir(parents=True)
        np.save(npy_dir / "matrices/mu_gap_20260814.npy", mu)
        np.save(npy_dir / "matrices/omega_gap_20260814.npy", cov)
        result = generate_v2_production_portfolio("2026-08-14", npy_dir, config_for_probe())
        cases["npy_missing"] = {"audit":result["leakage"], "gross":float(np.abs(result["w_final"]).sum()),
            "fallback":result["fallback"], "diagnostics":result["diagnostics"]}
        return cases


def dummy_overlay():
    return overlay.MLOrderOverlayModel(lgbm=helpers._DummyLGBM(), cont_cols=["score"],
        target_std=1.0, use_ticker=False, use_classification=False, per_ticker_interactions=False)


def train_meta(end):
    return {"metadata_status": "verified", "train_start": "2015-01-05", "train_end": end,
            "data_hash": "probe-data", "config_hash": "probe-config"}


def invalid_artifact_dates():
    cases = {}
    with tempfile.TemporaryDirectory(prefix="abc-r3-date-") as tmp:
        for name, cutoff in (("null", None), ("nat", "NaT"), ("valid_future", "2026-08-14")):
            path = Path(tmp) / name
            overlay.save_overlay_model(dummy_overlay(), path, training_metadata=train_meta(cutoff))
            loaded = overlay.load_overlay_model(path)
            date = pd.Timestamp("2021-01-04")
            base = helpers._base_result(np.linspace(-1.0, 1.0, len(JP_TICKERS)))
            base["w_final"] = np.array([.2] * 5 + [-.2] * 5 + [0.] * 7)
            try:
                result = overlay.apply_overlay(base, helpers._make_df_exec(date), loaded, str(date.date()))
                cases[name] = {"load_accepted": True, "apply_accepted": True,
                               "train_end": loaded.metadata["train_end"],
                               "gross": float(np.abs(result["w_final"]).sum())}
            except ValueError as exc:
                cases[name] = {"load_accepted": True, "apply_accepted": False, "error": str(exc)}
    return cases


def legacy_artifact_mix():
    with tempfile.TemporaryDirectory(prefix="abc-r3-legacy-") as tmp:
        path = Path(tmp)
        model = dummy_overlay()
        object.__setattr__(model, "metadata", {"metadata_version": 2, **train_meta("2026-08-14")})
        (path / "model.pkl").write_bytes(pickle.dumps(model))
        (path / "metadata.json").write_text(json.dumps({"metadata_version": 2, **train_meta("2020-12-31")}))
        loaded = overlay.load_overlay_model(path)
        return {"embedded_cutoff_before_load": "2026-08-14", "loaded_cutoff": loaded.metadata["train_end"],
                "current_exists": (path / "CURRENT").exists(), "accepted": True}


def fill_failure_status():
    class FilledBroker(TachibanaBrokerClient):
        def __init__(self):
            self.details = 0
        def submit_orders_batch(self, orders, **kwargs):
            return [OrderResult(order_id="fake-" + str(i), status=OrderStatus.FILLED,
                ticker=o.ticker, side=o.side, quantity=o.quantity, eigyou_day="20260814")
                for i, o in enumerate(orders)]
        def get_order_detail(self, *args):
            self.details += 1
            raise OSError("injected detail endpoint failure")
        def close(self):
            pass
        def get_positions(self, **kwargs):
            from leadlag.broker.base import Position
            return [Position(ticker=JP_TICKERS[0], side="BUY", quantity=400, price=1000., execution_id="fake-pos")]
    result = {}
    with tempfile.TemporaryDirectory(prefix="abc-r3-fill-") as tmp:
        api = FilledBroker()
        frame = pd.DataFrame({"ticker": JP_TICKERS[:2], "quantity": [100, 100], "action": ["BUY", "SELL"]})
        with patch.object(post_decision, "save_decision_output", return_value="fake.csv"), \
             patch.object(broker_ops, "split_large_orders", side_effect=lambda orders:(orders, [])), \
             patch.object(broker_ops, "_wait_for_fills_sync", side_effect=lambda api, rows, **kwargs:(rows, True)), \
             patch.object(pricing.time, "sleep", return_value=None), \
             patch.object(post_decision, "save_position_snapshot", return_value="position.json"), \
             patch.object(post_decision, "save_wallet_snapshot", return_value="wallet.json"), \
             patch.object(post_decision, "save_daily_journal", return_value="journal.json"):
            try:
                value = post_decision._write_decision_output_and_submit(frame,
                    {"trade_date": pd.Timestamp("2026-08-14")}, tmp, False, api, {})
                result["decision"] = {"returned_normally": True, "returned": value}
            except Exception as exc:
                result["decision"] = {"returned_normally": False, "error": str(exc)}
        saved = json.loads((Path(tmp) / "api_execution_log.json").read_text())
        result["decision"].update({"detail_calls":api.details, "fill_statuses":[
            row.get("fill_status") for row in saved["buy_results"] + saved["sell_results"]],
            "reconciliation_errors":saved.get("reconciliation_errors")})
        api.details=0
        with patch("leadlag.core.market_calendar.is_market_closed",return_value=False), \
             patch.object(close,"build_api_client",return_value=api), \
             patch.object(close,"build_output_dir",return_value=tmp), \
             patch.object(close,"save_position_snapshot",return_value="positions.json"), \
             patch.object(close,"save_wallet_snapshot",return_value="wallet.json"), \
             patch.object(close,"save_daily_journal",return_value="journal.json"), \
             patch.object(pricing.time,"sleep",return_value=None):
            code=main(["close"])
        saved_close=json.loads((Path(tmp)/"close_execution_log.json").read_text())
        result["close"]={"cli_return_code":code,"detail_calls":api.details,
            "fill_statuses":[r.get("fill_status") for r in saved_close["close_results"]],
            "close_incomplete":saved_close["close_incomplete"],
            "reconciliation_errors":saved_close.get("reconciliation_errors")}
    return result


def failed_initial_log():
    calls=[]
    class PartialBroker:
        def submit_orders_batch(self, orders, **kwargs):
            return [OrderResult(order_id="fake", status=OrderStatus.PARTIALLY_FILLED,
                ticker=o.ticker, side=o.side, quantity=o.quantity) for o in orders]
    with tempfile.TemporaryDirectory(prefix="abc-r3-log-") as tmp:
        with patch.object(post_decision,"save_decision_output",return_value="fake.csv"), \
             patch.object(broker_ops,"split_large_orders",side_effect=lambda orders:(orders,[])), \
             patch.object(broker_ops,"_wait_for_fills_sync",side_effect=lambda api,rows,**kwargs:(rows,True)), \
             patch.object(broker_ops,"_write_api_execution_log",side_effect=OSError("injected log disk error")), \
             patch.object(post_decision,"fetch_fill_prices",side_effect=lambda *a,**k:calls.append("fills")), \
             patch.object(post_decision,"save_position_snapshot",side_effect=lambda *a,**k:calls.append("positions")), \
             patch.object(post_decision,"save_wallet_snapshot",side_effect=lambda *a,**k:calls.append("wallet")):
            try:
                post_decision._write_decision_output_and_submit(pd.DataFrame({"ticker":JP_TICKERS[:2],
                    "quantity":[100,100],"action":["BUY","SELL"]}),{"trade_date":"2026-08-14"},tmp,False,PartialBroker(),{})
            except Exception as exc:
                return {"error_type":type(exc).__name__,"error":str(exc),"reconciliation_calls":calls}


def inactive_artifact_fingerprint():
    with tempfile.TemporaryDirectory(prefix="abc-r3-fingerprint-") as tmp:
        path=Path(tmp)
        overlay.save_overlay_model(dummy_overlay(),path,training_metadata=train_meta("2020-12-31"))
        before=var_history._file_manifest_fingerprint(path)
        selected_before=overlay.load_overlay_model(path).metadata["artifact_version"]
        staging=path/".staging-interrupted"
        staging.mkdir()
        (staging/"model.pkl").write_bytes(b"incomplete, never activated")
        after=var_history._file_manifest_fingerprint(path)
        selected_after=overlay.load_overlay_model(path).metadata["artifact_version"]
        return {"same_loaded_version":selected_before==selected_after,"cache_key_changed":before!=after}


probe("missing_distribution_metadata", missing_distribution_provenance)
probe("invalid_artifact_dates", invalid_artifact_dates)
probe("legacy_artifact_mix", legacy_artifact_mix)
probe("fill_failure_silent_success", fill_failure_status)
probe("initial_log_failure_skips_reconciliation", failed_initial_log)
probe("inactive_staging_invalidates_var_cache", inactive_artifact_fingerprint)
print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
