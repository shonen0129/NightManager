"""Regression tests for the stage A-C safety fixes."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from leadlag.core.types import OrderResult, OrderSide, OrderStatus, OrderType
from leadlag.data.gap_store import GapStore
from leadlag.data.preprocessor import preprocess_data
from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER, US_TICKERS
from leadlag.domain.portfolio import PortfolioDecision
from leadlag.execution import post_decision
from leadlag.execution.broker_ops import (
    OrderExecutionIncomplete,
    _wait_for_fills_sync,
    submit_orders_via_api,
)
from leadlag.execution.config import load_config_from_yaml
from leadlag.execution.var_inputs import _active_overlay_artifact_fingerprint
from leadlag.models.ml_order_overlay import (
    MLOrderOverlayModel,
    apply_overlay,
    load_overlay_model,
    save_overlay_model,
)
from leadlag.models.production_v2 import ProductionV2Model
from leadlag.models.v2.decision_engine import _derive_signal_date
from leadlag.models.v2.overlay_applier import _multi_horizon_scores_with_metadata
from leadlag.reporting.metrics import compute_drawdown_series
from leadlag.utils.gap_provenance import bundle_identity


def _overlay_decision(n_j: int | None = None) -> PortfolioDecision:
    """Build the typed decision fixture used by overlay contract tests."""
    n = n_j or len(JP_TICKERS)
    return PortfolioDecision(
        w_final=np.zeros(n),
        scores=np.linspace(-1.0, 1.0, n),
        mu_gap=np.zeros(n),
        sigma_gap=np.ones(n),
        Omega_gap=np.eye(n),
        fallback={"gap_data_missing": False, "audit_failure": False},
        pit_binning={"multiplier": 1.0},
        leakage={"status": "PASSED"},
        numerical={"status": "PASSED"},
        alerts=[],
        summary={},
        run_config=SimpleNamespace(baseline_gross=2.0),
    )


class _PartialThenFilledBroker:
    def __init__(self) -> None:
        self.calls = 0

    def get_order_status(self, _order_id: str) -> OrderStatus:
        self.calls += 1
        if self.calls == 1:
            return OrderStatus.SUBMITTED
        if self.calls == 2:
            return OrderStatus.PARTIALLY_FILLED
        return OrderStatus.FILLED


def test_partial_fill_remains_pending_until_terminal_fill() -> None:
    result = OrderResult(
        order_id="O-1",
        status=OrderStatus.SUBMITTED,
        ticker="1617.T",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
    )
    broker = _PartialThenFilledBroker()
    results, polled = _wait_for_fills_sync(
        broker, [result], timeout_seconds=1.0, poll_interval=0.0
    )
    assert polled is True
    assert broker.calls >= 3
    assert results[0].status is OrderStatus.FILLED


def test_multihorizon_future_provenance_flats_in_real_decide_path(tmp_path) -> None:
    cfg = load_config_from_yaml("configs/production/production.yaml", strict=True).v2
    cfg = cfg.model_copy(
        deep=True,
        update={
            "macro_kappa_enabled": False,
            "macro_direction_enabled": False,
            "cs_overlay_enabled": False,
            "ml_overlay_enabled": False,
            "gap_input_dir": str(tmp_path / "gap.sqlite"),
        },
    )
    df_exec = pd.DataFrame(
        {"sig_date": [pd.Timestamp("2026-08-13")]},
        index=pd.DatetimeIndex(["2026-08-14"]),
    )
    identity = bundle_identity(df_exec, "2026-08-14", config=cfg)
    h1_identity = bundle_identity(
        df_exec,
        "2026-08-14",
        config=cfg,
        gap_inputs=(
            np.zeros(len(JP_TICKERS)),
            np.zeros(len(JP_TICKERS)),
            0.0,
        ),
        horizon=1,
    )
    store = GapStore(tmp_path / "gap.sqlite")
    for horizon in (None, 3, 5):
        store.save_horizon(
            "2026-08-14",
            np.linspace(-0.03, 0.03, len(JP_TICKERS)),
            np.eye(len(JP_TICKERS)) * 0.001,
            metadata={
                "sig_date": "2026-08-17" if horizon == 3 else "2026-08-13",
                **(h1_identity if horizon is None else identity),
            },
            horizon=horizon,
        )
    model = ProductionV2Model(cfg, blpx_model=SimpleNamespace())
    result = model.decide(
        "2026-08-14",
        gap_input_dir=tmp_path / "gap.sqlite",
        df_exec=df_exec,
        current_prices={ticker: 1000.0 for ticker in JP_TICKERS},
        overlay_enabled=False,
    )
    assert np.allclose(result.w_final, 0.0)
    assert result.fallback["audit_failure"] is True
    assert result.fallback["gap_data_missing"] is False
    assert result.diagnostics["distribution_provenance"]["status"] == "rejected"
    assert any("h=3" in alert for alert in result.alerts)


def test_multihorizon_rejects_bad_cache_provenance_and_recomputes_on_demand(tmp_path, monkeypatch) -> None:
    cfg = load_config_from_yaml("configs/production/production.yaml", strict=True).v2.model_copy(
        deep=True,
        update={
            "mh_horizons": (1, 3),
            "mh_weights": (0.5, 0.5),
            "gap_input_dir": str(tmp_path / "gap.sqlite"),
        },
    )
    df_exec = pd.DataFrame(
        {"sig_date": [pd.Timestamp("2026-08-13")]},
        index=pd.DatetimeIndex(["2026-08-14"]),
    )
    identity = bundle_identity(df_exec, "2026-08-14", config=cfg)
    store = GapStore(tmp_path / "gap.sqlite")
    n_j = len(JP_TICKERS)
    store.save_horizon(
        "2026-08-14",
        np.ones(n_j),
        np.eye(n_j),
        metadata={"sig_date": "2026-08-13", **identity},
        horizon=None,
    )
    store.save_horizon(
        "2026-08-14",
        np.ones(n_j),
        np.eye(n_j),
        metadata={"sig_date": "2026-08-17", **identity},
        horizon=3,
    )
    model = ProductionV2Model(cfg, blpx_model=SimpleNamespace())
    monkeypatch.setattr(
        "leadlag.models.v2.distribution_source._compute_ondemand",
        lambda *args, **kwargs: (np.full(n_j, 0.25), np.eye(n_j) * 0.01),
    )
    _mu, _omega, _scores, provenance = _multi_horizon_scores_with_metadata(
        model,
        trade_date="2026-08-14",
        df_exec=df_exec,
        current_prices={ticker: 1000.0 for ticker in JP_TICKERS},
    )
    assert provenance["horizons"]["1"]["source"] == "file_cache"
    assert provenance["horizons"]["3"]["source"] == "on_demand"
    assert provenance["horizons"]["3"]["metadata"]["sig_date"] == "2026-08-13"


def test_single_horizon_future_provenance_does_not_reach_audit(tmp_path) -> None:
    cfg = load_config_from_yaml("configs/production/production.yaml", strict=True).v2.model_copy(
        deep=True,
        update={
            "mh_blend_enabled": False,
            "macro_kappa_enabled": False,
            "macro_direction_enabled": False,
            "cs_overlay_enabled": False,
            "ml_overlay_enabled": False,
            "gap_input_dir": str(tmp_path / "gap.sqlite"),
        },
    )
    n_j = len(JP_TICKERS)
    GapStore(tmp_path / "gap.sqlite").save_horizon(
        "2026-08-14",
        np.ones(n_j),
        np.eye(n_j),
        metadata={"sig_date": "2026-08-17"},
        horizon=None,
    )
    model = ProductionV2Model(cfg, blpx_model=SimpleNamespace())
    result = model.decide(
        "2026-08-14",
        gap_input_dir=tmp_path / "gap.sqlite",
        df_exec=pd.DataFrame(
            {"sig_date": [pd.Timestamp("2026-08-13")]},
            index=pd.DatetimeIndex(["2026-08-14"]),
        ),
        current_prices={ticker: 1000.0 for ticker in JP_TICKERS},
        overlay_enabled=False,
    )
    assert np.allclose(result.w_final, 0.0)
    assert result.fallback["audit_failure"] is True


def test_single_horizon_npy_without_metadata_flats_before_leakage_audit(tmp_path) -> None:
    """A legacy matrix pair cannot create a normal decision without sig_date."""
    cfg = load_config_from_yaml("configs/production/production.yaml", strict=True).v2.model_copy(
        deep=True,
        update={
            "mh_blend_enabled": False,
            "macro_kappa_enabled": False,
            "macro_direction_enabled": False,
            "cs_overlay_enabled": False,
            "ml_overlay_enabled": False,
            "gap_input_dir": str(tmp_path),
        },
    )
    matrix_dir = tmp_path / "matrices"
    matrix_dir.mkdir()
    np.save(matrix_dir / "mu_gap_20260814.npy", np.ones(len(JP_TICKERS)))
    np.save(matrix_dir / "omega_gap_20260814.npy", np.eye(len(JP_TICKERS)))

    result = ProductionV2Model(cfg).decide(
        "2026-08-14",
        gap_input_dir=tmp_path,
        df_exec=None,
        current_prices=None,
        overlay_enabled=False,
    )
    assert np.allclose(result.w_final, 0.0)
    assert result.fallback["audit_failure"] is True
    assert result.diagnostics["distribution_provenance"]["status"] == "rejected"
    assert any("provenance missing" in alert.lower() for alert in result.alerts)


class _PartialBroker:
    def submit_orders_batch(self, orders, **kwargs):
        return [
            OrderResult(
                order_id=f"partial-{i}",
                status=OrderStatus.PARTIALLY_FILLED,
                ticker=order.ticker,
                side=order.side,
                quantity=order.quantity,
            )
            for i, order in enumerate(orders)
        ]


def test_partial_submission_reconciles_before_propagating(tmp_path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {"ticker": ["1617.T"], "action": ["BUY"], "quantity": [100]}
    )
    with pytest.raises(OrderExecutionIncomplete) as raised:
        submit_orders_via_api(frame, _PartialBroker(), tmp_path)
    assert raised.value.summary["partial_orders_count"] == 1
    assert (tmp_path / "api_execution_log.json").exists()

    called: list[str] = []
    decision = {"trade_date": pd.Timestamp("2026-08-14")}
    with monkeypatch.context() as patch:
        patch.setattr(post_decision, "save_decision_output", lambda *a, **k: "decision.csv")
        patch.setattr(post_decision, "submit_orders_via_api", lambda **k: (_ for _ in ()).throw(
            OrderExecutionIncomplete("incomplete", raised.value.summary, str(tmp_path / "api_execution_log.json"))
        ))
        patch.setattr(post_decision, "save_position_snapshot", lambda *a, **k: called.append("positions") or None)
        patch.setattr(post_decision, "save_wallet_snapshot", lambda *a, **k: called.append("wallet") or None)
        patch.setattr(post_decision, "save_daily_journal", lambda *a, **k: called.append("journal") or "journal")
        patch.setattr(post_decision, "fetch_fill_prices", lambda *a, **k: called.append("fills"))
        with pytest.raises(OrderExecutionIncomplete):
            post_decision._write_decision_output_and_submit(
                frame, decision, tmp_path, False, _PartialBroker(), None
            )
    assert called == ["fills", "positions", "wallet", "journal"]


def test_cancelled_order_still_queries_accumulated_fill() -> None:
    from leadlag.broker.tachibana.client import TachibanaBrokerClient
    from leadlag.execution.pricing import fetch_fill_prices

    calls: list[tuple] = []
    api = object.__new__(TachibanaBrokerClient)
    api._client = SimpleNamespace(
        get_order_detail=lambda *args: calls.append(args)
        or {"sYakuzyouPrice": "1000", "sYakuzyouSuryou": "30", "sOrderStatus": "7"}
    )
    rows = [{"order_id": "O-1", "status": "CANCELLED", "eigyou_day": "20260814"}]
    fetch_fill_prices(api, rows, wait_seconds=0.0)
    assert len(calls) == 1
    assert rows[0]["fill_quantity"] == 30


def test_fill_query_failure_preserves_previously_confirmed_quantity() -> None:
    from leadlag.broker.tachibana.client import TachibanaBrokerClient
    from leadlag.execution.pricing import FillPriceReconciliationError, fetch_fill_prices

    api = object.__new__(TachibanaBrokerClient)
    api._client = SimpleNamespace(
        get_order_detail=lambda *args: {"sYakuzyouPrice": "1000", "sYakuzyouSuryou": "30"}
    )
    row = {"order_id": "O-1", "status": "CANCELLED", "fill_quantity": 30, "fill_price": 1000.0}
    fetch_fill_prices(api, [row], wait_seconds=0.0)
    api._client.get_order_detail = lambda *args: (_ for _ in ()).throw(OSError("temporary"))
    with pytest.raises(FillPriceReconciliationError, match="O-1"):
        fetch_fill_prices(api, [row], wait_seconds=0.0)
    assert row["fill_quantity"] == 30
    assert row["fill_price"] == 1000.0


def test_post_decision_marks_fill_detail_failure_incomplete(tmp_path, monkeypatch) -> None:
    """A terminal submit result still fails the command when detail lookup fails."""
    frame = pd.DataFrame({"ticker": ["1617.T"], "action": ["BUY"], "quantity": [100]})
    decision = {"trade_date": pd.Timestamp("2026-08-14")}
    summary = {
        "buy_results": [{"order_id": "O-1", "status": "FILLED", "ticker": "1617.T"}],
        "sell_results": [],
        "close_results": [],
    }
    with monkeypatch.context() as patch:
        patch.setattr(post_decision, "save_decision_output", lambda *a, **k: "decision.csv")
        patch.setattr(post_decision, "submit_orders_via_api", lambda **k: summary)
        patch.setattr(
            post_decision,
            "fetch_fill_prices",
            lambda *a, **k: (_ for _ in ()).throw(OSError("detail endpoint unavailable")),
        )
        patch.setattr(post_decision, "save_position_snapshot", lambda *a, **k: None)
        patch.setattr(post_decision, "save_wallet_snapshot", lambda *a, **k: None)
        patch.setattr(post_decision, "save_daily_journal", lambda *a, **k: "journal.json")
        with pytest.raises(RuntimeError, match="Post-decision reconciliation incomplete"):
            post_decision._write_decision_output_and_submit(
                frame, decision, tmp_path, False, object(), None
            )
    persisted = json.loads((tmp_path / "api_execution_log.json").read_text(encoding="utf-8"))
    assert "fill_prices: detail endpoint unavailable" in persisted["reconciliation_errors"][0]


def test_close_marks_terminal_orders_incomplete_when_fill_reconciliation_fails(
    tmp_path, monkeypatch
) -> None:
    from leadlag.broker.base import Position
    from leadlag.execution import close as close_module

    position = Position(
        ticker="1617.T",
        side="BUY",
        quantity=100,
        price=1000.0,
        exchange=27,
        execution_id="position-1",
    )
    api = SimpleNamespace(
        get_positions=lambda: [position],
        submit_orders_batch=lambda orders, **kwargs: [
            OrderResult(
                order_id="C-1",
                status=OrderStatus.FILLED,
                ticker=order.ticker,
                side=order.side,
                quantity=order.quantity,
                order_type=order.order_type,
            )
            for order in orders
        ],
    )
    monkeypatch.setattr(
        close_module,
        "fetch_fill_prices",
        lambda *a, **k: (_ for _ in ()).throw(OSError("detail endpoint unavailable")),
    )
    summary = close_module.close_all_positions(api, tmp_path, dry_run=False)
    assert summary["filled_orders_count"] == 1
    assert summary["close_incomplete"] is True
    assert "fill_prices: detail endpoint unavailable" in summary["reconciliation_errors"]


class _FilledBroker:
    def submit_orders_batch(self, orders, **_kwargs):
        return [
            OrderResult(
                order_id=f"filled-{index}",
                status=OrderStatus.FILLED,
                ticker=order.ticker,
                side=order.side,
                quantity=order.quantity,
                order_type=order.order_type,
            )
            for index, order in enumerate(orders)
        ]


def test_initial_execution_log_failure_is_reconciled_before_rethrow(tmp_path, monkeypatch) -> None:
    """A failed initial log write must not bypass post-submission reconciliation."""
    frame = pd.DataFrame({"ticker": ["1617.T"], "action": ["BUY"], "quantity": [100]})
    broker = _FilledBroker()
    with monkeypatch.context() as patch:
        patch.setattr(
            "leadlag.execution.broker_ops._write_api_execution_log",
            lambda *a, **k: (_ for _ in ()).throw(OSError("disk unavailable")),
        )
        with pytest.raises(OrderExecutionIncomplete) as raised:
            submit_orders_via_api(frame, broker, tmp_path)

    assert raised.value.summary["execution_log_errors"] == [
        "api_execution_log: disk unavailable"
    ]
    decision = {"trade_date": pd.Timestamp("2026-08-14")}
    calls: list[str] = []
    with monkeypatch.context() as patch:
        patch.setattr(post_decision, "save_decision_output", lambda *a, **k: "decision.csv")
        patch.setattr(
            post_decision,
            "submit_orders_via_api",
            lambda **k: (_ for _ in ()).throw(raised.value),
        )
        patch.setattr(post_decision, "fetch_fill_prices", lambda *a, **k: calls.append("fills"))
        patch.setattr(post_decision, "save_position_snapshot", lambda *a, **k: calls.append("positions"))
        patch.setattr(post_decision, "save_wallet_snapshot", lambda *a, **k: calls.append("wallet"))
        patch.setattr(post_decision, "save_daily_journal", lambda *a, **k: calls.append("journal"))
        with pytest.raises(OrderExecutionIncomplete):
            post_decision._write_decision_output_and_submit(
                frame, decision, tmp_path, False, broker, None
            )
    persisted = json.loads((tmp_path / "api_execution_log.json").read_text(encoding="utf-8"))
    assert persisted["execution_log_errors"] == ["api_execution_log: disk unavailable"]
    assert "api_execution_log: disk unavailable" in persisted["reconciliation_errors"]
    assert calls == ["fills", "positions", "wallet", "journal"]


def test_empty_position_snapshot_is_distinct_from_query_failure(tmp_path) -> None:
    from leadlag.execution.output_ops import save_position_snapshot

    empty_client = SimpleNamespace(get_positions=lambda: [])
    snapshot_path = save_position_snapshot(empty_client, tmp_path, raise_on_error=True)
    assert snapshot_path is not None
    snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    assert snapshot["position_count"] == 0
    assert snapshot["positions"] == []


def test_close_cli_returns_nonzero_for_all_rejected_orders(monkeypatch, tmp_path) -> None:
    from leadlag.cli import main
    from leadlag.execution import close as close_module

    summary = {
        "close_incomplete": True,
        "filled_orders_count": 0,
        "pending_orders_count": 0,
        "failed_orders_count": 1,
        "close_results": [{"status": "FAILED", "ticker": "1617.T", "quantity": 100}],
    }
    calls: list[str] = []
    fake_client = SimpleNamespace(close=lambda: calls.append("close"))
    with monkeypatch.context() as patch:
        patch.setattr("leadlag.core.market_calendar.is_market_closed", lambda _: False)
        patch.setattr(close_module, "build_api_client", lambda *a, **k: fake_client)
        patch.setattr(close_module, "build_output_dir", lambda *a, **k: str(tmp_path))
        patch.setattr(close_module, "close_all_positions", lambda *a, **k: summary)
        patch.setattr(close_module, "save_position_snapshot", lambda *a, **k: None)
        patch.setattr(close_module, "save_wallet_snapshot", lambda *a, **k: None)
        patch.setattr(close_module, "save_daily_journal", lambda *a, **k: None)
        code = main(["close"])
    assert code == 2
    assert calls == ["close"]


def test_gap_metadata_is_used_for_leakage_signal_date(tmp_path) -> None:
    store_path = tmp_path / "gap.sqlite"
    GapStore(store_path).save(
        "2026-08-14",
        np.zeros(len(JP_TICKERS)),
        np.eye(len(JP_TICKERS)),
        metadata={"sig_date": "2026-08-17"},
    )
    assert _derive_signal_date(store_path, "2026-08-14") == "2026-08-17"


def test_overlay_publish_failure_keeps_previous_active_version(tmp_path, monkeypatch) -> None:
    import leadlag.models.ml_order_overlay as overlay

    old = MLOrderOverlayModel(
        lgbm=SimpleNamespace(marker="old"),
        cont_cols=[],
        target_std=1.0,
        use_ticker=False,
        use_classification=False,
        per_ticker_interactions=False,
    )
    metadata = {
        "metadata_status": "verified",
        "train_start": "2015-01-05",
        "train_end": "2026-08-13",
        "data_hash": "old-data",
        "config_hash": "old-config",
    }
    save_overlay_model(old, tmp_path, training_metadata=metadata)
    previous = (tmp_path / "CURRENT").read_text(encoding="utf-8")
    new = MLOrderOverlayModel(
        lgbm=SimpleNamespace(marker="new"),
        cont_cols=[],
        target_std=1.0,
        use_ticker=False,
        use_classification=False,
        per_ticker_interactions=False,
    )
    original_replace = overlay.os.replace

    def fail_current(source, destination):
        if destination == str(tmp_path / "CURRENT") or destination == tmp_path / "CURRENT":
            raise OSError("injected publication failure")
        return original_replace(source, destination)

    monkeypatch.setattr(overlay.os, "replace", fail_current)
    with pytest.raises(OSError):
        save_overlay_model(new, tmp_path, training_metadata={**metadata, "train_end": "2026-08-14"})
    assert (tmp_path / "CURRENT").read_text(encoding="utf-8") == previous
    assert load_overlay_model(tmp_path).lgbm.marker == "old"


def test_overlay_loader_rejects_active_model_digest_mismatch(tmp_path) -> None:
    model = MLOrderOverlayModel(
        lgbm=SimpleNamespace(marker="active"),
        cont_cols=[],
        target_std=1.0,
        use_ticker=False,
        use_classification=False,
        per_ticker_interactions=False,
    )
    metadata = {
        "metadata_status": "verified",
        "train_start": "2015-01-05",
        "train_end": "2026-08-13",
        "data_hash": "data",
        "config_hash": "config",
    }
    save_overlay_model(model, tmp_path, training_metadata=metadata)
    active = (tmp_path / "CURRENT").read_text(encoding="utf-8").strip()
    model_path = tmp_path / "versions" / active / "model.pkl"
    tampered = MLOrderOverlayModel(
        lgbm=SimpleNamespace(marker="tampered"),
        cont_cols=[],
        target_std=1.0,
        use_ticker=False,
        use_classification=False,
        per_ticker_interactions=False,
    )
    model_path.write_bytes(pickle.dumps(tampered))
    with pytest.raises(ValueError, match="digest mismatch"):
        load_overlay_model(tmp_path)


def test_overlay_loader_rejects_path_traversal_current_pointer(tmp_path) -> None:
    (tmp_path / "CURRENT").write_text("..\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid active overlay artifact version pointer"):
        load_overlay_model(tmp_path)


def test_overlay_loader_rejects_external_provenance_mismatch(tmp_path) -> None:
    model = MLOrderOverlayModel(
        lgbm=SimpleNamespace(marker="active"),
        cont_cols=[],
        target_std=1.0,
        use_ticker=False,
        use_classification=False,
        per_ticker_interactions=False,
    )
    save_overlay_model(
        model,
        tmp_path,
        training_metadata={
            "metadata_status": "verified",
            "train_start": "2015-01-05",
            "train_end": "2026-08-13",
            "data_hash": "data",
            "config_hash": "config",
        },
    )
    active = (tmp_path / "CURRENT").read_text(encoding="utf-8").strip()
    metadata_path = tmp_path / "versions" / active / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["train_end"] = "2020-12-31"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="field mismatch"):
        load_overlay_model(tmp_path)


@pytest.mark.parametrize("invalid_train_end", [None, pd.NaT])
def test_overlay_publication_and_application_reject_null_train_end(
    tmp_path, invalid_train_end
) -> None:
    metadata = {
        "metadata_status": "verified",
        "train_start": "2015-01-05",
        "train_end": invalid_train_end,
        "data_hash": "data",
        "config_hash": "config",
    }
    model = MLOrderOverlayModel(
        lgbm=SimpleNamespace(predict=lambda x: np.zeros(len(x))),
        cont_cols=["score"],
        target_std=1.0,
        use_ticker=False,
        use_classification=False,
        per_ticker_interactions=False,
        metadata={"metadata_version": 2, **metadata},
    )
    with pytest.raises(ValueError, match="train_end"):
        save_overlay_model(model, tmp_path, training_metadata=metadata)
    with pytest.raises(ValueError, match="train_end"):
        apply_overlay(
            _overlay_decision(),
            pd.DataFrame(),
            model,
            "2026-08-14",
        )


def test_overlay_loader_rejects_legacy_root_even_when_metadata_looks_verified(tmp_path) -> None:
    embedded_metadata = {
        "metadata_version": 2,
        "metadata_status": "verified",
        "train_start": "2015-01-05",
        "train_end": "2026-08-13",
        "data_hash": "embedded-data",
        "config_hash": "embedded-config",
    }
    model = MLOrderOverlayModel(
        lgbm=SimpleNamespace(marker="legacy"),
        cont_cols=[],
        target_std=1.0,
        use_ticker=False,
        use_classification=False,
        per_ticker_interactions=False,
        metadata=embedded_metadata,
    )
    (tmp_path / "model.pkl").write_bytes(pickle.dumps(model))
    (tmp_path / "metadata.json").write_text(
        json.dumps({**embedded_metadata, "train_end": "2020-12-31"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Legacy root overlay artifact is not accepted"):
        load_overlay_model(tmp_path)


def test_var_cache_fingerprint_tracks_only_current_overlay_version(tmp_path) -> None:
    artifact_root = tmp_path / "overlay"
    active = artifact_root / "versions" / "active"
    inactive = artifact_root / "versions" / "inactive"
    staging = artifact_root / ".staging-new"
    for directory in (active, inactive, staging):
        directory.mkdir(parents=True)
    (artifact_root / "CURRENT").write_text("active\n", encoding="utf-8")
    (active / "model.pkl").write_bytes(b"active-v1")
    (inactive / "model.pkl").write_bytes(b"inactive-v1")
    (staging / "model.pkl").write_bytes(b"staging-v1")

    initial = _active_overlay_artifact_fingerprint(artifact_root)
    (inactive / "model.pkl").write_bytes(b"inactive-v2")
    (staging / "model.pkl").write_bytes(b"staging-v2")
    assert _active_overlay_artifact_fingerprint(artifact_root) == initial
    (active / "model.pkl").write_bytes(b"active-v2")
    assert _active_overlay_artifact_fingerprint(artifact_root) != initial

    (artifact_root / "CURRENT").write_text("inactive\n", encoding="utf-8")
    assert _active_overlay_artifact_fingerprint(artifact_root) != initial


def test_overlay_save_does_not_accept_conflicting_structural_metadata(tmp_path) -> None:
    model = MLOrderOverlayModel(
        lgbm=SimpleNamespace(marker="active"),
        cont_cols=["score"],
        target_std=1.0,
        use_ticker=False,
        use_classification=False,
        per_ticker_interactions=False,
    )
    with pytest.raises(ValueError, match="conflicts with fitted overlay field: cont_cols"):
        save_overlay_model(
            model,
            tmp_path,
            training_metadata={
                "cont_cols": ["different"],
                "metadata_status": "verified",
                "train_start": "2015-01-05",
                "train_end": "2026-08-13",
                "data_hash": "data",
                "config_hash": "config",
            },
        )


def test_jp_holiday_nan_padding_does_not_remove_following_sessions() -> None:
    dates = pd.bdate_range("2026-08-03", "2026-08-14")
    us = pd.DataFrame(100.0, index=dates, columns=US_TICKERS)
    jp = pd.DataFrame(100.0, index=dates, columns=JP_TICKERS + [TOPIX_TICKER])
    jp.loc["2026-08-11"] = np.nan
    raw = {"us_close": us, "jp_close": jp, "jp_open": jp.copy()}
    result = preprocess_data(raw, strict_validation=True, beta_window=2)
    assert pd.Timestamp("2026-08-12") in result.index
    assert pd.Timestamp("2026-08-13") in result.index


def test_in_sample_overlay_application_fails_closed() -> None:
    n_j = len(JP_TICKERS)
    trade_date = pd.Timestamp("2026-08-13")
    dates = pd.date_range(trade_date - pd.Timedelta(days=30), trade_date, freq="B")
    df_exec = pd.DataFrame(
        {
            "topix_night_return": np.zeros(len(dates)),
            **{f"jp_gap_{tk}": np.zeros(len(dates)) for tk in JP_TICKERS},
            **{f"jp_beta_{tk}": np.ones(len(dates)) for tk in JP_TICKERS},
            **{f"jp_oc_{tk}": np.zeros(len(dates)) for tk in JP_TICKERS},
        },
        index=dates,
    )
    result = _overlay_decision(n_j)
    model = MLOrderOverlayModel(
        lgbm=SimpleNamespace(predict=lambda x: np.zeros(len(x))),
        cont_cols=["score"],
        target_std=1.0,
        use_ticker=False,
        use_classification=False,
        per_ticker_interactions=False,
        metadata={
            "metadata_version": 2,
            "metadata_status": "verified",
            "train_start": "2015-01-05",
            "train_end": "2026-08-14",
            "data_hash": "data",
            "config_hash": "config",
        },
    )
    with pytest.raises(ValueError, match="cannot be applied in-sample"):
        apply_overlay(result, df_exec, model, trade_date.strftime("%Y-%m-%d"))


def test_drawdown_series_includes_initial_wealth() -> None:
    drawdown = compute_drawdown_series(pd.Series([-0.10, 0.0]))
    np.testing.assert_allclose(drawdown.to_numpy(), [-0.10, -0.10])
