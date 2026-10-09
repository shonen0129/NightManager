from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from leadlag.cli import setup_parser
from leadlag.data.tickers import JP_TICKERS
from leadlag.domain.portfolio import CostBreakdown, PortfolioDecision
from leadlag.execution import v2_bridge
from leadlag.execution.config import load_config_from_yaml
from leadlag.reporting import ml_overlay_shadow


class _V2Config:
    def __init__(self, ml_overlay_enabled: bool = True) -> None:
        self.ml_overlay_enabled = ml_overlay_enabled
        self.ml_overlay_model_dir = "models/ml_order_overlay/test"
        self.cost_bps_per_gross = 10.0

    def model_dump(self, *, mode: str) -> dict:
        return {
            "ml_overlay_enabled": self.ml_overlay_enabled,
            "ml_overlay_model_dir": self.ml_overlay_model_dir,
            "cost_bps_per_gross": self.cost_bps_per_gross,
        }


class _AppConfig:
    def __init__(self, v2: _V2Config | None = None) -> None:
        self.v2 = v2 or _V2Config()

@dataclass
class _FakeInputs:
    known: SimpleNamespace
    version: SimpleNamespace


def _decision(weight: float) -> PortfolioDecision:
    return PortfolioDecision(
        w_final=np.array([weight, -weight]),
        scores=np.array([0.3, -0.3]),
        mu_gap=np.array([0.01, -0.01]),
        sigma_gap=np.array([0.02, 0.02]),
        Omega_gap=np.eye(2) * 0.0004,
        fallback={"gap_data_missing": False},
        pit_binning={"assigned_bin": "mid"},
        leakage={"status": "PASSED"},
        numerical={"status": "PASSED"},
        alerts=[],
        summary={"predicted_portfolio_ir": 1.2},
        run_config=SimpleNamespace(),
        scores_overlay=np.array([0.4, -0.4]),
        costs=CostBreakdown(slippage=0.001),
    )


def _inputs() -> _FakeInputs:
    return _FakeInputs(
        known=SimpleNamespace(
            trade_date=pd.Timestamp("2026-09-28"),
            as_of=pd.Timestamp("2026-09-28 09:10"),
            source="test_live_inputs",
            ticker_order=("A", "B"),
            current_prices={"A": 100.0, "B": 200.0},
            prev_closes={"A": 99.0, "B": 201.0},
            price_sources={"A": "broker", "B": "broker"},
        ),
        version=SimpleNamespace(
            schema_version="decision-inputs-v1",
            digest="digest",
            known_fingerprint="known",
            historical_fingerprint="historical",
        ),
    )


def test_append_ml_overlay_shadow_records_pair_and_is_idempotent(tmp_path, monkeypatch):
    calls: list[bool] = []

    class _Runner:
        def __init__(self, config: _AppConfig) -> None:
            calls.append(config.v2.ml_overlay_enabled)
            self.model = SimpleNamespace(
                _overlay_model=SimpleNamespace(
                    metadata={"artifact_version": "model-v1", "model_sha256": "abc"}
                )
            )

        def run(self, inputs: _FakeInputs) -> PortfolioDecision:
            assert inputs.known.trade_date == pd.Timestamp("2026-09-28")
            return _decision(0.3)

    monkeypatch.setattr(ml_overlay_shadow, "ProductionRunner", _Runner)
    args = dict(
        app_config=_AppConfig(_V2Config(ml_overlay_enabled=False)),
        overlay_config=_AppConfig(_V2Config(ml_overlay_enabled=True)),
        decision_inputs=_inputs(),
        ml_disabled_result=_decision(0.2),
        output_dir=tmp_path,
    )

    path = ml_overlay_shadow.append_ml_overlay_shadow(**args)
    ml_overlay_shadow.append_ml_overlay_shadow(**args)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    assert calls == [True, True]
    assert rows[0]["schema_version"] == "ml-overlay-research-paired-shadow-v2"
    assert rows[0]["status"] == "complete"
    assert rows[0]["variants"]["ml_enabled"]["weights"] == {"A": 0.3, "B": -0.3}
    assert rows[0]["variants"]["ml_disabled"]["weights"] == {"A": 0.2, "B": -0.2}
    assert rows[0]["overlay_artifact"]["artifact_version"] == "model-v1"
    assert rows[0]["cost_evidence"]["status"] == "modeled_only_pending_real_execution_reconciliation"


def test_append_ml_overlay_shadow_records_candidate_failure(tmp_path, monkeypatch):
    class _Runner:
        def __init__(self, config: _AppConfig) -> None:
            self.model = SimpleNamespace(_overlay_model=SimpleNamespace(metadata={}))

        def run(self, inputs: _FakeInputs) -> PortfolioDecision:
            raise RuntimeError("candidate unavailable")

    monkeypatch.setattr(ml_overlay_shadow, "ProductionRunner", _Runner)
    path = ml_overlay_shadow.append_ml_overlay_shadow(
        app_config=_AppConfig(_V2Config(ml_overlay_enabled=False)),
        overlay_config=_AppConfig(_V2Config(ml_overlay_enabled=True)),
        decision_inputs=_inputs(),
        ml_disabled_result=_decision(0.2),
        output_dir=tmp_path,
    )

    row = json.loads(path.read_text().splitlines()[0])
    assert row["status"] == "overlay_failed"
    assert row["variants"]["ml_enabled"] is None
    assert row["variants"]["ml_disabled"]["weights"] == {"A": 0.2, "B": -0.2}
    assert row["overlay_error"] == "RuntimeError: candidate unavailable"


def test_shadow_only_candidate_failure_is_persisted_then_raises(tmp_path, monkeypatch):
    class _Runner:
        def __init__(self, config: _AppConfig) -> None:
            self.model = SimpleNamespace(_overlay_model=SimpleNamespace(metadata={}))

        def run(self, inputs: _FakeInputs) -> PortfolioDecision:
            raise RuntimeError("candidate unavailable")

    monkeypatch.setattr(ml_overlay_shadow, "ProductionRunner", _Runner)
    try:
        ml_overlay_shadow.append_ml_overlay_shadow(
            app_config=_AppConfig(_V2Config(ml_overlay_enabled=False)),
            overlay_config=_AppConfig(_V2Config(ml_overlay_enabled=True)),
            decision_inputs=_inputs(),
            ml_disabled_result=_decision(0.2),
            output_dir=tmp_path,
            raise_on_overlay_failure=True,
        )
    except RuntimeError as exc:
        assert "incomplete shadow row is recorded" in str(exc)
    else:
        raise AssertionError("shadow-only mode must fail closed on an incomplete pair")

    row = json.loads((tmp_path / "daily.jsonl").read_text().splitlines()[0])
    assert row["status"] == "overlay_failed"


def test_shadow_requires_v2_only_production_and_enabled_candidate_configs(tmp_path):
    try:
        ml_overlay_shadow.append_ml_overlay_shadow(
            app_config=_AppConfig(_V2Config(ml_overlay_enabled=True)),
            overlay_config=_AppConfig(_V2Config(ml_overlay_enabled=True)),
            decision_inputs=_inputs(),
            ml_disabled_result=_decision(0.2),
            output_dir=tmp_path,
        )
    except ValueError as exc:
        assert "production V2-only config" in str(exc)
    else:
        raise AssertionError("production overlay config must be rejected")


def test_research_shadow_config_changes_only_overlay_settings():
    production = load_config_from_yaml("configs/production/production.yaml", strict=True)
    research = load_config_from_yaml(
        "configs/research/ml_overlay_forward_shadow_20261009.yaml", strict=True
    )

    assert ml_overlay_shadow._without_overlay_settings(production) == (
        ml_overlay_shadow._without_overlay_settings(research)
    )


def test_decision_cli_accepts_shadow_output_dir():
    args = setup_parser().parse_args(
        [
            "decision",
            "--ml-overlay-shadow-dir",
            "var/shadow_runs/ml_overlay_research_20261009",
            "--shadow-only",
        ]
    )
    assert args.ml_overlay_shadow_dir == "var/shadow_runs/ml_overlay_research_20261009"
    assert args.ml_overlay_shadow_config == "configs/research/ml_overlay_forward_shadow_20261009.yaml"
    assert args.shadow_only is True


def test_daily_cli_does_not_accept_shadow_only():
    with pytest.raises(SystemExit):
        setup_parser().parse_args(["daily", "--shadow-only"])


def test_shadow_only_rejects_dry_run_before_loading_inputs(monkeypatch, tmp_path):
    config = SimpleNamespace(v2=SimpleNamespace(ml_overlay_enabled=False))
    overlay_config = SimpleNamespace(v2=SimpleNamespace(ml_overlay_enabled=True))
    monkeypatch.setattr(
        v2_bridge,
        "load_config_from_yaml",
        lambda _path, **_kwargs: overlay_config if "research" in str(_path) else config,
    )
    monkeypatch.setattr(v2_bridge, "_resolve_trade_date", lambda _date, _path: "2026-09-28")
    monkeypatch.setattr(v2_bridge, "_resolve_gap_dir", lambda *_args: None)
    monkeypatch.setattr(
        v2_bridge,
        "_load_df_exec",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("inputs must not load")),
    )

    try:
        v2_bridge.run_v2_decision(
            config_path="unused.yaml",
            live_dir=tmp_path,
            api_enable=True,
            dry_run=True,
            ml_overlay_shadow_dir=tmp_path / "shadow",
            shadow_only=True,
        )
    except ValueError as exc:
        assert "live read-only market data" in str(exc)
    else:
        raise AssertionError("shadow-only mode must reject dry runs")


def test_shadow_only_returns_before_production_writes_positions_or_orders(monkeypatch, tmp_path):
    previous_closes = {ticker: 100.0 for ticker in JP_TICKERS}
    current_prices = {ticker: 101.0 for ticker in JP_TICKERS}
    decision_inputs = _inputs()
    config = SimpleNamespace(
        v2=SimpleNamespace(
            ml_overlay_enabled=False,
            macro_kappa_enabled=False,
            macro_direction_enabled=False,
            cs_overlay_enabled=False,
        ),
        broker_provider="tachibana",
        gap_distribution_dir=None,
    )

    class _ApiClient:
        closed = False

        def close(self) -> None:
            self.closed = True

    api_client = _ApiClient()

    class _Lake:
        def __init__(self, df_exec) -> None:
            self.df_exec = df_exec

        def history_frame(self):
            return self.df_exec.copy()

        def get_snapshot(self, as_of):
            return SimpleNamespace(
                as_of=as_of,
                trade_date=pd.Timestamp("2026-09-28"),
                us_returns=np.zeros(15),
                jp_betas=np.zeros((len(JP_TICKERS), 15)),
                topix_night_return=0.0,
                prev_closes={**previous_closes, "1306.T": 3000.0},
            )

        def build_decision_inputs(self, *_args, **_kwargs):
            return decision_inputs

    class _Snapshot:
        def __init__(self, **_kwargs) -> None:
            pass

        def validate(self):
            return True, []

    class _Runner:
        def __init__(self, _config) -> None:
            self.model = SimpleNamespace(_overlay_model=SimpleNamespace(metadata={"version": "test"}))

        def run(self, _inputs):
            return _decision(0.3)

    shadow_path = tmp_path / "shadow" / "daily.jsonl"
    shadow_calls = []
    overlay_config = SimpleNamespace(v2=SimpleNamespace(ml_overlay_enabled=True))
    monkeypatch.setattr(
        v2_bridge,
        "load_config_from_yaml",
        lambda _path, **_kwargs: overlay_config if "research" in str(_path) else config,
    )
    monkeypatch.setattr(v2_bridge, "_resolve_trade_date", lambda _date, _path: "2026-09-28")
    monkeypatch.setattr(v2_bridge, "_resolve_gap_dir", lambda *_args: None)
    monkeypatch.setattr(v2_bridge, "_load_df_exec", lambda *_args, **_kwargs: pd.DataFrame(index=[pd.Timestamp("2026-09-28")]))
    monkeypatch.setattr(v2_bridge, "load_current_prices_cache", lambda _date: (current_prices, 3000.0))
    monkeypatch.setattr(
        v2_bridge,
        "load_frozen_quote_snapshot",
        lambda *_args, **_kwargs: SimpleNamespace(
            as_of=pd.Timestamp("2026-09-28T09:10:05+09:00"),
            prices={**current_prices, "1306.T": 3000.0},
            observed_at={
                ticker: "2026-09-28T09:10:05+09:00" for ticker in JP_TICKERS
            },
            snapshot_id="frozen-test-quote",
        ),
    )
    monkeypatch.setattr(v2_bridge, "build_api_client", lambda **_kwargs: api_client)
    monkeypatch.setattr(v2_bridge, "PITDataLake", _Lake)
    monkeypatch.setattr(v2_bridge, "MarketSnapshot", _Snapshot)
    monkeypatch.setattr(v2_bridge, "build_open_910_returns", lambda *_args: np.zeros((1, len(JP_TICKERS))))
    monkeypatch.setattr(v2_bridge, "HistoricalInputs", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(v2_bridge.adr_data, "load_adr_features", lambda: None)
    monkeypatch.setattr(v2_bridge, "ProductionRunner", _Runner)
    monkeypatch.setattr(
        v2_bridge,
        "append_ml_overlay_shadow",
        lambda **kwargs: (shadow_calls.append(kwargs) or shadow_path),
    )

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("shadow-only mode entered the production execution path")

    monkeypatch.setattr(v2_bridge, "write_production_files", _forbidden)
    monkeypatch.setattr(v2_bridge, "fetch_current_positions", _forbidden)
    monkeypatch.setattr(v2_bridge, "execute_post_decision_flow", _forbidden)
    monkeypatch.setattr(v2_bridge, "build_output_dir", _forbidden)

    result = v2_bridge.run_v2_decision(
        config_path="unused.yaml",
        live_dir=tmp_path / "live",
        api_enable=True,
        ml_overlay_shadow_dir=tmp_path / "shadow",
        shadow_only=True,
    )

    assert result == str(shadow_path)
    assert shadow_calls[0]["raise_on_overlay_failure"] is True
    assert api_client.closed is True


@pytest.mark.parametrize("shadow_only", [False, True])
def test_candidate_config_failure_preserves_live_v2_but_stops_shadow_only(
    monkeypatch, tmp_path, shadow_only
):
    config = SimpleNamespace(v2=SimpleNamespace(ml_overlay_enabled=False))
    baseline = _decision(0.2)
    written = []
    client = SimpleNamespace(close=lambda: None)

    def load_config(path, **_kwargs):
        if str(path).endswith("missing-candidate.yaml"):
            raise FileNotFoundError("candidate config unavailable")
        return config

    class ProductionWriteReached(Exception):
        pass

    def write_baseline(_date, _path, result, **_kwargs):
        written.append(result)
        raise ProductionWriteReached

    monkeypatch.setattr(v2_bridge, "load_config_from_yaml", load_config)
    monkeypatch.setattr(v2_bridge, "_resolve_trade_date", lambda *_args: "2026-09-28")
    monkeypatch.setattr(v2_bridge, "_resolve_gap_dir", lambda *_args: None)
    monkeypatch.setattr(v2_bridge, "_load_df_exec", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(v2_bridge, "load_frozen_quote_snapshot", lambda *_args, **_kwargs: SimpleNamespace(
        as_of=pd.Timestamp("2026-09-28T09:10:05+09:00"), snapshot_id="frozen-test"
    ))
    monkeypatch.setattr(v2_bridge, "build_api_client", lambda **_kwargs: client)
    monkeypatch.setattr(v2_bridge, "_resolve_current_price_preflight", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(v2_bridge, "_build_quote_preflight", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(v2_bridge, "_build_run_owned_decision_inputs", lambda *_args: _inputs())
    monkeypatch.setattr(v2_bridge, "ProductionRunner", lambda _config: SimpleNamespace(run=lambda _inputs: baseline))
    monkeypatch.setattr(v2_bridge, "write_production_files", write_baseline)

    error = FileNotFoundError if shadow_only else ProductionWriteReached
    with pytest.raises(error):
        v2_bridge.run_v2_decision(
            config_path="production.yaml", live_dir=tmp_path,
            api_enable=True, ml_overlay_shadow_dir=tmp_path / "shadow",
            ml_overlay_shadow_config="missing-candidate.yaml", shadow_only=shadow_only,
        )
    assert written == ([] if shadow_only else [baseline])
