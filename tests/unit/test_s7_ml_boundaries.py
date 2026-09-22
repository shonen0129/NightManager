"""Regression checks for the S7 production/research ML boundary."""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import research.experiments.ml_overlay_training as training
from leadlag.config.schemas import ProductionV2RunConfig
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.domain.inputs import DecisionInputs
from leadlag.models.ml_order_overlay import MLOrderOverlayModel
from leadlag.models.ml_order_overlay import train_overlay_model as production_train
from leadlag.models.ml_overlay_artifact import load_overlay_model, save_overlay_model
from leadlag.models.ml_overlay_features import _build_ticker_features, overlay_continuous_columns
from leadlag.models.ml_overlay_inference import apply_overlay
from research.experiments.ml_overlay_training import train_overlay_model

ROOT = Path(__file__).resolve().parents[2]


def _training_cfg() -> SimpleNamespace:
    return SimpleNamespace(model_dump=lambda **_: {})


def test_pickle_class_path_and_canonical_module_boundaries() -> None:
    assert MLOrderOverlayModel.__module__ == "leadlag.models.ml_order_overlay"
    assert inspect.getmodule(_build_ticker_features).__name__ == (
        "leadlag.models.ml_overlay_features"
    )
    assert inspect.getmodule(apply_overlay).__name__ == "leadlag.models.ml_overlay_inference"
    assert inspect.getmodule(save_overlay_model).__name__ == "leadlag.models.ml_overlay_artifact"
    assert inspect.getmodule(load_overlay_model).__name__ == "leadlag.models.ml_overlay_artifact"
    assert inspect.getmodule(train_overlay_model).__name__ == (
        "research.experiments.ml_overlay_training"
    )
    assert overlay_continuous_columns(False)[0:3] == ["score", "mu_gap", "sigma_gap"]
    assert f"ticker_{JP_TICKERS[0]}_score" in overlay_continuous_columns(True)


def test_production_overlay_import_does_not_load_research(tmp_path) -> None:
    source = (ROOT / "src/leadlag/models/ml_order_overlay.py").read_text(encoding="utf-8")
    assert "from research" not in source
    assert "import research" not in source
    code = (
        "import sys; import leadlag.models.ml_order_overlay; "
        "assert not any(k == 'research' or k.startswith('research.') for k in sys.modules)"
    )
    script = tmp_path / "check_overlay_import.py"
    script.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_training_tool_points_at_research_entry() -> None:
    source = (ROOT / "tools/production/train_ml_order_overlay.py").read_text(encoding="utf-8")
    assert "from research.experiments.ml_overlay_training import train_overlay_model" in source


def test_retired_production_training_name_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="research.experiments.ml_overlay_training"):
        production_train()


def test_production_package_discovery_excludes_research() -> None:
    config = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'include = ["leadlag*"]' in config
    assert "research = [" in config


def test_training_entry_records_completion(monkeypatch, tmp_path) -> None:
    model = MLOrderOverlayModel(
        lgbm=SimpleNamespace(),
        cont_cols=[],
        target_std=1.0,
        use_ticker=False,
        use_classification=False,
        per_ticker_interactions=False,
        metadata={"data_hash": "d", "config_hash": "c", "artifact_version": "v"},
    )
    events: list[dict] = []
    monkeypatch.setattr(training, "_train_overlay_model_impl", lambda **_: model)
    monkeypatch.setattr(training, "_record_training_event", lambda **event: events.append(event))

    result = train_overlay_model(
        pd.DataFrame(),
        tmp_path,
        _training_cfg(),
        "2020-01-01",
        "2020-01-02",
        tmp_path / "out",
    )
    assert result is model
    assert events[0]["status"] == "completed"
    assert events[0]["metrics"]["artifact_version"] == "v"


def test_training_entry_records_interruption(monkeypatch, tmp_path) -> None:
    events: list[dict] = []

    def fail(**_: object) -> MLOrderOverlayModel:
        raise RuntimeError("fit failed")

    monkeypatch.setattr(training, "_train_overlay_model_impl", fail)
    monkeypatch.setattr(training, "_record_training_event", lambda **event: events.append(event))
    with pytest.raises(RuntimeError, match="fit failed"):
        train_overlay_model(
            pd.DataFrame(),
            tmp_path,
            _training_cfg(),
            "2020-01-01",
            "2020-01-02",
            tmp_path / "out",
        )
    assert events[0]["status"] == "interrupted"
    assert events[0]["metrics"]["error_type"] == "RuntimeError"


def test_training_impl_rejects_pre_baseline_start(monkeypatch, tmp_path) -> None:
    from research.experiments import ml_overlay_training as implementation

    with pytest.raises(ValueError, match="2015-01-05"):
        implementation._train_overlay_model_impl(
            df_exec=pd.DataFrame(),
            gap_input_dir=tmp_path,
            run_cfg=_training_cfg(),
            train_start="2014-12-31",
            train_end="2015-01-05",
            output_dir=tmp_path / "out",
        )


def test_training_collector_uses_typed_pit_inputs(tmp_path) -> None:
    date = pd.Timestamp("2026-09-16")
    columns: dict[str, list[float]] = {
        "topix_night_return": [0.0],
    }
    columns.update({f"us_cc_{ticker}": [0.0] for ticker in US_TICKERS})
    for ticker in JP_TICKERS:
        columns[f"jp_gap_{ticker}"] = [0.0]
        columns[f"jp_beta_{ticker}"] = [1.0]
        columns[f"jp_open_trade_{ticker}"] = [100.0]
        columns[f"jp_close_sig_{ticker}"] = [100.0]
    frame = pd.DataFrame(columns, index=pd.DatetimeIndex([date]))
    open_910 = pd.DataFrame(0.0, index=frame.index, columns=JP_TICKERS)
    market_vol = pd.DataFrame(0.01, index=frame.index, columns=JP_TICKERS)
    seen: list[DecisionInputs] = []

    class FakeDecisionModel:
        def decide(self, *, inputs, overlay_enabled, use_file_cache):
            assert overlay_enabled is False
            assert use_file_cache is True
            assert isinstance(inputs, DecisionInputs)
            seen.append(inputs)
            return SimpleNamespace(
                fallback={},
                scores=np.linspace(-1.0, 1.0, len(JP_TICKERS)),
                mu_gap=np.zeros(len(JP_TICKERS)),
                sigma_gap=np.ones(len(JP_TICKERS)),
            )

    result = training._collect_training_data(
        pd.DatetimeIndex([date]),
        frame,
        np.zeros((1, len(JP_TICKERS))),
        tmp_path,
        ProductionV2RunConfig(),
        market_vol,
        open_910_returns=open_910,
        decision_model=FakeDecisionModel(),
    )
    assert len(result) == len(JP_TICKERS)
    assert len(seen) == 1
    assert seen[0].known.as_of == pd.Timestamp("2026-09-16 09:10")
    assert seen[0].historical.open_910_returns is not None


def test_training_normalizes_timezone_aware_adr_before_date_slice(monkeypatch, tmp_path) -> None:
    """The research entry must use the same JST ADR keys as inference."""
    date = pd.Timestamp("2026-08-10")
    frame = pd.DataFrame(
        {f"jp_oc_{ticker}": [0.0] for ticker in JP_TICKERS},
        index=pd.DatetimeIndex([date]),
    )
    aware_adr = pd.DataFrame(
        {f"adr_{ticker}": [0.01] for ticker in JP_TICKERS},
        index=pd.DatetimeIndex(["2026-08-09T15:00:00+00:00"]),
    )
    run_cfg = ProductionV2RunConfig(
        macro_kappa_enabled=False,
        macro_direction_enabled=False,
        cs_overlay_enabled=False,
    )
    collector_called = False

    def collect(*_args: object, **_kwargs: object) -> pd.DataFrame:
        nonlocal collector_called
        collector_called = True
        return pd.DataFrame()

    monkeypatch.setattr(training, "_precompute_market_vol", lambda _: pd.DataFrame())
    monkeypatch.setattr(
        training,
        "build_open_910_returns",
        lambda *_: pd.DataFrame(0.0, index=[date], columns=JP_TICKERS),
    )
    monkeypatch.setattr(
        training,
        "compute_jp_target_returns",
        lambda *_args, **_kwargs: np.zeros((1, len(JP_TICKERS))),
    )
    monkeypatch.setattr(training, "load_adr_features", lambda: aware_adr)
    monkeypatch.setattr(training, "_collect_training_data", collect)

    with pytest.raises(ValueError, match="No training samples"):
        training._train_overlay_model_impl(
            df_exec=frame,
            gap_input_dir=tmp_path,
            run_cfg=run_cfg,
            train_start=str(date.date()),
            train_end=str(date.date()),
            output_dir=tmp_path / "out",
        )
    assert collector_called


def test_training_normalizes_timezone_aware_execution_frame(monkeypatch, tmp_path) -> None:
    """The training entry uses the same JST date keys as its target inputs."""
    date = pd.Timestamp("2026-08-10")
    frame = pd.DataFrame(
        {f"jp_oc_{ticker}": [0.0] for ticker in JP_TICKERS},
        index=pd.DatetimeIndex(["2026-08-09T15:00:00+00:00"]),
    )
    run_cfg = ProductionV2RunConfig(
        macro_kappa_enabled=False,
        macro_direction_enabled=False,
        cs_overlay_enabled=False,
    )
    seen: list[pd.DataFrame] = []

    monkeypatch.setattr(training, "_precompute_market_vol", lambda _: pd.DataFrame())
    monkeypatch.setattr(
        training,
        "build_open_910_returns",
        lambda supplied, *_: pd.DataFrame(0.0, index=supplied.index, columns=JP_TICKERS),
    )
    monkeypatch.setattr(
        training,
        "compute_jp_target_returns",
        lambda *_args, **_kwargs: np.zeros((1, len(JP_TICKERS))),
    )

    def collect(*args: object, **kwargs: object) -> pd.DataFrame:
        seen.append(args[1])
        return pd.DataFrame()

    monkeypatch.setattr(training, "load_adr_features", lambda: None)
    monkeypatch.setattr(training, "_collect_training_data", collect)
    with pytest.raises(ValueError, match="No training samples"):
        training._train_overlay_model_impl(
            df_exec=frame,
            gap_input_dir=tmp_path,
            run_cfg=run_cfg,
            train_start=str(date.date()),
            train_end=str(date.date()),
            output_dir=tmp_path / "out",
        )
    assert len(seen) == 1
    assert seen[0].index.equals(pd.DatetimeIndex([date]))


def test_training_collector_excludes_nonfinite_realized_targets(tmp_path) -> None:
    date = pd.Timestamp("2026-09-16")
    columns: dict[str, list[float]] = {"topix_night_return": [0.0]}
    columns.update({f"us_cc_{ticker}": [0.0] for ticker in US_TICKERS})
    for ticker in JP_TICKERS:
        columns[f"jp_gap_{ticker}"] = [0.0]
        columns[f"jp_beta_{ticker}"] = [1.0]
        columns[f"jp_open_trade_{ticker}"] = [100.0]
        columns[f"jp_close_sig_{ticker}"] = [100.0]
    frame = pd.DataFrame(columns, index=pd.DatetimeIndex([date]))
    market_vol = pd.DataFrame(0.01, index=frame.index, columns=JP_TICKERS)
    open_910 = pd.DataFrame(0.0, index=frame.index, columns=JP_TICKERS)

    class FakeDecisionModel:
        def decide(self, *, inputs, overlay_enabled, use_file_cache):
            return SimpleNamespace(
                fallback={},
                scores=np.ones(len(JP_TICKERS)),
                mu_gap=np.zeros(len(JP_TICKERS)),
                sigma_gap=np.ones(len(JP_TICKERS)),
            )

    y_target = np.zeros((1, len(JP_TICKERS)))
    y_target[0, 0] = np.nan
    result = training._collect_training_data(
        pd.DatetimeIndex([date]),
        frame,
        y_target,
        tmp_path,
        ProductionV2RunConfig(),
        market_vol,
        open_910_returns=open_910,
        decision_model=FakeDecisionModel(),
    )
    assert len(result) == len(JP_TICKERS) - 1
    assert JP_TICKERS[0] not in set(result["ticker"])


def test_training_fit_rejects_nonfinite_targets() -> None:
    columns = {column: [0.0] for column in training.overlay_continuous_columns(False)}
    columns["ticker"] = [JP_TICKERS[0]]
    columns["target"] = [np.nan]
    with pytest.raises(ValueError, match="non-finite"):
        training._train_overlay_lgbm(pd.DataFrame(columns), use_ticker=False)
