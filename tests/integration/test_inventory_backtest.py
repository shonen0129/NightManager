from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from leadlag.config.schemas import AppConfig
from leadlag.data.backtest_store import BacktestResultStore
from leadlag.data.tickers import JP_TICKERS
from leadlag.domain.inputs import HistoricalInputs
from leadlag.execution import var_history
from leadlag.execution.backtest import _save_detailed_backtest_results
from leadlag.execution.backtester import BacktestEngine


@pytest.mark.parametrize(
    "gap,morning,expected", [(0.0, 0.1, 0.075), (0.1, -0.1, -0.0075), (-0.1, 0.1, -0.0075)]
)
def test_carry_from_v2_entry_through_prices_pnl_and_saved_artifacts(
    monkeypatch, tmp_path, gap, morning, expected
):
    dates = pd.DatetimeIndex(["2026-10-09", "2026-10-13"])
    frame = pd.DataFrame(index=dates)
    for ticker in JP_TICKERS:
        frame[f"jp_open_trade_{ticker}"] = 100.0
        frame[f"jp_oc_{ticker}"] = 0.0
        frame[f"jp_gap_{ticker}"] = 0.0
    frame.loc[dates[1], f"jp_oc_{JP_TICKERS[0]}"] = morning
    frame.loc[dates[1], f"jp_gap_{JP_TICKERS[0]}"] = gap
    measured = pd.DataFrame(0.0, index=dates, columns=JP_TICKERS)
    measured.loc[dates[1], JP_TICKERS[0]] = morning
    weights = np.zeros((2, 17))
    weights[0, :2] = [1.0, -1.0]
    monkeypatch.setattr(
        BacktestEngine,
        "_generate_v2_weights",
        lambda *_: (weights.copy(), np.zeros(2, dtype=bool), [{}, {}]),
    )
    result = BacktestEngine.run_v2_backtest(
        AppConfig(),
        None,
        frame,
        start_date=str(dates[0].date()),
        historical_inputs=HistoricalInputs(frame, source="synthetic", open_910_returns=measured),
        slippage_bps=0.0,
        buy_interest_annual=0.0,
        borrow_fee_annual=0.0,
        reverse_fee_bps=0.0,
        overnight_alpha_long=0.75,
        overnight_alpha_short=0.5,
        side_leverage=1.0,
    )
    np.testing.assert_allclose(result["daily_returns"], [0.0, expected], atol=1e-14)
    np.testing.assert_allclose(result["daily_carry_gap_returns"], [0.0, 0.75 * gap])
    np.testing.assert_allclose(
        result["daily_carry_open_910_returns"], [0.0, 0.75 * (1 + gap) * morning]
    )
    assert result["terminal_inventory"]["cash"] == pytest.approx(1 + expected)
    np.testing.assert_allclose(
        result["daily_inventory_equity"],
        result["daily_cash"] + result["daily_holdings"].sum(axis=1),
    )
    _save_detailed_backtest_results(result, tmp_path / "csv")
    contract = json.loads((tmp_path / "csv/accounting_contract.json").read_text())
    assert contract["version"] == "inventory-v3"
    assert contract["carry_attribution"] == "receiving_trade_date"
    terminal = json.loads((tmp_path / "csv/terminal_inventory.json").read_text())
    assert terminal == result["terminal_inventory"]
    saved = pd.read_csv(tmp_path / "csv/daily_daily_carry_open_910_returns.csv", index_col=0)
    np.testing.assert_allclose(saved.iloc[:, 0], result["daily_carry_open_910_returns"])
    store = BacktestResultStore(tmp_path / "result.sqlite")
    run_id = store.save_run(result)
    loaded = store.load_results(run_id)
    assert loaded["terminal_inventory"] == terminal
    pd.testing.assert_frame_equal(loaded["daily_holdings"], result["daily_holdings"])


def test_v2_artifact_changes_transfer_inventory_without_reset(monkeypatch):
    dates = pd.DatetimeIndex(["2026-10-02", "2026-10-05"])
    frame = pd.DataFrame(index=dates)
    for ticker in JP_TICKERS:
        frame[f"jp_open_trade_{ticker}"] = 100.0
        frame[f"jp_oc_{ticker}"] = 0.0
        frame[f"jp_gap_{ticker}"] = 0.0
    frame.loc[dates[1], f"jp_gap_{JP_TICKERS[0]}"] = 0.1
    measured = pd.DataFrame(0.0, index=dates, columns=JP_TICKERS)
    history = HistoricalInputs(frame, source="synthetic", open_910_returns=measured)
    weights = np.zeros((2, 17))
    weights[0, :2] = [1.0, -1.0]

    def generate(_frame, _cfg, _gap, requested_dates, *args):
        return (
            weights[dates.get_indexer(requested_dates)],
            np.zeros(len(requested_dates), dtype=bool),
            [{} for _ in requested_dates],
        )

    monkeypatch.setattr(BacktestEngine, "_generate_v2_weights", generate)
    common = dict(
        cfg=AppConfig(),
        gap_input_dir=None,
        df_exec=frame,
        historical_inputs=history,
        slippage_bps=5.0,
        side_leverage=1.3,
    )
    full = BacktestEngine.run_v2_backtest(start_date="2026-10-02", **common)
    first = BacktestEngine.run_v2_backtest(
        start_date="2026-10-02", end_date="2026-10-02", terminal_policy="open_inventory", **common
    )
    terminal = first["terminal_inventory"]
    second = BacktestEngine.run_v2_backtest(
        start_date="2026-10-05",
        initial_holdings=np.array(terminal["holdings"]),
        initial_cash=terminal["cash"],
        initial_mark_date=terminal["mark_date"],
        initial_target_weights=np.array(terminal["target_weights"]),
        **common,
    )
    for key in ("daily_returns", "daily_costs", "daily_turnover", "daily_cash", "daily_holdings"):
        assertion = (
            pd.testing.assert_frame_equal
            if key == "daily_holdings"
            else pd.testing.assert_series_equal
        )
        assertion(full[key], pd.concat([first[key], second[key]]))


def test_live_var_history_keeps_inventory_across_artifact_versions(monkeypatch, tmp_path):
    dates = pd.DatetimeIndex(["2026-10-02", "2026-10-05"])
    frame = pd.DataFrame(index=dates)
    for ticker in JP_TICKERS:
        frame[f"jp_open_trade_{ticker}"] = 100.0
        frame[f"jp_oc_{ticker}"] = 0.0
        frame[f"jp_gap_{ticker}"] = 0.0
    frame.loc[dates[1], f"jp_gap_{JP_TICKERS[0]}"] = 0.1
    history = HistoricalInputs(
        frame,
        source="synthetic",
        open_910_returns=pd.DataFrame(0.0, index=dates, columns=JP_TICKERS),
    )
    cfg = AppConfig()
    cfg = cfg.model_copy(update={"v2": cfg.v2.model_copy(update={"ml_overlay_enabled": False})})
    monkeypatch.setattr(var_history, "load_df_exec_from_local_cache", lambda **_: frame.copy())
    monkeypatch.setattr(var_history, "load_config_from_yaml", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(var_history, "_build_var_historical_inputs", lambda *_: history)
    monkeypatch.setattr(
        var_history,
        "_resolve_overlay_history_chunks",
        lambda *_: ([(dates[:1], None), (dates[1:], None)], {"versions": ["old", "new"]}),
    )
    weights = np.zeros((2, 17))
    weights[0, :2] = [1.0, -1.0]

    def generate(_frame, _cfg, _gap, selected, *args):
        return (
            weights[dates.get_indexer(selected)],
            np.zeros(len(selected), dtype=bool),
            [{"audit_status": {"numerical": "PASSED", "leakage": "PASSED"}} for _ in selected],
        )

    monkeypatch.setattr(BacktestEngine, "_generate_v2_weights", generate)
    expected = BacktestEngine.run_v2_backtest(
        cfg,
        None,
        frame,
        start_date="2026-10-02",
        historical_inputs=history,
        terminal_policy="open_inventory",
    )
    returns = var_history.get_hist_returns_for_risk(
        SimpleNamespace(start_date="2026-10-02", slippage_bps=None, var_history_timeout=30),
        str(tmp_path),
        pd.Timestamp("2026-10-06"),
    )
    np.testing.assert_allclose(returns.to_numpy(), expected["daily_returns"].to_numpy(), atol=1e-14)
    assert returns.iloc[1] > 0  # incoming carry remains on the new-artifact flat day
