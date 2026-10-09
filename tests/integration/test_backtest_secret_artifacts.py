"""Exercise the CLI/config/artifact boundary without network or live broker I/O."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3

import pandas as pd

from leadlag.cli import main
from leadlag.config import AppConfig
from leadlag.data.backtest_store import BacktestResultStore, _safe_config
from leadlag.execution import backtest


def test_cli_backtest_keeps_secrets_out_of_all_artifacts(tmp_path, monkeypatch, caplog, capsys):
    secrets = {
        name: f"SYNTHETIC_{name}_34"
        for name in (
            "KABU_API_PASSWORD",
            "KABU_API_TOKEN",
            "TACHIBANA_AUTH_ID",
            "TACHIBANA_SECOND_PASSWORD",
            "TACHIBANA_PRIVATE_KEY_PATH",
        )
    }
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("BROKER_PROVIDER", "tachibana")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("risk:\n  var_window: 3\nblpx:\n  lambda_reg: 0.65\n")
    dates = pd.bdate_range("2026-10-01", periods=3)
    results = {"daily_returns": pd.Series([0.001, -0.002, 0.003], index=dates)}
    monkeypatch.setattr(backtest, "_load_df_exec", lambda *args: pd.DataFrame(index=dates))
    configs = []

    def simulate(**kwargs):
        configs.append(kwargs["cfg"])
        return results

    monkeypatch.setattr(backtest.BacktestEngine, "run_v2_backtest", simulate)
    with caplog.at_level(logging.INFO):
        assert (
            main(
                [
                    "backtest",
                    "--config",
                    str(config_path),
                    "--output-root",
                    str(tmp_path / "out"),
                    "--run-tag",
                    "secrets",
                    "--start-date",
                    "2026-10-01",
                    "--skip-chart",
                ]
            )
            == 0
        )
    assert configs[0].kabu.api_password == secrets["KABU_API_PASSWORD"]
    assert configs[0].tachibana.second_password == secrets["TACHIBANA_SECOND_PASSWORD"]
    databases = list((tmp_path / "out").rglob("backtest_store.sqlite"))
    assert len(databases) == 1
    store = BacktestResultStore(databases[0])
    assert store.list_runs() == ["1"]
    pd.testing.assert_series_equal(store.load_results(1)["daily_returns"], results["daily_returns"])
    with sqlite3.connect(store.path) as connection:
        saved = json.loads(connection.execute("SELECT config_json FROM run_info").fetchone()[0])
        dump = "\n".join(connection.iterdump())
    config_hash = saved.pop("config_hash")
    canonical = json.dumps(saved, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert config_hash == hashlib.sha256(canonical.encode()).hexdigest()
    assert "kabu" not in saved and "tachibana" not in saved
    assert AppConfig(**saved).v2 == configs[0].v2
    assert saved["risk"]["var_window"] == 3
    changed = configs[0].model_copy(
        update={"risk": configs[0].risk.model_copy(update={"var_window": 4})}
    )
    assert _safe_config(changed)["config_hash"] != config_hash
    output = capsys.readouterr()
    observed = (dump + caplog.text + output.out + output.err).encode()
    for path in (tmp_path / "out").rglob("*"):
        if path.is_file():
            observed += path.read_bytes()
    for value in secrets.values():
        assert value.encode() not in observed


def test_mapping_artifact_allowlist_ignores_unknown_credentials():
    clean = _safe_config(AppConfig())
    mapping = AppConfig().model_dump(mode="json")
    mapping.update(new_broker={"password": "SYNTHETIC_UNKNOWN"}, token="SYNTHETIC_UNKNOWN")
    assert _safe_config(mapping) == clean
