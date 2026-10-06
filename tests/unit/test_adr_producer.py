"""ADR production quality and atomic bundle regression cases (offline)."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from leadlag.data import adr_features, adr_producer
from leadlag.data.adr_features import load_adr_features, publish_adr_features, validate_adr_features
from leadlag.data.adr_producer import build_adr_features, refresh_adr_features
from leadlag.data.tickers import ADR_TICKERS, JP_TICKERS


def _inputs():
    dates = pd.DatetimeIndex(["2026-10-01", "2026-10-02"])
    execution = pd.DataFrame(
        {"sig_date": dates}, index=pd.DatetimeIndex(["2026-10-02", "2026-10-05"])
    )
    close = pd.DataFrame(
        {ticker: [100.0, 110.0, 99.0] for ticker in ADR_TICKERS},
        index=pd.DatetimeIndex(["2026-09-30", "2026-10-01", "2026-10-02"]),
    )
    return execution, close


def test_alignment_observed_zero_and_partial_coverage():
    execution, close = _inputs()
    close.loc[pd.Timestamp("2026-10-02"), "HMC"] = np.nan
    close["TAK"] = 100.0
    frame = build_adr_features(execution, close)
    assert frame.iloc[0]["adr_1622.T"] == pytest.approx(0.1)
    assert frame.iloc[1]["adr_1622.T"] == pytest.approx(-0.1)
    assert frame.iloc[1]["coverage_1622.T"] == 1
    assert frame.iloc[1]["adr_1621.T"] == 0  # An observed zero has coverage.
    assert frame.iloc[1]["coverage_1621.T"] == 1
    assert frame.iloc[1]["adr_1617.T"] == 0  # Structural zero has no ADR mapping.
    assert frame.iloc[1]["coverage_1617.T"] == 0
    assert frame.iloc[1]["sig_date"] == pd.Timestamp("2026-10-02")


@pytest.mark.parametrize("missing", ["sector", "signal_day", "invalid_price"])
def test_missing_observations_reject_required_row_without_zero_imputation(missing):
    execution, close = _inputs()
    if missing == "sector":
        close = close.drop(columns="TAK")
    elif missing == "signal_day":
        close = close.drop(index=pd.Timestamp("2026-10-02"))
    else:
        close.loc[pd.Timestamp("2026-10-02"), "TAK"] = np.inf
    frame = build_adr_features(execution, close)
    assert pd.isna(frame.iloc[-1]["adr_1621.T"])
    assert validate_adr_features(frame, execution.index[-1]) is None


@pytest.mark.parametrize("invalid", ["same_day", "duplicate", "nat", "reverse"])
def test_reject_invalid_us_jp_mapping(invalid):
    execution, close = _inputs()
    if invalid == "same_day":
        execution["sig_date"] = execution.index
    elif invalid == "duplicate":
        execution.index = [execution.index[0]] * 2
    elif invalid == "nat":
        execution.iloc[-1, 0] = pd.NaT
    else:
        execution = execution.iloc[::-1]
    with pytest.raises(ValueError):
        build_adr_features(execution, close)


def test_future_prices_and_unknown_jp_target_do_not_affect_features():
    execution, close = _inputs()
    baseline = build_adr_features(execution, close)
    close.loc[pd.Timestamp("2026-10-05")] = 999999.0
    execution["jp_oc_1621.T"] = [-1234.0, 1234.0]
    pd.testing.assert_frame_equal(build_adr_features(execution, close), baseline)


def test_bundle_contains_consistent_artifacts_and_quality_manifest(tmp_path):
    execution, close = _inputs()
    close.loc[pd.Timestamp("2026-10-01"), "TAK"] = np.nan
    # Missing history is retained; the required row must still be complete.
    close.loc[pd.Timestamp("2026-09-30"), "TAK"] = np.nan
    close.loc[pd.Timestamp("2026-10-01"), "TAK"] = 110.0
    frame = build_adr_features(execution, close)
    path = tmp_path / "adr.zip"
    manifest = publish_adr_features(frame, path, required_trade_date=execution.index[-1])
    assert manifest["incomplete_rows"] == 1
    assert manifest["latest_coverage"]["1621.T"] == 1
    with zipfile.ZipFile(path) as bundle:
        assert json.loads(bundle.read("manifest.json")) == manifest
        pickled = pd.read_pickle(io.BytesIO(bundle.read("features.pkl")))
        csv = pd.read_csv(io.BytesIO(bundle.read("features.csv")), index_col=0, parse_dates=[0, 1])
        pd.testing.assert_frame_equal(pickled, csv, check_freq=False)
    loaded = load_adr_features(path, trade_date=execution.index[-1])
    assert loaded is not None
    assert loaded.index.name == "trade_date"
    assert loaded.columns.tolist() == [f"adr_{ticker}" for ticker in JP_TICKERS]
    assert load_adr_features(path, trade_date=execution.index[0]) is None
    assert load_adr_features(path, trade_date="2026-10-06") is None


def test_failed_commit_preserves_old_bundle_and_next_run_recovers(tmp_path, monkeypatch):
    execution, close = _inputs()
    path = tmp_path / "adr.zip"
    frame = build_adr_features(execution, close)
    publish_adr_features(frame, path, required_trade_date=execution.index[-1])
    previous = path.read_bytes()
    replace = adr_features.os.replace
    monkeypatch.setattr(
        adr_features.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("commit failed"))
    )
    frame["adr_1621.T"] = 0.25
    with pytest.raises(OSError, match="commit failed"):
        publish_adr_features(frame, path, required_trade_date=execution.index[-1])
    assert path.read_bytes() == previous
    assert list(tmp_path.glob("*.tmp")) == []
    monkeypatch.setattr(adr_features.os, "replace", replace)
    publish_adr_features(frame, path, required_trade_date=execution.index[-1])
    assert load_adr_features(path).iloc[-1]["adr_1621.T"] == 0.25


@pytest.mark.parametrize("damage", ["pickle", "csv", "date", "signal_date", "coverage", "truncated"])
def test_corrupt_bundle_is_rejected(tmp_path, damage):
    execution, close = _inputs()
    path = tmp_path / "adr.zip"
    publish_adr_features(
        build_adr_features(execution, close), path, required_trade_date=execution.index[-1]
    )
    if damage == "truncated":
        path.write_bytes(path.read_bytes()[:30])
    else:
        with zipfile.ZipFile(path) as bundle:
            members = {name: bundle.read(name) for name in bundle.namelist()}
        if damage in {"date", "signal_date", "coverage"}:
            manifest = json.loads(members["manifest.json"])
            if damage == "date":
                manifest["latest_trade_date"] = "2026-10-06"
            elif damage == "signal_date":
                manifest["latest_signal_date"] = "2026-10-06"
            else:
                manifest["latest_coverage"]["1621.T"] = 0
            members["manifest.json"] = json.dumps(manifest).encode()
        else:
            members[f"features.{('pkl' if damage == 'pickle' else 'csv')}"] = b"damaged"
        with zipfile.ZipFile(path, "w") as bundle:
            for name, value in members.items():
                bundle.writestr(name, value)
    assert load_adr_features(path) is None


@pytest.mark.parametrize("damage", ["missing_coverage", "negative_count", "excess_count", "false_observation", "structural_nonzero", "future_signal"])
def test_inconsistent_publication_metadata_is_rejected(tmp_path, damage):
    execution, close = _inputs()
    frame = build_adr_features(execution, close)
    if damage == "missing_coverage":
        frame = frame.drop(columns="coverage_1621.T")
    elif damage == "negative_count":
        frame["coverage_1621.T"] = -1
    elif damage == "excess_count":
        frame["coverage_1621.T"] = 2
    elif damage == "false_observation":
        frame["coverage_1621.T"] = 0
    elif damage == "structural_nonzero":
        frame["adr_1617.T"] = 0.1
    else:
        frame["sig_date"] = execution.index
    with pytest.raises(ValueError):
        publish_adr_features(frame, tmp_path / "adr.zip", required_trade_date=execution.index[-1])
    assert not (tmp_path / "adr.zip").exists()


def test_producer_download_failure_and_stale_source_do_not_publish(tmp_path, monkeypatch):
    execution, close = _inputs()
    path = tmp_path / "adr.zip"
    publish_adr_features(
        build_adr_features(execution, close), path, required_trade_date=execution.index[-1]
    )
    previous = path.read_bytes()
    monkeypatch.setattr(
        adr_producer.yf,
        "download",
        lambda **kwargs: (_ for _ in ()).throw(TimeoutError("download")),
    )
    with pytest.raises(TimeoutError):
        refresh_adr_features(execution, required_trade_date=execution.index[-1], path=path)
    assert path.read_bytes() == previous
    raw = pd.concat({ticker: pd.DataFrame({"Close": close[ticker]}) for ticker in close}, axis=1)
    monkeypatch.setattr(adr_producer.yf, "download", lambda **kwargs: raw)
    with pytest.raises(ValueError):
        refresh_adr_features(execution, required_trade_date="2026-10-06", path=path)
    assert path.read_bytes() == previous
    manifest = refresh_adr_features(execution, required_trade_date=execution.index[-1], path=path)
    assert manifest["latest_signal_date"] == "2026-10-02"
    assert load_adr_features(path) is not None


def test_scheduled_updater_publishes_through_operational_service(tmp_path, monkeypatch):
    import functools
    import importlib.util

    entry = Path(__file__).resolve().parents[2] / "scripts/batch/_update_market_data.py"
    spec = importlib.util.spec_from_file_location("market_updater_under_test", entry)
    updater = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(updater)
    today = pd.Timestamp.now(tz="Asia/Tokyo").tz_localize(None).normalize()
    dates = pd.DatetimeIndex([today - pd.Timedelta(days=3), today - pd.Timedelta(days=1)])
    execution = pd.DataFrame(
        {"sig_date": dates}, index=pd.DatetimeIndex([today - pd.Timedelta(days=2), today])
    )
    close = pd.DataFrame(
        {ticker: [100.0, 110.0, 99.0] for ticker in ADR_TICKERS},
        index=pd.DatetimeIndex([dates[0] - pd.Timedelta(days=1), dates[0], dates[1]]),
    )
    raw = pd.concat({ticker: pd.DataFrame({"Close": close[ticker]}) for ticker in close}, axis=1)
    monkeypatch.setattr(updater, "download_data", lambda **kwargs: {})
    monkeypatch.setattr(updater, "load_df_exec_from_local_cache", lambda **kwargs: execution)
    monkeypatch.setattr(updater, "is_trading_day", lambda day: True)
    monkeypatch.setattr(
        updater,
        "refresh_adr_features",
        functools.partial(refresh_adr_features, path=tmp_path / "adr.zip"),
    )
    monkeypatch.setattr(adr_producer.yf, "download", lambda **kwargs: raw)
    assert updater.main() == 0
    assert load_adr_features(tmp_path / "adr.zip", trade_date=today) is not None
    previous = (tmp_path / "adr.zip").read_bytes()
    monkeypatch.setattr(
        updater, "load_df_exec_from_local_cache", lambda **kwargs: execution.iloc[:-1]
    )
    assert updater.main() == 1
    assert (tmp_path / "adr.zip").read_bytes() == previous
