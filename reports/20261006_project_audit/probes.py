"""Offline synthetic audit probes, with no orders, network, or cache writes."""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.broker.tachibana.api import TachibanaClient
from leadlag.config.schemas import AppConfig, KabuApiConfig, ProductionV2RunConfig, StrategyConfig, TachibanaApiConfig
from leadlag.core.market_calendar import is_trading_day
from leadlag.core.pnl import simulate_daily_pnl
from leadlag.data.adr_features import load_adr_features, validate_adr_features
from leadlag.data.backtest_store import _safe_config
from leadlag.data.providers.yfinance_provider import YFinanceProvider
from leadlag.data.preprocessor import preprocess_data
from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER, US_TICKERS
from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.output_ops import save_summary_files
from leadlag.experiment_registry import compute_deflated_sharpe
from leadlag.models.v2.decision_engine import generate_v2_production_portfolio_from_distribution
from leadlag.reporting.metrics import MetricsSpec, calculate_metrics


def pnl(weights, target, gap, dates, alpha=1.0, slip=0.0):
    return simulate_daily_pnl(weights=weights, target_returns=target,
        gap_returns=gap, sim_dates=dates, slip=slip, financing_daily=0.0,
        borrow_daily=0.0, reverse_daily=0.0, alpha_long=alpha, alpha_short=alpha)


def main():
    out = {}
    dates = pd.DatetimeIndex(["2026-10-01", "2026-10-02"])
    frame = pd.DataFrame(index=dates)
    for ticker in JP_TICKERS:
        frame[f"jp_open_trade_{ticker}"] = [100.0, 100.0]
        frame[f"jp_oc_{ticker}"] = [0.0, 0.1 if ticker == JP_TICKERS[0] else 0.0]
        frame[f"jp_gap_{ticker}"] = [0.0, 0.0]
    open_910 = pd.DataFrame(0.0, index=dates, columns=JP_TICKERS)
    open_910.iloc[1, 0] = 0.1
    target, gap = BacktestEngine._compute_target_and_gap_returns(frame, dates, dates, open_910_returns=open_910)
    weights = np.zeros_like(target)
    weights[0, 0] = 1.0
    weights[0, 1] = -1.0
    out["carry_open_to_910"] = {"path": "BacktestEngine target/gap extraction -> simulate_daily_pnl",
        "close_day1": 100, "open_day2": 100, "price_910_day2": 110, "close_day2": 110,
        "held_long_from_day1_to_day2_910": 1.0, "flat_price_short_weight": -1.0,
        "model_net": 0.0, "model_gross": 2.0,
        "computed_gross_returns": pnl(weights, target, gap, dates)["gross_returns"],
        "held_inventory_return_until_910": 0.1}
    stable = np.ones((2, 1))
    result = pnl(stable, np.zeros((2, 1)), np.zeros((2, 1)), dates, alpha=0.0, slip=0.001)
    out["turnover"] = {"reported_turnover": result["turnover"], "executed_one_way_volume_by_day": [2.0, 2.0], "slippage_costs": result["slip_costs"]}
    terminal = pnl(np.ones((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), dates[:1], alpha=1.0, slip=0.001)
    out["terminal_inventory"] = {"slippage_costs": terminal["slip_costs"], "cost_if_liquidated": 0.002, "unreported_final_inventory": 1.0}
    missing = pnl(np.array([[1.0], [0.0]]), np.array([[0.01], [np.nan]]), np.zeros((2, 1)), dates)
    missing_series = pd.Series(missing["net_returns"], index=dates)
    out["missing_flat_target"] = {"evaluation_days": 2, "pnl_non_finite_days": int((~np.isfinite(missing_series)).sum()), "metrics_after_dropna": calculate_metrics(missing_series), "finite_days_used": int(missing_series.notna().sum())}
    _, start, end = BacktestEngine._resolve_sim_dates(frame, "2026-09-01", "2026-09-30", 0)
    out["end_before_data"] = {"requested_end": "2026-09-30", "first_available": "2026-10-01", "selected": frame.index[start:end + 1].strftime("%Y-%m-%d").tolist()}
    prior_frame = pd.DataFrame(index=pd.DatetimeIndex(["2010-01-04", "2014-12-30", "2015-01-05"]))
    _, start, end = BacktestEngine._resolve_sim_dates(prior_frame, "2010-01-04", "2014-12-30", 0)
    out["baseline_interval"] = {"accepted_simulation_dates": prior_frame.index[start:end + 1].strftime("%Y-%m-%d").tolist(), "required_minimum_start": "2015-01-05"}
    cfg = ProductionV2RunConfig(long_count=5, short_count=3, minvar_enabled=True, minvar_alpha=0.8)
    result = generate_v2_production_portfolio_from_distribution(np.arange(-8.0, 9.0) * 0.001, np.eye(17) * 0.0001, "2026-10-06", cfg, None, None,
        distribution_metadata={"sig_date": "2026-10-05"}, allow_implicit_io=False)
    out["minvar_counts"] = {"requested_long": 5, "requested_short": 3, "actual_long": int(np.sum(result.w_final > 0)), "actual_short": int(np.sum(result.w_final < 0)), "summary": result.summary, "numerical": result.numerical}
    returns = [-0.01, 0.015, -0.005, 0.02]
    dsr_metrics = {"net_sharpe": 2.0, "trials": 10, "n_observations": 4, "returns": returns}
    out["dsr_inconsistent_count"] = {"finite_returns": 4, "dsr_T4": compute_deflated_sharpe(dsr_metrics), "dsr_T1000": compute_deflated_sharpe({**dsr_metrics, "n_observations": 1000}), "invalid_status_dsr": compute_deflated_sharpe({**dsr_metrics, "n_observations": 1000, "metric_status": "invalid"})}
    monthly_input = pd.Series([0.01, -0.005, 0.02, 0.005], index=pd.to_datetime(["2026-01-15", "2026-02-15", "2026-03-15", "2026-04-15"]))
    out["monthly_annualization"] = {"frequency_only": calculate_metrics(monthly_input, frequency="monthly"), "explicit_spec": calculate_metrics(monthly_input, spec=MetricsSpec(frequency="monthly", annualization_periods=12))}
    with tempfile.TemporaryDirectory(prefix="summary-probe-", dir=OUT) as directory:
        losses = pd.DataFrame({"daily_return": [-0.1, 0.0]}, index=dates)
        metrics = calculate_metrics(losses["daily_return"])
        save_summary_files(losses, metrics, StrategyConfig(), directory)
        summary = json.loads((Path(directory) / "run_summary.json").read_text())
        out["summary_mdd"] = {"metrics_MDD": metrics["MDD"], "run_summary_max_drawdown": summary["max_drawdown"]}
    fake_config = AppConfig(kabu=KabuApiConfig(api_password="AUDIT_DUMMY_PASSWORD", api_token="AUDIT_DUMMY_TOKEN"), tachibana=TachibanaApiConfig(auth_id="AUDIT_DUMMY_AUTH", second_password="AUDIT_DUMMY_SECOND"))
    serialized = json.dumps(_safe_config(fake_config))
    out["backtest_config_secrets"] = {"synthetic_password_preserved": "AUDIT_DUMMY_PASSWORD" in serialized, "synthetic_token_preserved": "AUDIT_DUMMY_TOKEN" in serialized, "synthetic_second_password_preserved": "AUDIT_DUMMY_SECOND" in serialized}
    api = TachibanaClient(TachibanaApiConfig(api_url="https://audit.invalid", auth_id="AUDIT_DUMMY_AUTH"))
    def fake_get(url, **kwargs):
        response = requests.Response()
        response.status_code = 404
        response.url = url
        return response
    with patch.object(api.session, "get", side_effect=fake_get):
        try:
            api.login()
        except requests.HTTPError as exc:
            out["auth_http_exception"] = {"encoded_auth_id_present_in_exception": "AUDIT_DUMMY_AUTH" in str(exc), "query_present_in_exception": "?" in str(exc)}
    api.session.close()
    daily = pd.DataFrame({"Open": [100.0], "High": [110.0], "Low": [90.0], "Close": [105.0]}, index=pd.to_datetime(["2026-10-01"]))
    provider = YFinanceProvider(download_fn=lambda tickers, start, end: daily)
    out["optional_volume"] = {"input_ohlc_rows": 1, "output_rows": len(provider.fetch_daily_ohlc(["TEST"], date(2026, 10, 1), date(2026, 10, 2))["TEST"])}
    history = pd.DataFrame({"Close": [100.0, 200.0]}, index=pd.to_datetime(["2026-10-01 09:10", "2026-10-01 15:00"]))
    with patch("leadlag.data.providers.yfinance_provider.yf.Ticker") as ticker:
        ticker.return_value.history.return_value = history
        out["intraday_at"] = {"requested": "2026-10-01 09:10", "returned": provider.fetch_intraday_quote(["TEST"], pd.Timestamp("2026-10-01 09:10"))["TEST"], "price_at_requested_time": 100.0}
    adr = load_adr_features()
    out["adr_local"] = {"exists": adr is not None, "latest": str(adr.index.max()) if adr is not None else None,
        "valid_for_20261006": validate_adr_features(adr, "2026-10-06") is not None}
    out["calendar_outside_static"] = {"2024-12-31_is_trading": is_trading_day(date(2024, 12, 31)), "2028-01-03_is_trading": is_trading_day(date(2028, 1, 3))}
    raw_dates = pd.bdate_range("2026-09-01", periods=5)
    raw = {"us_close": pd.DataFrame({tk: np.arange(5) + 100.0 for tk in [*US_TICKERS, "SPY"]}, index=raw_dates),
        "jp_close": pd.DataFrame({tk: np.arange(5) + 101.0 for tk in [*JP_TICKERS, TOPIX_TICKER]}, index=raw_dates),
        "jp_open": pd.DataFrame({tk: np.arange(5) + 100.0 for tk in [*JP_TICKERS, TOPIX_TICKER]}, index=raw_dates)}
    raw["us_close"].loc[raw_dates[1], "XLC"] = np.nan
    processed = preprocess_data(raw, beta_window=2, strict_validation=True)
    out["post_inception_proxy"] = {"missing_input_ticker": "XLC", "missing_signal_date": str(raw_dates[1].date()), "output_trade_date": str(raw_dates[2].date()), "output_XLC_return": float(processed.loc[raw_dates[2], "us_cc_XLC"]), "strict_validation_accepted": True, "proxy_source_column_present": any("proxy" in str(c).lower() for c in processed.columns)}
    path_namespace = {"__file__": "/tmp/audit-venv/lib/python3.12/site-packages/leadlag/config/paths.py"}
    exec(compile((ROOT / "src/leadlag/config/paths.py").read_text(), path_namespace["__file__"], "exec"), path_namespace)
    out["installed_package_root"] = {"simulation": "Execute the unchanged paths module at a conventional installed file location; no wheel installation", "package_file": path_namespace["__file__"], "resolved_runtime_root": str(path_namespace["project_root"]()), "checkout_root": str(ROOT)}
    def json_safe(value):
        if isinstance(value, dict):
            return {k: json_safe(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [json_safe(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value
    encoded = json.dumps(json_safe(out), ensure_ascii=False, indent=2, default=str, allow_nan=False)
    (OUT / "probes.json").write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
