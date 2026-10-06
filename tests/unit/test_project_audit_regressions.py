"""Offline regressions for the concrete boundaries found by the October audit."""
from __future__ import annotations
import json
from datetime import date
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from leadlag.config.schemas import AppConfig, ProductionV2RunConfig, StrategyConfig
from leadlag.core.market_calendar import is_trading_day, next_trading_day, previous_trading_day
from leadlag.core.pnl import simulate_daily_pnl
from leadlag.core.signal import build_weights_minvar
from leadlag.data.providers import YFinanceProvider
from leadlag.execution.backtester import BacktestEngine
from leadlag.execution.output_ops import save_summary_files
from leadlag.experiment_registry import compute_deflated_sharpe
from leadlag.models.v2.decision_engine import generate_v2_production_portfolio_from_distribution
from leadlag.reporting.metrics import MetricsSpec, calculate_metrics
from research.experiment_utils import record_backtest_experiment


@pytest.mark.parametrize('start,end', [('2026-09-01', '2026-09-30'), ('2026-11-01', 'latest'), ('2026-10-02', '2026-10-01'), ('NaT', 'latest'), ('2026-10-01', 'NaT')])
def test_requested_period_must_intersect_completed_data(start, end):
    frame = pd.DataFrame(index=pd.DatetimeIndex(['2026-10-01', '2026-10-02']))
    with pytest.raises(ValueError):
        BacktestEngine._resolve_sim_dates(frame, start, end, 0)


@pytest.mark.parametrize('index', [pd.DatetimeIndex([]), pd.DatetimeIndex(['NaT']), pd.DatetimeIndex(['2026-10-02', '2026-10-01']), pd.DatetimeIndex(['2026-10-01', '2026-10-01'])])
def test_invalid_source_dates_rejected(index):
    with pytest.raises(ValueError):
        BacktestEngine._resolve_sim_dates(pd.DataFrame(index=index), '2026-10-01', 'latest', 0)


def test_v2_entry_rejects_prior_evaluation_before_model_work():
    frame = pd.DataFrame(index=pd.DatetimeIndex(['2014-12-30', '2015-01-05']))
    with pytest.raises(ValueError, match='prior period'):
        BacktestEngine.run_v2_backtest(AppConfig(), None, frame, start_date='2014-12-30')


def test_monthly_short_form_equals_explicit_spec():
    returns = pd.Series([.01, -.005, .02, .005], index=pd.to_datetime(['2026-01-15', '2026-02-15', '2026-03-15', '2026-04-15']))
    expected = calculate_metrics(returns, spec=MetricsSpec(frequency='monthly', annualization_periods=12))
    assert calculate_metrics(returns, frequency='monthly') == expected
    assert expected['AR'] == pytest.approx((1.01 * .995 * 1.02 * 1.005) ** 3 - 1)


@pytest.mark.parametrize('missing', [np.nan, np.inf, -np.inf])
def test_metrics_never_drop_missing_evaluation_days(missing):
    with pytest.raises(ValueError, match='every evaluation day'):
        calculate_metrics(pd.Series([.01, missing]))


def test_flat_missing_label_is_zero_but_active_missing_label_is_invalid():
    result = simulate_daily_pnl(weights=np.array([[0.], [1.]]), target_returns=np.full((2, 1), np.nan), gap_returns=np.full((2, 1), np.nan), sim_dates=pd.DatetimeIndex(['2026-10-01', '2026-10-02']), slip=0., financing_daily=0., borrow_daily=0., reverse_daily=0., alpha_long=0., alpha_short=0.)
    assert result['net_returns'][0] == 0.
    assert np.isnan(result['net_returns'][1])


def test_summary_dd_matches_metrics_for_initial_loss(tmp_path):
    frame = pd.DataFrame({'daily_return': [-.1, 0.]}, index=pd.DatetimeIndex(['2026-10-01', '2026-10-02']))
    metrics = calculate_metrics(frame['daily_return'])
    save_summary_files(frame, metrics, StrategyConfig(), tmp_path)
    summary = json.loads((tmp_path / 'run_summary.json').read_text())
    assert summary['max_drawdown'] == pytest.approx(metrics['MDD'])
    assert summary['max_drawdown'] == pytest.approx(-.1)


def test_asymmetric_minvar_uses_selected_baskets_once():
    cfg = ProductionV2RunConfig(long_count=5, short_count=3, minvar_enabled=True, minvar_alpha=.8)
    result = generate_v2_production_portfolio_from_distribution(np.arange(-8., 9.) * .001, np.eye(17) * .0001, '2026-10-06', cfg, None, None, distribution_metadata={'sig_date': '2026-10-05'}, allow_implicit_io=False)
    assert (result.w_final > 0).sum() == 5
    assert (result.w_final < 0).sum() == 3
    assert result.w_final.sum() == pytest.approx(0., abs=1e-12)
    assert result.numerical['status'] == 'PASSED'


def test_minvar_rejects_overlapping_indices_and_impossible_config():
    with pytest.raises(ValueError, match='disjoint'):
        build_weights_minvar(np.arange(6.), np.array([4, 5]), np.array([0, 5]))
    with pytest.raises(ValueError, match='universe'):
        ProductionV2RunConfig(long_count=10, short_count=8)


@pytest.mark.parametrize('override', [{'n_observations': 1000}, {'returns': [.01, np.nan, -.01, .02]}, {'metric_status': 'invalid'}, {'net_sharpe': np.nan}, {'net_sharpe_frequency': 'monthly'}, {'trading_days_per_year': 0}, {'trials': 1.5}, {'trial_sharpe_variance': np.nan}, {'trial_sharpes': [np.inf] * 10}, {'trial_sharpes': [1., 2.]}])
def test_dsr_rejects_invalid_or_inconsistent_inputs(override):
    metrics = {'net_sharpe': 2., 'trials': 10, 'n_observations': 4, 'returns': [-.01, .015, -.005, .02]}
    assert compute_deflated_sharpe(metrics) is not None
    assert compute_deflated_sharpe({**metrics, **override}) is None


@pytest.mark.parametrize('extra', [{'n_observations': 1000}, {'metric_status': 'valid'}, {'net_sharpe': 100.}])
def test_extra_metrics_cannot_replace_computed_or_invalid_values(tmp_path, extra):
    with pytest.raises(ValueError, match='cannot override'):
        record_backtest_experiment('audit', 'contract', None, results={'daily_returns': pd.Series([.01, np.nan])}, extra_metrics=extra, registry_path=tmp_path / 'registry.jsonl')
    assert not (tmp_path / 'registry.jsonl').exists()


def test_provider_keeps_ohlc_without_optional_volume_and_avoids_future_bar():
    daily = pd.DataFrame({'Open': [100.], 'High': [110.], 'Low': [90.], 'Close': [105.]}, index=pd.DatetimeIndex(['2026-10-01']))
    provider = YFinanceProvider(download_fn=lambda *args: daily)
    assert len(provider.fetch_daily_ohlc(['TEST'], date(2026, 10, 1), date(2026, 10, 2))['TEST']) == 1
    history = pd.DataFrame({'Close': [100., 110., 200.]}, index=pd.DatetimeIndex(['2026-10-01 09:09', '2026-10-01 09:10', '2026-10-01 15:00'], tz='Asia/Tokyo'))
    with patch('leadlag.data.providers.yfinance_provider.yf.Ticker') as ticker:
        ticker.return_value.history.return_value = history
        assert provider.fetch_intraday_quote(['TEST'], pd.Timestamp('2026-10-01 09:10')) == {'TEST': 100.}
        assert provider.fetch_intraday_quote(['TEST'], pd.Timestamp('2026-09-30 20:10', tz='America/New_York')) == {'TEST': 100.}
        assert provider.fetch_intraday_quote(['TEST'], pd.Timestamp('2026-09-30 09:10')) == {}


@pytest.mark.parametrize('day', [date(2024, 12, 31), date(2028, 1, 3), date(2029, 1, 2)])
def test_exchange_closures_apply_outside_static_years(day):
    assert not is_trading_day(day)


def test_year_boundary_sessions_skip_exchange_closures():
    assert previous_trading_day(date(2028, 1, 4)) == date(2027, 12, 30)
    assert next_trading_day(date(2027, 12, 30)) == date(2028, 1, 4)


@pytest.mark.parametrize('side,gap,morning', [(1., 0., .1), (-1., 0., -.1), (1., .1, -.1)])
def test_carry_reaches_next_entry_including_gap_and_morning(side, gap, morning):
    from leadlag.data.tickers import JP_TICKERS

    dates = pd.DatetimeIndex(['2026-10-02', '2026-10-05'])
    frame = pd.DataFrame(index=dates)
    for ticker in JP_TICKERS:
        frame[f'jp_oc_{ticker}'] = [0., morning]
        frame[f'jp_gap_{ticker}'] = [0., gap]
        frame[f'jp_open_trade_{ticker}'] = 100.
    measured = pd.DataFrame([np.zeros(17), np.full(17, morning)], index=dates, columns=JP_TICKERS)
    target, carry = BacktestEngine._compute_target_and_gap_returns(frame, dates, dates, measured)
    weights = np.zeros_like(target)
    weights[0, 0] = side
    pnl = simulate_daily_pnl(weights=weights, target_returns=target, gap_returns=carry, sim_dates=dates, slip=0., financing_daily=0., borrow_daily=0., reverse_daily=0., alpha_long=1., alpha_short=1.)
    assert sum(pnl['gross_returns']) == pytest.approx(side * ((1 + gap) * (1 + morning) - 1))
    assert np.max(np.abs(target)) == pytest.approx(0.)


def test_turnover_is_effective_inventory_flow_and_cost_matches_volume():
    dates = pd.DatetimeIndex(['2026-10-01', '2026-10-02'])
    pnl = simulate_daily_pnl(weights=np.ones((2, 1)), target_returns=np.zeros((2, 1)), gap_returns=np.zeros((2, 1)), sim_dates=dates, slip=.001, financing_daily=0., borrow_daily=0., reverse_daily=0., alpha_long=0., alpha_short=0., side_leverage=1.3)
    assert pnl['execution_volume'] == [2.6, 2.6]
    assert pnl['turnover'] == [1.3, 1.3]
    assert pnl['target_weight_turnover'] == [.5, 0.]
    np.testing.assert_allclose(pnl['slip_costs'], np.array(pnl['execution_volume']) * .001)


def test_listed_us_missing_prices_are_rejected_not_proxied():
    from leadlag.data.preprocessor import preprocess_data
    from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER, US_TICKERS
    from leadlag.data.validation import DataValidationError

    dates = pd.bdate_range('2026-09-01', periods=5)
    raw = {'us_close': pd.DataFrame({tk: np.arange(5) + 100. for tk in [*US_TICKERS, 'SPY']}, index=dates), 'jp_close': pd.DataFrame({tk: np.arange(5) + 101. for tk in [*JP_TICKERS, TOPIX_TICKER]}, index=dates), 'jp_open': pd.DataFrame({tk: np.arange(5) + 100. for tk in [*JP_TICKERS, TOPIX_TICKER]}, index=dates)}
    raw['us_close'].loc[dates[1], 'XLC'] = np.nan
    with pytest.raises(DataValidationError, match='Post-inception'):
        preprocess_data(raw, strict_validation=True)
    frame = preprocess_data(raw)
    assert dates[2] not in frame.index
    assert dates[3] not in frame.index


def test_pre_inception_proxy_keeps_per_cell_provenance():
    from leadlag.data.preprocessor import preprocess_data
    from leadlag.data.tickers import JP_TICKERS, TOPIX_TICKER, US_TICKERS

    dates = pd.bdate_range('2010-01-04', periods=5)
    raw = {'us_close': pd.DataFrame({tk: np.arange(5) + 100. for tk in [*US_TICKERS, 'SPY']}, index=dates), 'jp_close': pd.DataFrame({tk: np.arange(5) + 101. for tk in [*JP_TICKERS, TOPIX_TICKER]}, index=dates), 'jp_open': pd.DataFrame({tk: np.arange(5) + 100. for tk in [*JP_TICKERS, TOPIX_TICKER]}, index=dates)}
    raw['us_close']['XLC'] = np.nan
    frame = preprocess_data(raw, strict_validation=True)
    assert frame['us_proxy_XLC'].all()
    assert np.isfinite(frame['us_cc_XLC']).all()
    assert not frame['us_proxy_XLRE'].any()
