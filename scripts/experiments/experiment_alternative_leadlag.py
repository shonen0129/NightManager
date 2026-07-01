"""Alternative Lead-Lag Pathways — IC Diagnostic & Combined Model Test.

Current system: US sector ETF (15) cc return → BLPX → JP sector ETF (17) OC return
This experiment tests ALTERNATIVE lead signals:

  1. Commodities: WTI oil, Brent, Copper, Gold, Natural Gas
  2. Asian indices: Shanghai Composite, Hang Seng, KOSPI, Taiwan Weighted
  3. FX: USD/JPY, EUR/JPY, EUR/USD, DXY
  4. Rates: US 10yr yield, Japan 10yr (JGB)
  5. Volatility: VIX
  6. Global indices: DAX, FTSE, S&P500, NASDAQ
  7. China ETFs: FXI, KWEB, MCHI

For each source, compute 1-day lagged Rank IC vs each JP sector's OC return.
Then test if adding significant sources to BLPX improves Sharpe.

Data sources:
  - yfinance: CL=F, ^VIX, USDJPY=X, ^TNX, etc.
  - Stooq (HTTP): Shanghai Composite, KOSPI, Taiwan, JGB yield
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.execution.backtester import BacktestEngine
from leadlag.models.sre import compute_jp_target_returns
from leadlag.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
    _BLP_CORR_CACHE,
    _RAW_PCA_RESIDUAL_PCA_CACHE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_PARAMS = {
    "alpha_xx": 0.20, "alpha_yy": 0.50, "alpha_yx": 0.15,
    "lambda_pca": 0.10, "lambda_sector": 0.60, "beta_conf": 0.25,
    "rho": 0.01, "winsor_sigma": 3.0, "blp_window": 504,
    "blp_ewma_halflife": 120, "sector_eta": 0.5, "sector_gamma": 4.0,
}

SIGNAL_WEIGHTS = {
    "raw_pca": {"enabled": True, "weight": 0.2},
    "residual_pca": {"enabled": False, "weight": 0.0},
    "raw_blpx": {"enabled": True, "weight": 0.8},
    "residual_blpx": {"enabled": False, "weight": 0.0},
}


# ---------------------------------------------------------------------------
# Data downloaders
# ---------------------------------------------------------------------------

def download_yfinance(tickers: dict[str, str], start: str = "2008-01-01") -> pd.DataFrame:
    """Download daily close from yfinance. Returns DataFrame with named columns."""
    import yfinance as yf

    cols = {}
    for symbol, name in tickers.items():
        try:
            df = yf.download(symbol, start=start, auto_adjust=False, progress=False)
            if not df.empty and "Close" in df.columns:
                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
                close.name = name
                cols[name] = close
                logger.info("  yfinance OK: %s (%s) — %d rows", symbol, name, len(close))
            else:
                logger.warning("  yfinance EMPTY: %s (%s)", symbol, name)
        except Exception as e:
            logger.warning("  yfinance FAIL: %s (%s) — %s", symbol, name, e)

    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols)


def download_stooq(tickers: dict[str, str], start: str = "2008-01-01") -> pd.DataFrame:
    """Download daily close from Stooq via direct HTTP (no API key needed).

    Stooq URL format: https://stooq.com/q/d/l/?s={symbol}&d1={start}&d2={end}&i=d
    """
    end = pd.Timestamp.now().strftime("%Y%m%d")
    start_fmt = pd.to_datetime(start).strftime("%Y%m%d")

    cols = {}
    for symbol, name in tickers.items():
        try:
            url = f"https://stooq.com/q/d/l/?s={symbol}&d1={start_fmt}&d2={end}&i=d"
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.content) > 10:
                df = pd.read_csv(io.StringIO(resp.text))
                if "Date" in df.columns and "Close" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.set_index("Date")
                    close = df["Close"].astype(float)
                    close.index = close.index.normalize()
                    close.name = name
                    if len(close) > 100:
                        cols[name] = close
                        logger.info("  stooq OK: %s (%s) — %d rows", symbol, name, len(close))
                    else:
                        logger.warning("  stooq SHORT: %s (%s) — %d rows", symbol, name, len(close))
                else:
                    logger.warning("  stooq FORMAT: %s (%s) — columns: %s", symbol, name, list(df.columns))
            else:
                logger.warning("  stooq HTTP %d: %s (%s)", resp.status_code, symbol, name)
        except Exception as e:
            logger.warning("  stooq FAIL: %s (%s) — %s", symbol, name, e)

    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols)


def download_all_alternative_data(start: str = "2008-01-01") -> dict[str, pd.DataFrame]:
    """Download all alternative data sources.

    Returns dict of {source_name: DataFrame of daily returns}.
    """
    logger.info("=== Downloading alternative data sources ===")

    # yfinance sources
    yf_tickers = {
        "CL=F": "wti_oil",
        "BZ=F": "brent_oil",
        "HG=F": "copper",
        "GC=F": "gold",
        "NG=F": "natgas",
        "^VIX": "vix",
        "^TNX": "us_10yr_yield",
        "USDJPY=X": "usd_jpy",
        "EURJPY=X": "eur_jpy",
        "EURUSD=X": "eur_usd",
        "DX-Y.NYB": "dxy",
        "^GSPC": "sp500",
        "^IXIC": "nasdaq",
        "^DJI": "dow",
        "^GDAXI": "dax",
        "^FTSE": "ftse",
        "^HSI": "hang_seng",
        "^N225": "nikkei",
        "FXI": "china_large_cap",
        "KWEB": "china_internet",
        "MCHI": "msci_china",
        "EEM": "emerging_mkt",
        "UNG": "nat_gas_etf",
        "DBA": "agriculture",
        "DBC": "commodity_index",
        "SOXX": "semiconductor",
        "XLE": "us_energy",
        "XLF": "us_financial",
        "XLK": "us_tech",
    }
    yf_data = download_yfinance(yf_tickers, start=start)

    # Stooq sources (for Asian indices not well covered by yfinance)
    stooq_tickers = {
        "000001": "shanghai_comp",      # Shanghai Composite
        "399001": "shenzhen_comp",      # Shenzhen Component
        "KS11": "kospi",                # Korea KOSPI
        "TWII": "taiwan_weighted",      # Taiwan Weighted Index
        "STI": "straits_times",         # Singapore STI
        "JKSE": "jakarta_comp",         # Indonesia Jakarta Composite
        "SET": "set_index",             # Thailand SET
        "10JPYB": "japan_10yr_jgb",     # Japan 10yr JGB yield
    }
    stooq_data = download_stooq(stooq_tickers, start=start)

    # Combine and convert to returns
    all_data = {}
    for df, prefix in [(yf_data, "yf"), (stooq_data, "stooq")]:
        if df.empty:
            continue
        for col in df.columns:
            series = df[col].dropna()
            if len(series) < 200:
                continue
            # Convert to daily returns
            rets = series.pct_change().replace([np.inf, -np.inf], np.nan)
            # For VIX and yields, use diff instead of pct_change
            if col in ("vix", "us_10yr_yield", "japan_10yr_jgb"):
                rets = series.diff().replace([np.inf, -np.inf], np.nan)
            all_data[col] = rets

    logger.info("Total alternative data sources: %d", len(all_data))
    return all_data


# ---------------------------------------------------------------------------
# IC diagnostic
# ---------------------------------------------------------------------------

def compute_lagged_ic(
    alt_returns: pd.Series,
    jp_target: np.ndarray,
    sim_dates: pd.DatetimeIndex,
    jp_tickers: list[str],
    start_idx: int,
    lag: int = 1,
) -> pd.DataFrame:
    """Compute 1-day lagged Rank IC between alternative signal and each JP sector.

    Signal on day T → JP OC return on day T+lag.
    """
    # Align alternative returns to sim_dates (shift by lag)
    alt_aligned = alt_returns.reindex(sim_dates).shift(lag)

    ic_results = []
    for j, tk in enumerate(jp_tickers):
        y_j = jp_target[:, j]
        ic_list = []
        for i in range(start_idx, len(sim_dates)):
            alt_val = alt_aligned.iloc[i]
            y_val = y_j[i]
            if np.isnan(alt_val) or np.isnan(y_val):
                continue
            # Cross-sectional: we need panel data for rank IC
            # But alternative signal is a single value per day
            # So we compute time-series IC: corr(alt_signal, y_j) over rolling window
            # Actually, for a single signal → single target, we compute
            # the rank correlation across ALL days (aggregate IC)
            ic_list.append((alt_val, y_val))

        if len(ic_list) < 50:
            ic_results.append({"ticker": tk, "mean_ic": np.nan, "t_stat": np.nan, "n": 0})
            continue

        arr = np.array(ic_list)
        alt_vals = arr[:, 0]
        y_vals = arr[:, 1]

        # Time-series rank IC: Spearman correlation
        rho, pval = stats.spearmanr(alt_vals, y_vals)
        n = len(arr)
        t_stat = rho * np.sqrt(n - 2) / np.sqrt(1 - rho**2) if abs(rho) < 1 else np.nan
        ic_results.append({
            "ticker": tk, "mean_ic": float(rho), "t_stat": float(t_stat),
            "p_value": float(pval), "n": n,
        })

    return pd.DataFrame(ic_results)


def compute_cross_sectional_ic(
    alt_returns_dict: dict[str, pd.Series],
    jp_target: np.ndarray,
    sim_dates: pd.DatetimeIndex,
    jp_tickers: list[str],
    start_idx: int,
    lag: int = 1,
) -> pd.DataFrame:
    """Cross-sectional approach: for each day, rank JP sectors by predicted return
    (based on alternative signal sensitivity) and compute Rank IC with actual returns.

    Since alternative signals are market-level (not sector-level), we use a different
    approach: compute the time-series correlation between each alternative signal and
    each JP sector's return. This tells us which sectors are most predictable by
    each signal.

    Then for signals that affect different sectors differently, we can build a
    multi-signal model.
    """
    all_results = []

    for signal_name, signal_series in alt_returns_dict.items():
        signal_aligned = signal_series.reindex(sim_dates).shift(lag)

        for j, tk in enumerate(jp_tickers):
            y_j = jp_target[:, j]

            pairs = []
            for i in range(start_idx, len(sim_dates)):
                sig_val = signal_aligned.iloc[i]
                y_val = y_j[i]
                if np.isfinite(sig_val) and np.isfinite(y_val):
                    pairs.append((sig_val, y_val))

            if len(pairs) < 50:
                continue

            arr = np.array(pairs)
            sig_vals = arr[:, 0]
            y_vals = arr[:, 1]

            # Spearman rank correlation (time-series IC)
            rho, pval = stats.spearmanr(sig_vals, y_vals)
            n = len(arr)
            t_stat = rho * np.sqrt(n - 2) / np.sqrt(1 - rho**2) if abs(rho) < 1 else np.nan

            # Also compute Pearson for comparison
            r_pearson, _ = stats.pearsonr(sig_vals, y_vals)

            all_results.append({
                "signal": signal_name,
                "ticker": tk,
                "sector": _ticker_to_sector(tk),
                "rank_ic": float(rho),
                "pearson_r": float(r_pearson),
                "t_stat": float(t_stat),
                "p_value": float(pval),
                "n": n,
                "significant": pval < 0.05,
            })

    return pd.DataFrame(all_results)


def _ticker_to_sector(tk: str) -> str:
    """Map JP ticker to sector name."""
    sectors = {
        "1617.T": "食品", "1618.T": "エネルギー", "1619.T": "建設・資材",
        "1620.T": "素材・化学", "1621.T": "医薬品", "1622.T": "自動車・輸送機",
        "1623.T": "鉄鋼・非鉄", "1624.T": "機械", "1625.T": "電機・精密",
        "1626.T": "情報通信", "1627.T": "電力・ガス", "1628.T": "運輸・物流",
        "1629.T": "商社・卸売", "1630.T": "小売", "1631.T": "銀行",
        "1632.T": "金融(除銀行)", "1633.T": "不動産",
    }
    return sectors.get(tk, tk)


# ---------------------------------------------------------------------------
# Combined model: BLPX + alternative signals
# ---------------------------------------------------------------------------

def build_combined_signal(
    blpx_signals: pd.DataFrame,
    alt_data: dict[str, pd.Series],
    significant_signals: list[tuple[str, str, float]],
    sim_dates: pd.DatetimeIndex,
    jp_tickers: list[str],
    start_idx: int,
    blend_weight: float = 0.1,
) -> pd.DataFrame:
    """Build combined signal: BLPX + alternative signals.

    For each significant (signal, ticker) pair, add a linear adjustment:
        signal_adjusted[j] += blend_weight * z(alt_signal[t]) * beta[signal, j]

    where beta is the regression coefficient from the IC analysis.
    """
    n_j = len(jp_tickers)
    combined = blpx_signals.reindex(sim_dates).fillna(0.0).copy()

    # Group significant signals by ticker
    ticker_signals: dict[str, list[tuple[str, float]]] = {}
    for signal_name, ticker, ic in significant_signals:
        if ticker not in ticker_signals:
            ticker_signals[ticker] = []
        ticker_signals[ticker].append((signal_name, ic))

    for j, tk in enumerate(jp_tickers):
        if tk not in ticker_signals:
            continue

        for signal_name, ic in ticker_signals[tk]:
            if signal_name not in alt_data:
                continue
            alt_series = alt_data[signal_name].reindex(sim_dates).shift(1)

            # Z-score the alternative signal (rolling, no look-ahead)
            rolling_mean = alt_series.rolling(252, min_periods=60).mean()
            rolling_std = alt_series.rolling(252, min_periods=60).std()
            z_alt = (alt_series - rolling_mean) / rolling_std.replace(0, np.nan)

            # Add to BLPX signal: scaled by IC and blend weight
            adjustment = blend_weight * ic * z_alt
            combined[tk] = combined[tk] + adjustment.reindex(sim_dates).fillna(0.0).values

    return combined


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class CombinedSignalModel:
    """Uses pre-computed combined signals for backtest."""

    def __init__(self, combined_signals, df_exec):
        self.combined_signals = combined_signals
        self.df_exec = df_exec
        self.n_j = len(JP_TICKERS)
        self.n_u = len(US_TICKERS)
        self.corr_window = 60
        self.slippage_bps = 5.0
        self.weight_mode = "signal"
        self.q = 0.3
        self.overnight_alpha_long = 0.0
        self.overnight_alpha_short = 0.0
        self.normalization_method = "zscore"
        self._weight_counter = 0

    def predict_signals(self, df_exec):
        start_dt = pd.to_datetime("2015-01-01")
        start_idx = max(df_exec.index.searchsorted(start_dt), self.corr_window)
        self._weight_counter = start_idx

        T = len(df_exec)
        sim_dates = df_exec.index
        blpx = self.combined_signals.reindex(sim_dates).fillna(0.0)
        empty = pd.DataFrame(np.zeros((T, self.n_j)), index=sim_dates, columns=JP_TICKERS)
        y_jp_oc_df = df_exec[[f"jp_oc_{tk}" for tk in JP_TICKERS]].rename(
            columns=lambda c: c.replace("jp_oc_", "")
        )
        return {
            "raw_pca_signals": empty,
            "residual_pca_signals": empty,
            "p4_signals": empty,
            "signals": blpx,
            "normalized_signals": blpx,
            "y_jp_oc_df": y_jp_oc_df,
        }

    def build_weights(self, signal, q=None):
        from leadlag.core.signal import build_weights
        if self._weight_counter < len(self.df_exec):
            w = build_weights(signal, q=self.q, n_j=self.n_j,
                              weight_mode="signal", enforce_sign=False)
            self._weight_counter += 1
            return w
        return np.zeros(self.n_j)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_rank_ic(signals_df, y_target, sim_dates, start_idx):
    y_df = pd.DataFrame(y_target, index=sim_dates, columns=JP_TICKERS)
    ic_list = []
    for i in range(start_idx, len(sim_dates)):
        date = sim_dates[i]
        if date not in signals_df.index:
            continue
        sig_t = signals_df.loc[date].values
        y_t = y_df.loc[date].values
        valid = ~(np.isnan(sig_t) | np.isnan(y_t))
        if valid.sum() >= 3:
            rho, _ = stats.spearmanr(sig_t[valid], y_t[valid])
            if np.isfinite(rho):
                ic_list.append(float(rho))
    if not ic_list:
        return np.nan, np.nan
    ic_arr = np.array(ic_list)
    mean_ic = float(np.mean(ic_arr))
    std_ic = float(np.std(ic_arr, ddof=1))
    icir = (mean_ic / std_ic * np.sqrt(252)) if std_ic > 1e-8 else np.nan
    return mean_ic, icir


def run_backtest(name, model, df_exec, y_target, slippage_bps=5.0):
    t0 = time.perf_counter()
    results = BacktestEngine.run_backtest(
        model, df_exec=df_exec, start_date="2015-01-01",
        overnight_alpha_long=0.75, overnight_alpha_short=0.5,
        buy_interest_annual=0.025, borrow_fee_annual=0.0115,
        reverse_fee_bps=2.0, slippage_bps=slippage_bps,
    )
    elapsed = time.perf_counter() - t0
    dr = results["daily_returns"]
    ar = float(dr.mean() * 245)
    vol = float(dr.std(ddof=1) * np.sqrt(245))
    sharpe = ar / vol if vol > 0 else np.nan
    wealth = (1.0 + dr).cumprod()
    mdd = float(((wealth / wealth.cummax()) - 1.0).min())
    turnover = float(results["daily_turnover"].mean())
    gross_exp = float(results["daily_gross_exps"].mean())
    sim_dates = df_exec.index
    start_idx = max(df_exec.index.searchsorted(pd.to_datetime("2015-01-01")), 60)
    mean_ic, icir = compute_rank_ic(results["signals"], y_target, sim_dates, start_idx)
    return {
        "name": name, "AR_net": ar, "Vol_net": vol, "Sharpe_net": sharpe,
        "MDD": mdd, "Turnover": turnover, "Gross_exp": gross_exp,
        "Mean_Rank_IC": mean_ic, "ICIR": icir, "elapsed_s": elapsed,
    }


def build_config(yaml_path, blpx_overrides=None, signal_components=None):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if blpx_overrides:
        cfg.setdefault("blpx", {}).update(blpx_overrides)
    if signal_components is not None:
        cfg["signal_components"] = signal_components
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Alternative Lead-Lag Pathways")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--output-dir", default="artifacts/alternative_leadlag")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, use cached alternative data")
    args = parser.parse_args()

    yaml_path = str(ROOT / "configs" / "production.yaml")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading df_exec from cache...")
    df_exec = load_df_exec_from_local_cache()
    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)
    sim_dates = df_exec.index
    start_idx = max(df_exec.index.searchsorted(pd.to_datetime("2015-01-01")), 60)

    # --- Download alternative data ---
    cache_path = output_dir / "alt_data.pkl"
    if args.skip_download and cache_path.exists():
        logger.info("Loading cached alternative data...")
        alt_data = pd.read_pickle(cache_path)
    else:
        alt_data = download_all_alternative_data(start="2008-01-01")
        pd.to_pickle(alt_data, cache_path)
        logger.info("Cached alternative data to %s", cache_path)

    if not alt_data:
        logger.error("No alternative data available. Exiting.")
        return

    # --- IC Diagnostic ---
    logger.info("=== Computing lagged IC for all alternative signals ===")
    ic_df = compute_cross_sectional_ic(
        alt_data, y_target, sim_dates, JP_TICKERS, start_idx, lag=1
    )
    ic_df.to_csv(output_dir / "ic_diagnostic.csv", index=False)

    # Print IC summary by signal
    print("\n" + "=" * 100)
    print("ALTERNATIVE LEAD-LAG IC DIAGNOSTIC (1-day lag)")
    print("=" * 100)

    # Aggregate by signal
    signal_summary = ic_df.groupby("signal").agg(
        mean_ic=("rank_ic", "mean"),
        max_ic=("rank_ic", "max"),
        min_ic=("rank_ic", "min"),
        n_significant=("significant", "sum"),
        n_sectors=("ticker", "count"),
    ).sort_values("n_significant", ascending=False)

    print(f"\n{'Signal':<25} {'Mean IC':<10} {'Max IC':<10} {'Min IC':<10} "
          f"{'#Sig':<6} {'#Sectors':<10}")
    print("-" * 75)
    for signal_name, row in signal_summary.iterrows():
        print(f"{signal_name:<25} {row['mean_ic']:<10.4f} {row['max_ic']:<10.4f} "
              f"{row['min_ic']:<10.4f} {int(row['n_significant']):<6} "
              f"{int(row['n_sectors']):<10}")

    # Print significant signal-sector pairs
    sig_pairs = ic_df[ic_df["significant"]].sort_values("rank_ic", ascending=False)
    print(f"\n--- Significant Signal-Sector Pairs (p<0.05) — {len(sig_pairs)} pairs ---")
    print(f"{'Signal':<25} {'Sector':<20} {'Rank IC':<10} {'t-stat':<10} {'p-value':<10}")
    print("-" * 80)
    for _, row in sig_pairs.head(30).iterrows():
        print(f"{row['signal']:<25} {row['sector']:<20} {row['rank_ic']:<10.4f} "
              f"{row['t_stat']:<10.2f} {row['p_value']:<10.4f}")

    # Also show strongest negative ICs
    print(f"\n--- Strongest Negative ICs ---")
    print(f"{'Signal':<25} {'Sector':<20} {'Rank IC':<10} {'t-stat':<10} {'p-value':<10}")
    print("-" * 80)
    for _, row in sig_pairs.sort_values("rank_ic").head(15).iterrows():
        print(f"{row['signal']:<25} {row['sector']:<20} {row['rank_ic']:<10.4f} "
              f"{row['t_stat']:<10.2f} {row['p_value']:<10.4f}")

    # --- Combined model test ---
    logger.info("=== Building combined model ===")

    # Get baseline BLPX signals
    cfg_base = build_config(yaml_path, blpx_overrides=BASE_PARAMS,
                            signal_components=SIGNAL_WEIGHTS)
    _BLP_CORR_CACHE.clear()
    _RAW_PCA_RESIDUAL_PCA_CACHE.clear()
    model_base = SectorRelativeEnsembleBLPEnhancedModel(cfg_base)
    pred_base = model_base.predict_signals(df_exec)
    blpx_signals = pred_base["signals"]

    # Select significant signals for combined model
    # Use top N most significant pairs
    sig_for_model = []
    for _, row in sig_pairs.iterrows():
        sig_for_model.append((row["signal"], row["ticker"], row["rank_ic"]))

    # Test different blend weights
    blend_weights = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    all_results = []

    for bw in blend_weights:
        if bw == 0.0:
            name = "baseline_blpx"
        else:
            name = f"combined_bw{bw:.2f}"

        logger.info("=== %s ===", name)
        combined = build_combined_signal(
            blpx_signals, alt_data, sig_for_model,
            sim_dates, JP_TICKERS, start_idx, blend_weight=bw
        )
        model = CombinedSignalModel(combined, df_exec)
        m = run_backtest(name, model, df_exec, y_target, slippage_bps=args.slippage_bps)
        m["blend_weight"] = bw
        all_results.append(m)
        logger.info("  %s: Sharpe=%.4f AR=%.4f IC=%.4f", name, m["Sharpe_net"], m["AR_net"], m["Mean_Rank_IC"])

    # Print combined model results
    print("\n" + "=" * 100)
    print("COMBINED MODEL RESULTS (BLPX + Alternative Signals)")
    print("=" * 100)

    baseline_sharpe = all_results[0]["Sharpe_net"]
    baseline_ic = all_results[0]["Mean_Rank_IC"]

    print(f"\n{'Name':<25} {'Sharpe':<10} {'AR':<10} {'Vol':<10} {'MDD%':<8} "
          f"{'Turnover':<10} {'IC':<10} {'ICIR':<8} {'ΔSharpe':<8} {'ΔIC':<8}")
    print("-" * 110)
    for r in all_results:
        delta_s = r["Sharpe_net"] - baseline_sharpe if np.isfinite(r["Sharpe_net"]) else np.nan
        delta_ic = r["Mean_Rank_IC"] - baseline_ic if np.isfinite(r["Mean_Rank_IC"]) else np.nan
        print(f"{r['name']:<25} {r['Sharpe_net']:<10.4f} {r['AR_net']:<10.4f} {r['Vol_net']:<10.4f} "
              f"{r['MDD']*100:<8.2f} {r['Turnover']:<10.2f} {r['Mean_Rank_IC']:<10.4f} "
              f"{r['ICIR']:<8.2f} {delta_s:+.4f} {delta_ic:+.4f}")

    valid = [r for r in all_results if r["blend_weight"] > 0 and np.isfinite(r["Sharpe_net"])]
    if valid:
        best = max(valid, key=lambda x: x["Sharpe_net"])
        print(f"\nBest: {best['name']} Sharpe={best['Sharpe_net']:.4f} "
              f"(ΔSharpe={best['Sharpe_net']-baseline_sharpe:+.4f}, "
              f"ΔIC={best['Mean_Rank_IC']-baseline_ic:+.4f})")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "combined_results.csv", index=False)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
