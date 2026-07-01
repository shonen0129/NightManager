"""Novel Alpha Sources — Free-Thinking Exploration.

Beyond commodities and global indices, test truly novel lead-lag pathways:

  1. US futures overnight: ES=F, NQ=F, CL=F (more timely than cash close)
  2. Cross-asset ratios: copper/gold, oil/gold, silver/gold
  3. Credit spreads: HYG, LQD (credit risk → equity lead)
  4. Crypto: BTC-USD (weekend risk sentiment, 24/7 trading)
  5. Vol of vol: VVIX, OVX, GVZ (different vol regimes)
  6. Treasury curve: 2yr, 5yr, 30yr, 2s10s spread, 5s30s spread
  7. Korean ETFs: EWY, EWY.S (country-level), KWEB already tested
  8. Taiwan semi: TSM (TSMC ADR), SOXX already tested
  9. Australia: EWA (commodity-linked)
  10. US sector intraday: last-hour momentum via 1h data
  11. Currency crosses: AUD/JPY, CAD/JPY, NOK/JPY (risk sentiment)
  12. Baltic Dry Index (shipping → trade sectors)
  13. US put/call ratio (sentiment)
  14. Fear & Greed proxy: VIX + credit spread + momentum composite
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
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
# Data download
# ---------------------------------------------------------------------------

def download_novel_data(start: str = "2008-01-01") -> dict[str, pd.Series]:
    """Download all novel alpha sources via yfinance."""
    import yfinance as yf

    # Single-ticker sources (price → return)
    price_tickers = {
        # US futures (overnight session)
        "ES=F": "es_futures",
        "NQ=F": "nq_futures",
        "YM=F": "ym_futures",
        "RTY=F": "rty_futures",
        # Cross-asset ETFs / commodities for ratio computation
        "SI=F": "silver",
        # Credit
        "HYG": "hy_credit",
        "LQD": "ig_credit",
        # Crypto
        "BTC-USD": "btc",
        "ETH-USD": "eth",
        # Vol of vol
        "^VVIX": "vvix",
        # Treasury yields
        "^IRX": "us_13w_yield",
        "^FVX": "us_5yr_yield",
        "^TYX": "us_30yr_yield",
        # Country ETFs
        "EWY": "korea_etf",
        "EWA": "australia_etf",
        "EWC": "canada_etf",
        "EWG": "germany_etf",
        "EWQ": "france_etf",
        "EWH": "hongkong_etf",
        "EWT": "taiwan_etf",
        "EWS": "singapore_etf",
        "EWM": "malaysia_etf",
        "EPI": "india_etf",
        "EWZ": "brazil_etf",
        # Individual stocks as signals
        "TSM": "tsmc_adr",
        "AAPL": "apple",
        "NVDA": "nvidia",
        "AMD": "amd",
        "TSLA": "tesla",
        # Currency crosses (risk sentiment)
        "AUDJPY=X": "aud_jpy",
        "CADJPY=X": "cad_jpy",
        "NOKJPY=X": "nok_jpy",
        "AUDUSD=X": "aud_usd",
        "NZDUSD=X": "nzd_usd",
        # Additional commodities
        "ZC=F": "corn",
        "ZW=F": "wheat",
        "KC=F": "coffee",
        "CT=F": "cotton",
        "SB=F": "sugar",
        "CC=F": "cocoa",
        "PL=F": "platinum",
        "PA=F": "palladium",
        "RB=F": "gasoline",
        "HO=F": "heating_oil",
        # Additional sector ETFs
        "XLB": "us_materials",
        "XLU": "us_utilities",
        "XLP": "us_staples",
        "XLY": "us_discretionary",
        "XLRE": "us_realestate",
        "XLC": "us_comms",
        "XLV": "us_healthcare",
        "MTUM": "us_momentum",
        "VLUE": "us_value",
        "IUSG": "us_growth",
        "USMV": "us_minvol",
        "QUAL": "us_quality",
        "SIZE": "us_size",
        "MOAT": "us_moat",
    }

    # Diff-based tickers (yield/level → diff)
    diff_tickers = {"^IRX", "^FVX", "^TYX", "^VVIX"}

    all_returns = {}

    # Download in batches for speed
    batch_size = 20
    all_symbols = list(price_tickers.keys())
    for i in range(0, len(all_symbols), batch_size):
        batch = all_symbols[i:i+batch_size]
        batch_names = [price_tickers[s] for s in batch]
        logger.info("  Downloading batch %d/%d: %s...", i//batch_size+1,
                    (len(all_symbols)+batch_size-1)//batch_size, batch[:3])

        try:
            data = yf.download(batch, start=start, auto_adjust=False, progress=False)
            if data.empty:
                continue

            close = data["Close"]
            if isinstance(close, pd.Series):
                close = close.to_frame(name=batch_names[0])

            for symbol, name in zip(batch, batch_names):
                if symbol not in close.columns and name not in close.columns:
                    continue
                col = symbol if symbol in close.columns else name
                series = close[col].dropna()
                if len(series) < 200:
                    logger.warning("  SKIP %s (%s): only %d rows", symbol, name, len(series))
                    continue

                series.index = pd.to_datetime(series.index).tz_localize(None).normalize()

                if symbol in diff_tickers:
                    rets = series.diff().replace([np.inf, -np.inf], np.nan)
                else:
                    rets = series.pct_change().replace([np.inf, -np.inf], np.nan)

                all_returns[name] = rets
                logger.info("  OK: %s (%s) — %d rows", symbol, name, len(series))
        except Exception as e:
            logger.warning("  Batch download failed: %s", e)

    # --- Build cross-asset ratio signals ---
    logger.info("=== Building cross-asset ratio signals ===")

    # copper/gold ratio (risk-on indicator)
    if "copper" in all_returns and "gold" in all_returns:
        # Need price levels, but we only have returns. Reconstruct cumulative.
        # Actually, ratio of returns is not the same as return of ratio.
        # We need to download price levels for ratio computation.
        try:
            ratio_data = yf.download(["HG=F", "GC=F"], start=start, auto_adjust=False, progress=False)
            if not ratio_data.empty:
                copper_p = ratio_data["Close"]["HG=F"].dropna()
                gold_p = ratio_data["Close"]["GC=F"].dropna()
                common_idx = copper_p.index.intersection(gold_p.index)
                cu_au = (copper_p.loc[common_idx] / gold_p.loc[common_idx]).dropna()
                cu_au.index = pd.to_datetime(cu_au.index).tz_localize(None).normalize()
                all_returns["copper_gold_ratio"] = cu_au.pct_change().replace([np.inf, -np.inf], np.nan)
                logger.info("  OK: copper/gold ratio — %d rows", len(cu_au))
        except Exception as e:
            logger.warning("  copper/gold ratio failed: %s", e)

    # oil/gold ratio (inflation pressure)
    try:
        ratio_data = yf.download(["CL=F", "GC=F"], start=start, auto_adjust=False, progress=False)
        if not ratio_data.empty:
            oil_p = ratio_data["Close"]["CL=F"].dropna()
            gold_p = ratio_data["Close"]["GC=F"].dropna()
            common_idx = oil_p.index.intersection(gold_p.index)
            oil_au = (oil_p.loc[common_idx] / gold_p.loc[common_idx]).dropna()
            oil_au.index = pd.to_datetime(oil_au.index).tz_localize(None).normalize()
            all_returns["oil_gold_ratio"] = oil_au.pct_change().replace([np.inf, -np.inf], np.nan)
            logger.info("  OK: oil/gold ratio — %d rows", len(oil_au))
    except Exception as e:
        logger.warning("  oil/gold ratio failed: %s", e)

    # silver/gold ratio (precious metals spread, risk indicator)
    try:
        ratio_data = yf.download(["SI=F", "GC=F"], start=start, auto_adjust=False, progress=False)
        if not ratio_data.empty:
            si_p = ratio_data["Close"]["SI=F"].dropna()
            au_p = ratio_data["Close"]["GC=F"].dropna()
            common_idx = si_p.index.intersection(au_p.index)
            si_au = (si_p.loc[common_idx] / au_p.loc[common_idx]).dropna()
            si_au.index = pd.to_datetime(si_au.index).tz_localize(None).normalize()
            all_returns["silver_gold_ratio"] = si_au.pct_change().replace([np.inf, -np.inf], np.nan)
            logger.info("  OK: silver/gold ratio — %d rows", len(si_au))
    except Exception as e:
        logger.warning("  silver/gold ratio failed: %s", e)

    # --- Build treasury curve signals ---
    logger.info("=== Building treasury curve signals ===")
    try:
        curve_data = yf.download(["^IRX", "^FVX", "^TNX", "^TYX"],
                                  start=start, auto_adjust=False, progress=False)
        if not curve_data.empty:
            yields = curve_data["Close"]
            yields.index = pd.to_datetime(yields.index).tz_localize(None).normalize()

            # 2s10s spread (using 13w as proxy for short end if 2yr unavailable)
            if "^TNX" in yields.columns and "^IRX" in yields.columns:
                spread_2s10s = (yields["^TNX"] - yields["^IRX"]).dropna()
                all_returns["ts_2s10s"] = spread_2s10s.diff().replace([np.inf, -np.inf], np.nan)
                logger.info("  OK: 2s10s spread — %d rows", len(spread_2s10s))

            # 5s30s spread
            if "^TYX" in yields.columns and "^FVX" in yields.columns:
                spread_5s30s = (yields["^TYX"] - yields["^FVX"]).dropna()
                all_returns["ts_5s30s"] = spread_5s30s.diff().replace([np.inf, -np.inf], np.nan)
                logger.info("  OK: 5s30s spread — %d rows", len(spread_5s30s))

            # 10s30s spread
            if "^TYX" in yields.columns and "^TNX" in yields.columns:
                spread_10s30s = (yields["^TYX"] - yields["^TNX"]).dropna()
                all_returns["ts_10s30s"] = spread_10s30s.diff().replace([np.inf, -np.inf], np.nan)
                logger.info("  OK: 10s30s spread — %d rows", len(spread_10s30s))
    except Exception as e:
        logger.warning("  Treasury curve failed: %s", e)

    # --- Build credit spread signals ---
    logger.info("=== Building credit spread signals ===")
    if "hy_credit" in all_returns and "ig_credit" in all_returns:
        # Credit spread proxy: HYG return - LQD return (positive = HY outperforming = risk-on)
        hy = all_returns["hy_credit"]
        ig = all_returns["ig_credit"]
        common = hy.index.intersection(ig.index)
        credit_spread = (hy.loc[common] - ig.loc[common]).dropna()
        all_returns["credit_spread_hy_ig"] = credit_spread
        logger.info("  OK: HY-IG credit spread — %d rows", len(credit_spread))

    logger.info("Total novel alpha sources: %d", len(all_returns))
    return all_returns


# ---------------------------------------------------------------------------
# IC diagnostic
# ---------------------------------------------------------------------------

def _ticker_to_sector(tk: str) -> str:
    sectors = {
        "1617.T": "食品", "1618.T": "エネルギー", "1619.T": "建設・資材",
        "1620.T": "素材・化学", "1621.T": "医薬品", "1622.T": "自動車・輸送機",
        "1623.T": "鉄鋼・非鉄", "1624.T": "機械", "1625.T": "電機・精密",
        "1626.T": "情報通信", "1627.T": "電力・ガス", "1628.T": "運輸・物流",
        "1629.T": "商社・卸売", "1630.T": "小売", "1631.T": "銀行",
        "1632.T": "金融(除銀行)", "1633.T": "不動産",
    }
    return sectors.get(tk, tk)


def compute_ic_diagnostic(
    alt_data: dict[str, pd.Series],
    y_target: np.ndarray,
    sim_dates: pd.DatetimeIndex,
    jp_tickers: list[str],
    start_idx: int,
    lag: int = 1,
) -> pd.DataFrame:
    """Compute lagged time-series Rank IC for each signal × sector pair."""
    all_results = []

    for signal_name, signal_series in alt_data.items():
        signal_aligned = signal_series.reindex(sim_dates).shift(lag)

        for j, tk in enumerate(jp_tickers):
            y_j = y_target[:, j]
            pairs = []
            for i in range(start_idx, len(sim_dates)):
                sig_val = signal_aligned.iloc[i]
                y_val = y_j[i]
                if np.isfinite(sig_val) and np.isfinite(y_val):
                    pairs.append((sig_val, y_val))

            if len(pairs) < 50:
                continue

            arr = np.array(pairs)
            rho, pval = stats.spearmanr(arr[:, 0], arr[:, 1])
            n = len(arr)
            t_stat = rho * np.sqrt(n - 2) / np.sqrt(1 - rho**2) if abs(rho) < 1 else np.nan

            all_results.append({
                "signal": signal_name,
                "ticker": tk,
                "sector": _ticker_to_sector(tk),
                "rank_ic": float(rho),
                "t_stat": float(t_stat),
                "p_value": float(pval),
                "n": n,
                "significant": pval < 0.05,
            })

    return pd.DataFrame(all_results)


# ---------------------------------------------------------------------------
# Combined model
# ---------------------------------------------------------------------------

def build_combined_signal(
    blpx_signals: pd.DataFrame,
    alt_data: dict[str, pd.Series],
    significant_signals: list[tuple[str, str, float]],
    sim_dates: pd.DatetimeIndex,
    jp_tickers: list[str],
    blend_weight: float = 0.1,
) -> pd.DataFrame:
    """Build combined signal: BLPX + significant alternative signals."""
    combined = blpx_signals.reindex(sim_dates).fillna(0.0).copy()

    ticker_signals: dict[str, list[tuple[str, float]]] = {}
    for signal_name, ticker, ic in significant_signals:
        ticker_signals.setdefault(ticker, []).append((signal_name, ic))

    for j, tk in enumerate(jp_tickers):
        if tk not in ticker_signals:
            continue
        for signal_name, ic in ticker_signals[tk]:
            if signal_name not in alt_data:
                continue
            alt_series = alt_data[signal_name].reindex(sim_dates).shift(1)
            rolling_mean = alt_series.rolling(252, min_periods=60).mean()
            rolling_std = alt_series.rolling(252, min_periods=60).std()
            z_alt = (alt_series - rolling_mean) / rolling_std.replace(0, np.nan)
            adjustment = blend_weight * ic * z_alt
            combined[tk] = combined[tk] + adjustment.reindex(sim_dates).fillna(0.0).values

    return combined


class CombinedSignalModel:
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
            columns=lambda c: c.replace("jp_oc_", ""))
        return {
            "raw_pca_signals": empty, "residual_pca_signals": empty,
            "p4_signals": empty, "signals": blpx,
            "normalized_signals": blpx, "y_jp_oc_df": y_jp_oc_df,
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
    parser = argparse.ArgumentParser(description="Novel Alpha Sources Exploration")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--output-dir", default="artifacts/novel_alpha")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    yaml_path = str(ROOT / "configs" / "production.yaml")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading df_exec from cache...")
    df_exec = load_df_exec_from_local_cache()
    y_target = compute_jp_target_returns(df_exec, JP_TICKERS)
    sim_dates = df_exec.index
    start_idx = max(df_exec.index.searchsorted(pd.to_datetime("2015-01-01")), 60)

    # --- Download novel data ---
    cache_path = output_dir / "novel_data.pkl"
    if args.skip_download and cache_path.exists():
        logger.info("Loading cached novel data...")
        alt_data = pd.read_pickle(cache_path)
    else:
        alt_data = download_novel_data(start="2008-01-01")
        pd.to_pickle(alt_data, cache_path)
        logger.info("Cached novel data to %s", cache_path)

    if not alt_data:
        logger.error("No novel data available. Exiting.")
        return

    # --- IC Diagnostic ---
    logger.info("=== Computing IC diagnostic for %d novel signals ===", len(alt_data))
    ic_df = compute_ic_diagnostic(alt_data, y_target, sim_dates, JP_TICKERS, start_idx, lag=1)
    ic_df.to_csv(output_dir / "ic_diagnostic.csv", index=False)

    # Print signal summary
    print("\n" + "=" * 100)
    print("NOVEL ALPHA SOURCES — IC DIAGNOSTIC (1-day lag)")
    print("=" * 100)

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

    # Print top significant pairs
    sig_pairs = ic_df[ic_df["significant"]].sort_values("rank_ic", ascending=False)
    print(f"\n--- TOP 40 Significant Signal-Sector Pairs ({len(sig_pairs)} total) ---")
    print(f"{'Signal':<25} {'Sector':<16} {'Rank IC':<10} {'t-stat':<10} {'p-value':<12}")
    print("-" * 80)
    for _, row in sig_pairs.head(40).iterrows():
        print(f"{row['signal']:<25} {row['sector']:<16} {row['rank_ic']:<10.4f} "
              f"{row['t_stat']:<10.2f} {row['p_value']:<12.6f}")

    print(f"\n--- TOP 15 NEGATIVE ICs ---")
    for _, row in sig_pairs.sort_values("rank_ic").head(15).iterrows():
        print(f"{row['signal']:<25} {row['sector']:<16} {row['rank_ic']:<10.4f} "
              f"{row['t_stat']:<10.2f} {row['p_value']:<12.6f}")

    # --- Combined model test ---
    logger.info("=== Building combined model ===")
    cfg_base = build_config(yaml_path, blpx_overrides=BASE_PARAMS,
                            signal_components=SIGNAL_WEIGHTS)
    _BLP_CORR_CACHE.clear()
    _RAW_PCA_RESIDUAL_PCA_CACHE.clear()
    model_base = SectorRelativeEnsembleBLPEnhancedModel(cfg_base)
    pred_base = model_base.predict_signals(df_exec)
    blpx_signals = pred_base["signals"]

    sig_for_model = [(r["signal"], r["ticker"], r["rank_ic"])
                     for _, r in sig_pairs.iterrows()]

    blend_weights = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    all_results = []

    for bw in blend_weights:
        name = "baseline_blpx" if bw == 0.0 else f"combined_bw{bw:.2f}"
        logger.info("=== %s ===", name)
        combined = build_combined_signal(
            blpx_signals, alt_data, sig_for_model,
            sim_dates, JP_TICKERS, blend_weight=bw)
        model = CombinedSignalModel(combined, df_exec)
        m = run_backtest(name, model, df_exec, y_target, slippage_bps=args.slippage_bps)
        m["blend_weight"] = bw
        all_results.append(m)
        logger.info("  %s: Sharpe=%.4f AR=%.4f IC=%.4f", name, m["Sharpe_net"], m["AR_net"], m["Mean_Rank_IC"])

    # Print combined results
    print("\n" + "=" * 100)
    print("COMBINED MODEL RESULTS (BLPX + Novel Alpha Sources)")
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
