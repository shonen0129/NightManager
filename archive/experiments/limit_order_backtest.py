"""指値エントリーバックテスト検証スクリプト

目的:
  寄付き成行エントリーを指値エントリーに変更した場合の影響を検証。
  - 主系列: シグナル理論価格（r_hat_cc）基準に対し、利益マージン m bps を確保
  - 対照群: 前日終値から k bps オフセット

使用方法:
  cd <project_root>
  python tools/limit_order_backtest.py [--start-date 2015-01-01] [--oos-start-date 2020-01-01]

仮定・注意事項 (約定モデル):
  1. 約定判定: 当日の High/Low のみ使用
     - ロング買い指値: P_low <= 指値 → 約定
     - ショート売り指値: P_high >= 指値 → 約定
  2. 約定価格の決定:
     - 寄付き時点で既に有利: min(指値, P_open) / max(指値, P_open)
     - 日中到達: 指値価格
  3. 不約定銘柄はノーポジション（持ち越し禁止）
  4. High/Low は日中のいずれかの時点での最高値/最安値であり、
     到達のタイミング（寄付き直後 vs 引け前）は区別できない（日足データの限界）
  5. r_hat_cc 定義: compute_signal() が返す r_hat_jp_cc
     = mu_jp + sigma_jp * z_hat_j_t1（ギャップ補正前の PCA 予測 CC リターン）
     フェア終値 = P_close[t-1] * (1 + r_hat_cc)

ルックアヘッドバイアス対策:
  - 指値価格: P_close[t-1]（前日終値）+ r_hat_cc（前日の米国情報のみで計算）
  - 約定判定: 当日の P_high, P_low のみ
  - ウェイト: 前日シグナルで決定（既存ロジックと同一）
  - 決済: 当日終値（P_close）

スリッページ仮定:
  - 成行版: 片道5bps（往復10bps × グロスエクスポージャー）
  - 指値版 slip_on_entry=False: エントリー側スリッページ不要（指値で確定）、
    決済（引け成行）のみ片道5bps（往復半分を適用）
  - 指値版 slip_on_entry=True（保守的）: 往復5bps を維持

作成: tools/limit_order_backtest.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# --- パス設定 ---
_TOOLS_DIR = Path(__file__).parent
_PROJECT_ROOT = _TOOLS_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
_DATA_DIR = _PROJECT_ROOT / "data"
_RESULTS_DIR = _PROJECT_ROOT / "results"
_HIGHLOW_CACHE_PATH = _DATA_DIR / "jp_highlow.pkl"

sys.path.insert(0, str(_SRC_DIR))

from config import STRATEGY_DEFAULTS, N_US_ASSETS, N_JP_ASSETS
from data.ticker_registry import JP_TICKERS, JP_TICKERS_WITH_TOPIX
from domain.models.types import StrategyConfig
from domain.signals import lead_lag as signals
from performance import calculate_metrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 245
DEFAULT_SLIPPAGE_BPS = 5.0
THEORY_MARGINS_BPS = [0, 5, 10, 20, 30, 50]  # m
PREV_CLOSE_OFFSETS_BPS = [-10, -5, 0, 5, 10, 20]  # k
OOS_DEFAULT_START = "2020-01-01"


# ---------------------------------------------------------------------------
# 1. High/Low データ取得・キャッシュ
# ---------------------------------------------------------------------------

def download_jp_highlow(
    cache_path: Path = _HIGHLOW_CACHE_PATH,
    start_date: str = "2009-01-01",
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """JP ETF の日足 High/Low を yfinance から取得・キャッシュ。

    Returns:
        {"jp_high": DataFrame, "jp_low": DataFrame}
        Index: date (tz-naive), Columns: ticker names (JP_TICKERS_WITH_TOPIX)
    """
    import yfinance as yf

    # キャッシュチェック
    if cache_path.exists() and not force_refresh:
        try:
            cached = pd.read_pickle(cache_path)
            if isinstance(cached, dict) and "jp_high" in cached and "jp_low" in cached:
                # データが古すぎないか確認（当日データは含まれない前提）
                latest = cached["jp_high"].index.max()
                if (pd.Timestamp.now() - latest).days <= 2:
                    logger.info("High/Low キャッシュを使用 (最終: %s)", latest.date())
                    return cached
                else:
                    logger.info("High/Low キャッシュが古い (最終: %s)、更新します", latest.date())
        except Exception as e:
            logger.warning("High/Low キャッシュ読み込み失敗: %s、再ダウンロード", e)

    logger.info("JP ETF High/Low をダウンロード中... (ティッカー数=%d)", len(JP_TICKERS_WITH_TOPIX))
    print("  yfinance から JP ETF High/Low をダウンロード中 (約30秒〜1分)...")

    raw = yf.download(
        JP_TICKERS_WITH_TOPIX,
        start=start_date,
        end=None,
        auto_adjust=False,
        progress=False,
    )

    if raw.empty:
        raise ValueError("yfinance から High/Low データを取得できませんでした")

    def _extract_clean(price_type: str) -> pd.DataFrame:
        df = raw[price_type].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df = df.sort_index()
        # 列名を JP_TICKERS_WITH_TOPIX 順に揃える
        available = [t for t in JP_TICKERS_WITH_TOPIX if t in df.columns]
        return df[available]

    result = {
        "jp_high": _extract_clean("High"),
        "jp_low": _extract_clean("Low"),
        "jp_open": _extract_clean("Open"),   # 追加: 寄付き価格（約定モデル用）
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(result, cache_path)
    logger.info("High/Low データをキャッシュしました: %s", cache_path)
    print(f"  High/Low データをキャッシュ: {cache_path}")

    return result


# ---------------------------------------------------------------------------
# 2. データ準備: df_exec + High/Low の結合
# ---------------------------------------------------------------------------

def load_df_exec() -> pd.DataFrame:
    """既存のキャッシュまたはダウンロードで df_exec を構築する。"""
    from data_loader import download_data, preprocess_data
    from data.cache import is_decision_cache_valid, load_decision_cache, save_decision_cache

    if is_decision_cache_valid():
        logger.info("decision_cache を読み込み (fast path)")
        return load_decision_cache()

    logger.info("etf_data.pkl からデータをロード・前処理中...")
    data = download_data(beta_window=STRATEGY_DEFAULTS["beta_window"])
    df_exec = preprocess_data(data, beta_window=STRATEGY_DEFAULTS["beta_window"])
    save_decision_cache(df_exec)
    return df_exec


def build_ohlc_arrays(
    df_exec: pd.DataFrame,
    highlow: dict[str, pd.DataFrame],
) -> dict[str, np.ndarray]:
    """df_exec の trade_date に対応する High/Low/Open 配列を構築。

    Returns:
        {
            "jp_high": (T, N_J) array,
            "jp_low":  (T, N_J) array,
            "jp_open_direct": (T, N_J) array,  ← highlow からの Open（Cross-check 用）
        }
    """
    trade_dates = df_exec.index
    tickers = JP_TICKERS  # 17銘柄のみ（TOPIX除く）

    jp_high = highlow["jp_high"].reindex(index=trade_dates, columns=tickers)
    jp_low = highlow["jp_low"].reindex(index=trade_dates, columns=tickers)
    jp_open_direct = highlow.get("jp_open", pd.DataFrame()).reindex(
        index=trade_dates, columns=tickers
    )

    return {
        "jp_high": np.array(jp_high, dtype=float),    # (T, 17)
        "jp_low": np.array(jp_low, dtype=float),       # (T, 17)
        "jp_open_direct": np.array(jp_open_direct, dtype=float),  # (T, 17)
    }


# ---------------------------------------------------------------------------
# 3. シグナル抽出ループ
# ---------------------------------------------------------------------------

def run_signal_extraction(
    df_exec: pd.DataFrame,
    config: StrategyConfig,
    start_date: str = "2015-01-01",
) -> pd.DataFrame:
    """各日×銘柄の シグナル・ウェイト・r_hat_cc・価格を計算して返す。

    Returns:
        columns: trade_date, ticker_idx, ticker, signal, r_hat_cc,
                 weight, is_long, is_short,
                 P_close_prev, P_open, P_oc_return
        index: RangeIndex
    """
    all_cc_cols = [
        c for c in df_exec.columns
        if c.startswith("us_cc_") or c.startswith("jp_cc_")
    ]
    jp_oc_cols = [c for c in df_exec.columns if c.startswith("jp_oc_")]
    jp_close_sig_cols = [c for c in df_exec.columns if c.startswith("jp_close_sig_")]
    jp_open_trade_cols = [c for c in df_exec.columns if c.startswith("jp_open_trade_")]
    gap_cols = [c for c in df_exec.columns if c.startswith("jp_gap_")]
    beta_cols = [c for c in df_exec.columns if c.startswith("jp_beta_")]

    all_returns = df_exec[all_cc_cols].values
    date_index = df_exec.index.values
    n_u = N_US_ASSETS
    n_j = N_JP_ASSETS

    # ベースライン相関行列
    c_full = signals.compute_baseline_correlation(
        all_returns, date_index, config.ewma_half_life
    )
    v0_static = signals.build_v3_static(n_u, n_j, config.include_v4_prior)
    base_vectors = signals.build_base_vectors(n_u, n_j)
    v1, v2 = base_vectors["v1"], base_vectors["v2"]

    jp_gap = df_exec[gap_cols].values if len(gap_cols) == n_j else None
    jp_beta = df_exec[beta_cols].values if len(beta_cols) == n_j else None
    topix_night = (
        df_exec["topix_night_return"].values
        if "topix_night_return" in df_exec.columns
        else None
    )
    jp_oc = df_exec[jp_oc_cols].values if jp_oc_cols else None
    jp_close_sig = df_exec[jp_close_sig_cols].values if jp_close_sig_cols else None
    jp_open_trade = df_exec[jp_open_trade_cols].values if jp_open_trade_cols else None

    start_idx = max(
        df_exec.index.searchsorted(pd.to_datetime(start_date)),
        config.corr_window,
    )

    # Dispersion history 初期化
    dispersion_history: list[float] = []
    history_start = max(0, start_idx - 60)
    for hist_i in range(history_start, start_idx):
        gap_hist = (
            np.nan_to_num(jp_gap[hist_i], nan=0.0) if jp_gap is not None else np.zeros(n_j)
        )
        betas_hist = np.asarray(jp_beta[hist_i], dtype=float) if jp_beta is not None else None
        topix_hist = float(topix_night[hist_i]) if topix_night is not None else None
        sig_hist = signals.compute_signal(
            all_returns, hist_i, n_u, config.corr_window, c_full, v0_static,
            v1, v2, config.k, config.lambda_reg, config.lambda_lw, config.lw_target,
            config.ewma_half_life, v3_dynamic=(config.v3_mode == "dynamic"),
            gap_override=gap_hist if config.signal_mode == "gap_residual" else None,
            gap_open_coef=config.gap_open_coef,
            topix_beta_coef=config.topix_beta_coef,
            betas_t=betas_hist, topix_night_t=topix_hist,
        )
        ind = signals.compute_dispersion_indicator(
            np.asarray(sig_hist["signal"], dtype=float),
            config.q, n_j, config.dispersion_metric,
        )
        dispersion_history.append(ind)

    logger.info("シグナル抽出: %d 日処理中...", len(df_exec) - start_idx)
    records = []

    for i in range(start_idx, len(df_exec)):
        t_trade = df_exec.index[i]

        gap_t = (
            np.nan_to_num(jp_gap[i], nan=0.0) if jp_gap is not None else np.zeros(n_j)
        )
        betas_t = np.asarray(jp_beta[i], dtype=float) if jp_beta is not None else None
        topix_t = float(topix_night[i]) if topix_night is not None else None

        sig_result = signals.compute_signal(
            all_returns, i, n_u, config.corr_window, c_full, v0_static,
            v1, v2, config.k, config.lambda_reg, config.lambda_lw, config.lw_target,
            config.ewma_half_life, v3_dynamic=(config.v3_mode == "dynamic"),
            gap_override=gap_t if config.signal_mode == "gap_residual" else None,
            gap_open_coef=config.gap_open_coef,
            topix_beta_coef=config.topix_beta_coef,
            betas_t=betas_t, topix_night_t=topix_t,
        )

        signal = np.asarray(sig_result["signal"], dtype=float)
        r_hat_cc = np.asarray(sig_result["r_hat_jp_cc"], dtype=float)  # ギャップ補正前CC予測

        # ウェイト計算（既存ロジック）
        weights = signals.build_weights(signal, config.q, n_j, config.weight_mode)
        disp_ind = signals.compute_dispersion_indicator(
            signal, config.q, n_j, config.dispersion_metric
        )
        scale = signals.dispersion_scale(disp_ind, dispersion_history, config.dispersion_filter)
        dispersion_history.append(disp_ind)
        scaled_weights = weights * scale

        # 価格（前日終値・当日始値・当日終値）
        p_close_prev = jp_close_sig[i] if jp_close_sig is not None else np.full(n_j, np.nan)
        p_open = jp_open_trade[i] if jp_open_trade is not None else np.full(n_j, np.nan)
        r_oc = jp_oc[i] if jp_oc is not None else np.zeros(n_j)

        for j in range(n_j):
            records.append({
                "trade_date": t_trade,
                "ticker_idx": j,
                "ticker": JP_TICKERS[j],
                "signal": signal[j],
                "r_hat_cc": r_hat_cc[j],
                "weight": scaled_weights[j],
                "is_long": scaled_weights[j] > 1e-12,
                "is_short": scaled_weights[j] < -1e-12,
                "P_close_prev": p_close_prev[j],  # 前日終値（sig_date close）
                "P_open": p_open[j],               # 当日始値（trade_date open）
                "r_oc": r_oc[j],                   # OC リターン（大引けリターン計算用）
            })

    df_signals = pd.DataFrame(records)
    logger.info("シグナル抽出完了: %d 行 (%d 日)", len(df_signals), len(df_signals["trade_date"].unique()))
    return df_signals


# ---------------------------------------------------------------------------
# 4. 指値価格・約定判定エンジン
# ---------------------------------------------------------------------------

@dataclass
class LimitOrderScenario:
    basis: str              # "market" | "theory" | "prev_close"
    margin_bps: float       # m (theory) or k (prev_close)
    renormalize: bool = False   # 約定銘柄で再正規化してグロスを2に戻すか
    slip_on_entry: bool = True  # エントリー側スリッページを課すか
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS

    @property
    def label(self) -> str:
        if self.basis == "market":
            return "baseline_market"
        bps_str = f"{self.margin_bps:+.0f}bps" if self.basis == "prev_close" else f"{self.margin_bps:.0f}bps"
        renorm = "_renorm" if self.renormalize else ""
        slip = "_slip" if self.slip_on_entry else "_noslip"
        return f"{self.basis}_{bps_str}{renorm}{slip}"


def compute_limit_prices(
    df_signals: pd.DataFrame,
    scenario: LimitOrderScenario,
) -> pd.Series:
    """各銘柄・各日の指値価格を計算。

    Returns:
        Series: 指値価格 (NaN = 指値なし/成行)
    """
    if scenario.basis == "market":
        # 成行: 指値なし（NaN）
        return pd.Series(np.nan, index=df_signals.index)

    p_close_prev = df_signals["P_close_prev"].values
    r_hat_cc = df_signals["r_hat_cc"].values
    is_long = df_signals["is_long"].values
    is_short = df_signals["is_short"].values
    m_bps = scenario.margin_bps
    k_bps = scenario.margin_bps

    limit_prices = np.full(len(df_signals), np.nan)

    if scenario.basis == "theory":
        # フェア価格 = P_close_prev * (1 + r_hat_cc)
        p_fair = p_close_prev * (1.0 + r_hat_cc)
        # ロング: フェア価格より安く買う
        limit_prices[is_long] = p_fair[is_long] * (1.0 - m_bps / 10000.0)
        # ショート: フェア価格より高く売る
        limit_prices[is_short] = p_fair[is_short] * (1.0 + m_bps / 10000.0)

    elif scenario.basis == "prev_close":
        # 前日終値からオフセット
        # ロング: 前日終値より安く買う（k>0 → 有利方向、k<0 → 不利方向）
        limit_prices[is_long] = p_close_prev[is_long] * (1.0 - k_bps / 10000.0)
        # ショート: 前日終値より高く売る
        limit_prices[is_short] = p_close_prev[is_short] * (1.0 + k_bps / 10000.0)

    return pd.Series(limit_prices, index=df_signals.index)


def evaluate_fills(
    df_signals: pd.DataFrame,
    limit_prices: pd.Series,
    ohlc: dict[str, np.ndarray],
    df_exec: pd.DataFrame,
    scenario: LimitOrderScenario,
) -> pd.DataFrame:
    """約定判定と約定価格の決定。

    約定ルール:
    - 成行: P_open で約定
    - ロング指値（買い）: P_low <= 指値 → 約定
      約定価格 = min(指値, P_open)  ← P_open <= 指値 なら寄付き価格で約定
    - ショート指値（売り）: P_high >= 指値 → 約定
      約定価格 = max(指値, P_open)  ← P_open >= 指値 なら寄付き価格で約定

    Args:
        ohlc: {"jp_high": (T, N_J), "jp_low": (T, N_J)} arrays

    Returns:
        df_signals + ["limit_price", "filled", "fill_price"]
    """
    # trade_date -> index in df_exec
    trade_dates = df_exec.index
    date_to_idx = {d: i for i, d in enumerate(trade_dates)}

    jp_high_arr = ohlc["jp_high"]   # (T, 17)
    jp_low_arr = ohlc["jp_low"]     # (T, 17)

    result = df_signals.copy()
    result["limit_price"] = limit_prices.values
    result["filled"] = False
    result["fill_price"] = np.nan
    result["fill_type"] = ""  # "market" | "open_gap" | "intraday"

    lp = limit_prices.values
    p_open = df_signals["P_open"].values
    is_long = df_signals["is_long"].values
    is_short = df_signals["is_short"].values
    is_active = is_long | is_short  # ロングorショートに選ばれた銘柄

    filled = np.zeros(len(df_signals), dtype=bool)
    fill_price = np.full(len(df_signals), np.nan)
    fill_type = np.full(len(df_signals), "", dtype=object)

    for idx, row in enumerate(df_signals.itertuples(index=False)):
        if not is_active[idx]:
            continue

        trade_date = row.trade_date
        j = row.ticker_idx
        t_idx = date_to_idx.get(trade_date)
        if t_idx is None:
            continue

        p_h = jp_high_arr[t_idx, j]
        p_l = jp_low_arr[t_idx, j]
        p_o = p_open[idx]
        lp_val = lp[idx]

        # NaN チェック
        if not (np.isfinite(p_h) and np.isfinite(p_l) and np.isfinite(p_o)):
            continue

        if scenario.basis == "market":
            # 成行: 寄付き価格で約定
            filled[idx] = True
            fill_price[idx] = p_o
            fill_type[idx] = "market"
        else:
            if not np.isfinite(lp_val):
                continue

            if is_long[idx]:
                # ロング買い指値
                if p_l <= lp_val:
                    filled[idx] = True
                    # 寄付き価格がすでに指値以下: 寄付き価格で約定
                    fill_price[idx] = min(lp_val, p_o)
                    fill_type[idx] = "open_gap" if p_o <= lp_val else "intraday"
            elif is_short[idx]:
                # ショート売り指値
                if p_h >= lp_val:
                    filled[idx] = True
                    # 寄付き価格がすでに指値以上: 寄付き価格で約定
                    fill_price[idx] = max(lp_val, p_o)
                    fill_type[idx] = "open_gap" if p_o >= lp_val else "intraday"

    result["filled"] = filled
    result["fill_price"] = fill_price
    result["fill_type"] = fill_type

    return result


# ---------------------------------------------------------------------------
# 5. ポートフォリオリターン計算
# ---------------------------------------------------------------------------

def compute_portfolio_returns(
    fills_df: pd.DataFrame,
    scenario: LimitOrderScenario,
) -> pd.DataFrame:
    """約定銘柄のみでポートフォリオリターンを計算。

    リターン計算:
      ロング: P_close / fill_price - 1
      ショート: fill_price / P_close - 1  ← weight が負なので最終的に符号正

    P_close の取得: r_oc = P_close / P_open - 1 より
      P_close = P_open * (1 + r_oc)

    Returns:
        日次リターン DataFrame (indexed by trade_date)
    """
    slippage_bps = scenario.slippage_bps

    # 銘柄リターン計算
    # fill_price → close return
    p_open = fills_df["P_open"].values
    r_oc = fills_df["r_oc"].values
    p_close = p_open * (1.0 + r_oc)

    fill_price = fills_df["fill_price"].values
    is_long = fills_df["is_long"].values
    is_short = fills_df["is_short"].values

    asset_return = np.zeros(len(fills_df))
    mask_long = fills_df["filled"].values & is_long
    mask_short = fills_df["filled"].values & is_short

    with np.errstate(divide="ignore", invalid="ignore"):
        # ロング: (close - fill) / fill
        asset_return[mask_long] = np.where(
            fill_price[mask_long] > 0,
            p_close[mask_long] / fill_price[mask_long] - 1.0,
            0.0,
        )
        # ショート: (fill - close) / fill = fill/close - 1 に符号を加味
        # ウェイトが負なので: w * (fill/close - 1) = |w| * (1 - close/fill)
        # → asset_return をロングと同じ方向（正 = 利益）で定義し、
        #    weight の符号で最終計算する
        # asset_return[short] = close/fill - 1 (負 = 価格下落 = 利益)
        # → Σ w * r を計算するので w<0, r<0 → 正の貢献
        asset_return[mask_short] = np.where(
            fill_price[mask_short] > 0,
            p_close[mask_short] / fill_price[mask_short] - 1.0,
            0.0,
        )

    fills_df = fills_df.copy()
    fills_df["asset_return"] = asset_return

    # 日次集計
    daily_records = []
    for trade_date, grp in fills_df.groupby("trade_date"):
        filled_mask = grp["filled"].values
        weights_orig = grp["weight"].values
        ar = grp["asset_return"].values
        is_l = grp["is_long"].values
        is_s = grp["is_short"].values

        filled_long = filled_mask & is_l
        filled_short = filled_mask & is_s
        n_long_filled = int(filled_long.sum())
        n_short_filled = int(filled_short.sum())
        n_long_total = int(is_l.sum())
        n_short_total = int(is_s.sum())

        # --- 主系列: 元ウェイト維持版（グロス目減り許容） ---
        w_used = weights_orig.copy()
        w_used[~filled_mask] = 0.0
        gross = float(np.sum(np.abs(w_used)))
        daily_ret_gross = float(np.sum(w_used * ar))

        # スリッページ
        if scenario.basis == "market":
            # 成行: 往復 2 × bps/10000 × gross
            slip = 2.0 * slippage_bps / 10000.0 * gross
        elif scenario.slip_on_entry:
            # 保守: 成行と同じ往復コスト
            slip = 2.0 * slippage_bps / 10000.0 * gross
        else:
            # 指値: エントリー側スリッページ不要（片道のみ）
            slip = 1.0 * slippage_bps / 10000.0 * gross

        daily_ret = daily_ret_gross - slip

        # ロング / ショート 別リターン
        long_ret = float(np.sum(w_used[filled_long] * ar[filled_long]))
        short_ret = float(np.sum(w_used[filled_short] * ar[filled_short]))

        # --- 再正規化版（グロスを2に戻す） ---
        w_renorm = weights_orig.copy()
        w_renorm[~filled_mask] = 0.0
        long_w = w_renorm[filled_long]
        short_w = w_renorm[filled_short]
        if len(long_w) > 0 and np.sum(long_w) > 1e-12:
            w_renorm[filled_long] = long_w / np.sum(long_w)
        if len(short_w) > 0 and np.sum(-short_w) > 1e-12:
            w_renorm[filled_short] = short_w / np.sum(-short_w)
        gross_renorm = float(np.sum(np.abs(w_renorm)))
        daily_ret_renorm_gross = float(np.sum(w_renorm * ar))
        slip_renorm = (
            2.0 * slippage_bps / 10000.0 * gross_renorm
            if (scenario.basis == "market" or scenario.slip_on_entry)
            else 1.0 * slippage_bps / 10000.0 * gross_renorm
        )
        daily_ret_renorm = daily_ret_renorm_gross - slip_renorm

        # ネットエクスポージャー（ドルニュートラル偏差）
        net_exposure = float(np.sum(w_used))
        net_exposure_renorm = float(np.sum(w_renorm))

        daily_records.append({
            "trade_date": trade_date,
            "daily_return": daily_ret,
            "daily_return_gross": daily_ret_gross,
            "daily_return_renorm": daily_ret_renorm,
            "daily_return_renorm_gross": daily_ret_renorm_gross,
            "long_ret": long_ret,
            "short_ret": short_ret,
            "slippage_cost": slip,
            "gross_exposure": gross,
            "gross_exposure_renorm": gross_renorm,
            "net_exposure": net_exposure,
            "net_exposure_renorm": net_exposure_renorm,
            "n_long_filled": n_long_filled,
            "n_short_filled": n_short_filled,
            "n_long_total": n_long_total,
            "n_short_total": n_short_total,
            "n_filled": n_long_filled + n_short_filled,
        })

    df_daily = pd.DataFrame(daily_records).set_index("trade_date")
    return df_daily


# ---------------------------------------------------------------------------
# 6. 分析・診断モジュール
# ---------------------------------------------------------------------------

def compute_fill_stats(fills_df: pd.DataFrame) -> dict:
    """約定率統計。"""
    active = fills_df[fills_df["is_long"] | fills_df["is_short"]]
    filled = active[active["filled"]]
    long_active = fills_df[fills_df["is_long"]]
    short_active = fills_df[fills_df["is_short"]]
    long_filled = long_active[long_active["filled"]]
    short_filled = short_active[short_active["filled"]]

    n_active = len(active)
    n_filled = len(filled)
    n_long = len(long_active)
    n_short = len(short_active)

    # 日別約定本数
    daily_fills = fills_df.groupby("trade_date")["filled"].sum()

    return {
        "fill_rate_total": n_filled / n_active if n_active > 0 else np.nan,
        "fill_rate_long": len(long_filled) / n_long if n_long > 0 else np.nan,
        "fill_rate_short": len(short_filled) / n_short if n_short > 0 else np.nan,
        "daily_fills_mean": float(daily_fills.mean()),
        "daily_fills_std": float(daily_fills.std()),
        "daily_fills_p10": float(daily_fills.quantile(0.10)),
        "daily_fills_p50": float(daily_fills.quantile(0.50)),
        "daily_fills_p90": float(daily_fills.quantile(0.90)),
        "n_active": n_active,
        "n_filled": n_filled,
    }


def compute_neutrality_metrics(
    df_daily: pd.DataFrame,
    fills_df: pd.DataFrame,
    df_exec: pd.DataFrame,
    beta_threshold: float = 0.1,
    net_threshold: float = 0.05,
) -> dict:
    """ドルニュートラル・βニュートラルの歪み分析。"""
    net = df_daily["net_exposure"]

    # jp_beta 抽出
    beta_cols = [c for c in df_exec.columns if c.startswith("jp_beta_")]
    jp_beta_df = df_exec[beta_cols].copy()
    jp_beta_df.columns = [c.replace("jp_beta_", "") for c in beta_cols]

    # ポートフォリオβ = Σ β_j × w_j（約定銘柄のみ）
    filled_df = fills_df[fills_df["filled"]].copy()
    portfolio_betas = {}
    for trade_date, grp in filled_df.groupby("trade_date"):
        if trade_date not in df_exec.index:
            continue
        betas_row = jp_beta_df.loc[trade_date]
        port_beta = 0.0
        for _, row in grp.iterrows():
            tk = row["ticker"]
            b = betas_row.get(tk, 0.0)
            port_beta += float(b) * float(row["weight"])
        portfolio_betas[trade_date] = port_beta

    pb_series = pd.Series(portfolio_betas).sort_index()

    return {
        # ドルニュートラル
        "net_mean": float(net.mean()),
        "net_std": float(net.std()),
        "net_abs_gt5pct_frac": float((net.abs() > net_threshold).mean()),
        # βニュートラル
        "port_beta_mean": float(pb_series.mean()) if len(pb_series) > 0 else np.nan,
        "port_beta_std": float(pb_series.std()) if len(pb_series) > 0 else np.nan,
        "port_beta_abs_gt_thresh_frac": (
            float((pb_series.abs() > beta_threshold).mean()) if len(pb_series) > 0 else np.nan
        ),
        "port_beta_series": pb_series,
    }


def compute_adverse_selection(
    fills_df: pd.DataFrame,
    scenario: LimitOrderScenario,
) -> pd.DataFrame:
    """逆選択診断: 約定 vs 不約定銘柄の成行リターン比較。

    Returns:
        DataFrame with columns: trade_date, group, n, mean_market_return, std_market_return
    """
    if scenario.basis == "market":
        return pd.DataFrame()

    # 成行リターン = P_open_to_close = r_oc
    filled = fills_df[fills_df["filled"]]["r_oc"]
    not_filled_mask = (
        (fills_df["is_long"] | fills_df["is_short"]) & ~fills_df["filled"]
    )
    not_filled = fills_df[not_filled_mask]["r_oc"]

    # 方向加味: ロング銘柄のリターンはそのまま、ショート銘柄は符号反転
    def directional_return(sub_df):
        r = sub_df["r_oc"].values * np.where(sub_df["is_long"].values, 1.0, -1.0)
        return r

    filled_dr = directional_return(fills_df[fills_df["filled"] & (fills_df["is_long"] | fills_df["is_short"])])
    not_filled_active = fills_df[not_filled_mask]
    nf_dr = directional_return(not_filled_active)

    results = []
    for label, arr in [("filled", filled_dr), ("not_filled", nf_dr)]:
        if len(arr) == 0:
            continue
        results.append({
            "group": label,
            "n": len(arr),
            "mean_directional_return": float(np.nanmean(arr)),
            "std_directional_return": float(np.nanstd(arr)),
            "positive_rate": float(np.nanmean(arr > 0)),
        })
    return pd.DataFrame(results)


def compute_performance(
    df_daily: pd.DataFrame,
    col: str = "daily_return",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict:
    """既存 performance.py を利用してパフォーマンス指標を計算。"""
    series = df_daily[col]
    if start:
        series = series[series.index >= pd.to_datetime(start)]
    if end:
        series = series[series.index < pd.to_datetime(end)]
    if len(series) == 0:
        return {}
    m = calculate_metrics(series)
    return m


# ---------------------------------------------------------------------------
# 7. 全シナリオ実行
# ---------------------------------------------------------------------------

def run_all_scenarios(
    df_signals: pd.DataFrame,
    ohlc: dict[str, np.ndarray],
    df_exec: pd.DataFrame,
    oos_start: str = OOS_DEFAULT_START,
) -> list[dict]:
    """全シナリオ（ベースライン + 主系列 + 対照群）を実行して結果リストを返す。"""

    # シナリオリスト構築
    scenarios: list[LimitOrderScenario] = []

    # ベースライン（成行）
    scenarios.append(LimitOrderScenario(basis="market", margin_bps=0, slip_on_entry=True))

    # 主系列: theory basis × margin × renorm × slip
    for m in THEORY_MARGINS_BPS:
        for renorm in [False, True]:
            for slip in [False, True]:
                scenarios.append(
                    LimitOrderScenario(
                        basis="theory", margin_bps=m,
                        renormalize=renorm, slip_on_entry=slip,
                    )
                )

    # 対照群: prev_close basis × offset × renorm × slip
    for k in PREV_CLOSE_OFFSETS_BPS:
        for renorm in [False, True]:
            for slip in [False, True]:
                scenarios.append(
                    LimitOrderScenario(
                        basis="prev_close", margin_bps=k,
                        renormalize=renorm, slip_on_entry=slip,
                    )
                )

    all_results = []
    total = len(scenarios)
    for i_sc, scenario in enumerate(scenarios):
        print(f"  [{i_sc + 1}/{total}] {scenario.label} ...", end=" ", flush=True)

        limit_prices = compute_limit_prices(df_signals, scenario)
        fills_df = evaluate_fills(df_signals, limit_prices, ohlc, df_exec, scenario)
        df_daily = compute_portfolio_returns(fills_df, scenario)

        # 統計
        fill_stats = compute_fill_stats(fills_df)
        neutrality = compute_neutrality_metrics(fills_df=fills_df, df_daily=df_daily, df_exec=df_exec)
        adverse = compute_adverse_selection(fills_df, scenario)

        # パフォーマンス（全期間 / OOS / OOS除外）
        perf_full = compute_performance(df_daily, "daily_return")
        perf_full_renorm = compute_performance(df_daily, "daily_return_renorm")
        perf_oos = compute_performance(df_daily, "daily_return", start=oos_start)
        perf_oos_renorm = compute_performance(df_daily, "daily_return_renorm", start=oos_start)

        all_results.append({
            "scenario": scenario,
            "fills_df": fills_df,
            "df_daily": df_daily,
            "fill_stats": fill_stats,
            "neutrality": neutrality,
            "adverse_selection": adverse,
            "perf_full": perf_full,
            "perf_full_renorm": perf_full_renorm,
            "perf_oos": perf_oos,
            "perf_oos_renorm": perf_oos_renorm,
        })

        print(
            f"AR={perf_full.get('AR', np.nan)*100:.2f}% "
            f"MDD={perf_full.get('MDD', np.nan)*100:.2f}% "
            f"fill={fill_stats['fill_rate_total']*100:.1f}%"
        )

    return all_results


# ---------------------------------------------------------------------------
# 8. レポート生成
# ---------------------------------------------------------------------------

def _perf_row(label: str, basis: str, margin_bps: float, renorm: bool, slip: bool,
              fill_stats: dict, neutrality: dict, perf: dict, perf_renorm: dict,
              period: str) -> dict:
    """サマリ表の1行を作成。"""
    return {
        "scenario": label,
        "basis": basis,
        "margin_bps": margin_bps,
        "renormalize": renorm,
        "slip_on_entry": slip,
        "period": period,
        # 約定率
        "fill_rate_total_%": round(fill_stats.get("fill_rate_total", np.nan) * 100, 1),
        "fill_rate_long_%": round(fill_stats.get("fill_rate_long", np.nan) * 100, 1),
        "fill_rate_short_%": round(fill_stats.get("fill_rate_short", np.nan) * 100, 1),
        "daily_fills_mean": round(fill_stats.get("daily_fills_mean", np.nan), 2),
        # パフォーマンス（元ウェイト版）
        "AR_%": round(perf.get("AR", np.nan) * 100, 2),
        "RISK_%": round(perf.get("RISK", np.nan) * 100, 2),
        "RR": round(perf.get("R/R", np.nan), 3),
        "MDD_%": round(perf.get("MDD", np.nan) * 100, 2),
        "Sharpe": round(perf.get("Sharpe", np.nan), 3),
        # パフォーマンス（再正規化版）
        "AR_renorm_%": round(perf_renorm.get("AR", np.nan) * 100, 2),
        "RISK_renorm_%": round(perf_renorm.get("RISK", np.nan) * 100, 2),
        "RR_renorm": round(perf_renorm.get("R/R", np.nan), 3),
        "MDD_renorm_%": round(perf_renorm.get("MDD", np.nan) * 100, 2),
        # 中立性
        "net_mean": round(neutrality.get("net_mean", np.nan), 4),
        "net_std": round(neutrality.get("net_std", np.nan), 4),
        "net_abs_gt5pct_%": round(neutrality.get("net_abs_gt5pct_frac", np.nan) * 100, 1),
        "port_beta_mean": round(neutrality.get("port_beta_mean", np.nan), 4),
        "port_beta_std": round(neutrality.get("port_beta_std", np.nan), 4),
    }


def generate_summary_tables(all_results: list[dict], output_dir: Path) -> None:
    """基準×水準のサマリ表を CSV と Markdown で出力。"""
    rows_full = []
    rows_oos = []

    for res in all_results:
        sc = res["scenario"]
        for period, perf_key, perf_renorm_key, rows in [
            ("full", "perf_full", "perf_full_renorm", rows_full),
            ("oos", "perf_oos", "perf_oos_renorm", rows_oos),
        ]:
            rows.append(_perf_row(
                label=sc.label,
                basis=sc.basis,
                margin_bps=sc.margin_bps,
                renorm=sc.renormalize,
                slip=sc.slip_on_entry,
                fill_stats=res["fill_stats"],
                neutrality=res["neutrality"],
                perf=res[perf_key],
                perf_renorm=res[perf_renorm_key],
                period=period,
            ))

    for period, rows in [("full", rows_full), ("oos", rows_oos)]:
        df = pd.DataFrame(rows)
        csv_path = output_dir / f"summary_{period}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info("サマリ CSV 出力: %s", csv_path)

        # ベースライン優先・basis 別のサブセット
        for basis in ["market", "theory", "prev_close"]:
            sub = df[df["basis"] == basis]
            if sub.empty:
                continue
            sub_csv = output_dir / f"summary_{period}_{basis}.csv"
            sub.to_csv(sub_csv, index=False, encoding="utf-8-sig")

        # Markdown サマリ（renorm=False, slip=True のみの主要指標）
        md_sub = df[(df["renormalize"] == False) & (df["slip_on_entry"] == True)].copy()
        key_cols = [
            "scenario", "basis", "margin_bps",
            "fill_rate_total_%", "daily_fills_mean",
            "AR_%", "RISK_%", "RR", "MDD_%",
            "net_abs_gt5pct_%", "port_beta_mean",
        ]
        available = [c for c in key_cols if c in md_sub.columns]
        md_path = output_dir / f"summary_{period}_main.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# 指値エントリーバックテスト サマリ ({period.upper()})\n\n")
            f.write(md_sub[available].to_markdown(index=False))
            f.write("\n")
        logger.info("Markdown サマリ出力: %s", md_path)


def generate_plots(all_results: list[dict], output_dir: Path) -> None:
    """ヒストグラム・推移グラフを生成。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        logger.warning("matplotlib が見つかりません。グラフ生成をスキップ。")
        return

    plt.rcParams["font.family"] = "IPAGothic" if os.name != "nt" else "MS Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    # --- (a) 約定本数ヒストグラム（主要シナリオのみ） ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    theory_results = [
        r for r in all_results
        if r["scenario"].basis == "theory"
        and not r["scenario"].renormalize
        and r["scenario"].slip_on_entry
    ]
    for i, res in enumerate(theory_results[:6]):
        ax = axes[i]
        daily_n = res["df_daily"]["n_filled"]
        ax.hist(daily_n.values, bins=range(0, 12), align="left", color="#4e79a7", edgecolor="white", alpha=0.8)
        ax.set_title(f"theory m={res['scenario'].margin_bps:.0f}bps\n"
                     f"fill={res['fill_stats']['fill_rate_total']*100:.1f}%", fontsize=9)
        ax.set_xlabel("日次約定本数")
        ax.set_ylabel("日数")
        ax.set_xticks(range(0, 11))
    fig.suptitle("1日あたり約定本数分布（主系列）", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "fill_count_histogram_theory.png", dpi=120)
    plt.close()

    # 対照群
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    pc_results = [
        r for r in all_results
        if r["scenario"].basis == "prev_close"
        and not r["scenario"].renormalize
        and r["scenario"].slip_on_entry
    ]
    for i, res in enumerate(pc_results[:6]):
        ax = axes[i]
        daily_n = res["df_daily"]["n_filled"]
        ax.hist(daily_n.values, bins=range(0, 12), align="left", color="#f28e2b", edgecolor="white", alpha=0.8)
        ax.set_title(f"prev_close k={res['scenario'].margin_bps:+.0f}bps\n"
                     f"fill={res['fill_stats']['fill_rate_total']*100:.1f}%", fontsize=9)
        ax.set_xlabel("日次約定本数")
        ax.set_ylabel("日数")
        ax.set_xticks(range(0, 11))
    fig.suptitle("1日あたり約定本数分布（対照群）", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "fill_count_histogram_prev_close.png", dpi=120)
    plt.close()

    # --- (b) ネットエクスポージャー分布 ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    baseline = next(r for r in all_results if r["scenario"].basis == "market")
    axes[0].hist(
        baseline["df_daily"]["net_exposure"].values * 100,
        bins=40, color="#59a14f", edgecolor="white", alpha=0.8
    )
    axes[0].axvline(x=-5, color="red", linestyle="--", label="±5%")
    axes[0].axvline(x=5, color="red", linestyle="--")
    axes[0].set_title("ネットエクスポージャー分布（成行ベースライン）")
    axes[0].set_xlabel("Net Exposure (%)")
    axes[0].legend()

    # 主系列 m=20 と比較
    theory_20 = next(
        (r for r in all_results if r["scenario"].basis == "theory"
         and r["scenario"].margin_bps == 20 and not r["scenario"].renormalize
         and r["scenario"].slip_on_entry),
        None,
    )
    if theory_20:
        axes[1].hist(
            theory_20["df_daily"]["net_exposure"].values * 100,
            bins=40, color="#4e79a7", edgecolor="white", alpha=0.8
        )
        axes[1].axvline(x=-5, color="red", linestyle="--", label="±5%")
        axes[1].axvline(x=5, color="red", linestyle="--")
        axes[1].set_title("ネットエクスポージャー分布（theory m=20bps）")
        axes[1].set_xlabel("Net Exposure (%)")
        axes[1].legend()
    fig.tight_layout()
    plt.savefig(output_dir / "net_exposure_histogram.png", dpi=120)
    plt.close()

    # --- (c) AR / MDD / 約定率の水準別推移グラフ ---
    theory_main = [
        r for r in all_results
        if r["scenario"].basis == "theory"
        and not r["scenario"].renormalize
        and r["scenario"].slip_on_entry
    ]
    pc_main = [
        r for r in all_results
        if r["scenario"].basis == "prev_close"
        and not r["scenario"].renormalize
        and r["scenario"].slip_on_entry
    ]
    baseline_result = next(r for r in all_results if r["scenario"].basis == "market")

    def _extract_series(results_list, metric_fn):
        xs = [r["scenario"].margin_bps for r in results_list]
        ys = [metric_fn(r) for r in results_list]
        return xs, ys

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)
    metrics_info = [
        ("AR (%)", lambda r: r["perf_full"].get("AR", np.nan) * 100,
         lambda r: r["perf_full"].get("AR", np.nan) * 100),
        ("MDD (%)", lambda r: r["perf_full"].get("MDD", np.nan) * 100,
         lambda r: r["perf_full"].get("MDD", np.nan) * 100),
        ("約定率 (%)", lambda r: r["fill_stats"].get("fill_rate_total", np.nan) * 100,
         lambda r: r["fill_stats"].get("fill_rate_total", np.nan) * 100),
    ]
    for ax, (ylabel, th_fn, pc_fn) in zip(axes, metrics_info):
        xs_th, ys_th = _extract_series(theory_main, th_fn)
        xs_pc, ys_pc = _extract_series(pc_main, pc_fn)
        baseline_val = th_fn(baseline_result)
        ax.axhline(baseline_val, color="gray", linestyle="--", label="成行ベースライン", linewidth=1.5)
        ax.plot(xs_th, ys_th, "o-", color="#4e79a7", label="主系列(theory)", linewidth=2, markersize=6)
        ax.plot(xs_pc, ys_pc, "s--", color="#f28e2b", label="対照群(prev_close)", linewidth=2, markersize=6)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if ylabel == "AR (%)":
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(xs_pc)
            ax2.set_xticklabels([f"{k:+.0f}" for k in xs_pc], fontsize=7, color="#f28e2b")
            ax2.set_xlabel("k (prev_close bps)", color="#f28e2b", fontsize=8)

    axes[0].set_title("指値エントリー: 水準別パフォーマンス推移 (全期間, 元ウェイト版, slip=True)")
    axes[-1].set_xlabel("m / k (bps)")
    fig.tight_layout()
    plt.savefig(output_dir / "performance_vs_margin.png", dpi=120)
    plt.close()

    # --- (d) ポートフォリオβ分布 ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, res_label in [
        (axes[0], "baseline_market"),
        (axes[1], None),
    ]:
        if res_label:
            res = next((r for r in all_results if r["scenario"].label == res_label), None)
        else:
            res = theory_20
        if res is None:
            continue
        pb = res["neutrality"].get("port_beta_series", pd.Series())
        if len(pb) == 0:
            continue
        ax.hist(pb.values, bins=40, edgecolor="white", alpha=0.8)
        ax.axvline(x=0.1, color="red", linestyle="--", label="±0.1", linewidth=1)
        ax.axvline(x=-0.1, color="red", linestyle="--", linewidth=1)
        title = res["scenario"].label
        ax.set_title(f"ポートフォリオβ分布\n{title}")
        ax.set_xlabel("Portfolio Beta (vs TOPIX)")
        ax.legend()
    fig.tight_layout()
    plt.savefig(output_dir / "portfolio_beta_histogram.png", dpi=120)
    plt.close()

    # --- (e) 逆選択診断 ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    theory_configs = [
        (0, "m=0bps"),
        (20, "m=20bps"),
        (50, "m=50bps"),
    ]
    for ax, (m_bps, label) in zip(axes, theory_configs):
        res = next(
            (r for r in all_results
             if r["scenario"].basis == "theory"
             and r["scenario"].margin_bps == m_bps
             and not r["scenario"].renormalize
             and r["scenario"].slip_on_entry),
            None,
        )
        if res is None or res["adverse_selection"].empty:
            ax.set_visible(False)
            continue
        adv = res["adverse_selection"]
        groups = adv["group"].tolist()
        means = adv["mean_directional_return"].tolist()
        colors = ["#4e79a7" if g == "filled" else "#e15759" for g in groups]
        bars = ax.bar(groups, [m * 100 for m in means], color=colors, alpha=0.8)
        ax.set_title(f"逆選択診断 (theory {label})")
        ax.set_ylabel("平均方向性リターン (%)")
        ax.axhline(0, color="black", linewidth=0.8)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{m*100:.3f}%", ha="center", va="bottom", fontsize=9)
    fig.suptitle("逆選択診断: 約定銘柄 vs 不約定銘柄の方向性リターン")
    fig.tight_layout()
    plt.savefig(output_dir / "adverse_selection.png", dpi=120)
    plt.close()

    # --- (f) 累積リターン比較（主要シナリオ） ---
    fig, ax = plt.subplots(figsize=(12, 6))
    # ベースライン
    baseline_daily = baseline_result["df_daily"]["daily_return"]
    (1 + baseline_daily).cumprod().plot(ax=ax, label="成行ベースライン", color="gray", linewidth=2)
    colors_th = ["#cce5ff", "#99caff", "#4e79a7", "#2b5596", "#163561", "#0a1e3f"]
    for ci, res in enumerate(theory_main):
        m = res["scenario"].margin_bps
        dr = res["df_daily"]["daily_return"]
        (1 + dr).cumprod().plot(
            ax=ax, label=f"theory m={m:.0f}bps", color=colors_th[ci], linewidth=1.5, alpha=0.9
        )
    ax.set_title("累積リターン比較（主系列: theory basis）")
    ax.set_ylabel("累積資産倍率")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_return_theory.png", dpi=120)
    plt.close()

    logger.info("グラフ出力完了: %s", output_dir)


def generate_assumption_memo(
    output_dir: Path,
    config: StrategyConfig,
    start_date: str,
    oos_start: str,
) -> None:
    """約定モデル・仮定・r_hat 定義のメモを Markdown で出力。"""
    memo = f"""# 指値エントリーバックテスト: 仮定・モデル定義メモ

生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. r_hat_cc の定義

`compute_signal()` が返す `r_hat_jp_cc` を使用:

```
r_hat_jp_cc = mu_jp + sigma_jp * z_hat_j_t1
```

- `mu_jp`: ウィンドウ内の JP 銘柄 CC リターンの EWMA 平均
- `sigma_jp`: 同 EWMA 標準偏差  
- `z_hat_j_t1`: PCA 予測標準化リターン（V_J^K × f_t）

これはギャップ補正前の**PCA 予測 Close-to-Close リターン**（§4.6 ステップ3）。

最終シグナル `s_j`（`gap_residual` モード）はギャップを差し引いた残差であり `s_j ≠ r_hat_cc`。

フェア価格の算出: `P_fair[j] = P_close[t-1, j] × (1 + r_hat_cc[j])`

## 2. 指値価格の定義

### 主系列 (theory basis)
- フェア価格: `P_fair = P_close_prev × (1 + r_hat_cc)`
- ロング指値: `P_fair × (1 - m/10000)` （m bps 安く買う）
- ショート指値: `P_fair × (1 + m/10000)` （m bps 高く売る）
- m ∈ {THEORY_MARGINS_BPS}

### 対照群 (prev_close basis)
- ロング指値: `P_close_prev × (1 - k/10000)`
- ショート指値: `P_close_prev × (1 + k/10000)`  
- k ∈ {PREV_CLOSE_OFFSETS_BPS}（負=不利側で約定しやすい, 正=有利側）

## 3. 約定モデル

使用データ: 当日の日足 High (P_high) / Low (P_low)

| 方向 | 約定条件 | 約定価格 |
|------|----------|----------|
| ロング(買い) | `P_low ≤ 指値` | `min(指値, P_open)`（寄付き時点で有利なら P_open） |
| ショート(売り) | `P_high ≥ 指値` | `max(指値, P_open)`（寄付き時点で有利なら P_open） |

**限界・注意点:**
- 日足 High/Low は日中いずれかの時点での最高値/最安値。到達タイミング不明。
- 指値が日中の High/Low に到達している場合でも、実際の執行は指値価格で可能とは限らない（板薄の場合）。
- 本モデルは保守的ではなく「理論的に可能な最良ケース」に近い。過楽観バイアスに注意。

## 4. スリッページ仮定

| バリアント | エントリー側 | 決済側 | 往復コスト |
|-----------|-------------|--------|-----------|
| slip_on_entry=True（保守） | 5bps | 5bps | 10bps × gross |
| slip_on_entry=False（指値） | 0bps | 5bps | 5bps × gross |
| ベースライン(成行) | 5bps | 5bps | 10bps × gross |

## 5. ウェイト処理

- **主系列 (renormalize=False)**: 元の目標ウェイト `w_j` をそのまま使用。約定しなかった銘柄は0。グロスエクスポージャーが目減りする。
- **再正規化版 (renormalize=True)**: 約定銘柄のウェイトを正規化し、ロング合計=+1, ショート合計=-1 に戻す。グロスは2を維持。

## 6. 戦略設定（既存ロジックと同一）

| パラメータ | 値 |
|-----------|-----|
| signal_mode | {config.signal_mode} |
| gap_open_coef | {config.gap_open_coef} |
| topix_beta_coef | {config.topix_beta_coef} |
| weight_mode | {config.weight_mode} |
| dispersion_filter | {config.dispersion_filter} |
| q | {config.q} |
| K | {config.k} |
| corr_window | {config.corr_window} |
| ewma_half_life | {config.ewma_half_life} |
| lambda_reg | {config.lambda_reg} |
| lambda_lw | {config.lambda_lw} |
| slippage_bps | {config.slippage_bps} |

## 7. バックテスト期間

- 全期間: {start_date} 〜 データ末尾
- OOS 期間: {oos_start} 〜 データ末尾

## 8. ルックアヘッドバイアス対策

- [x] 指値価格: `P_close[t-1]`（前日終値）と `r_hat_cc`（前日の US 情報で計算）のみ
- [x] 約定判定: 当日の `P_high`, `P_low` のみ使用
- [x] ウェイト: 前日シグナルで決定（既存ロジックと同一）
- [x] 決済: 当日終値 = `P_open × (1 + r_oc)`（r_oc = Close/Open - 1）
"""

    memo = memo.replace("{THEORY_MARGINS_BPS}", str(THEORY_MARGINS_BPS))
    memo = memo.replace("{PREV_CLOSE_OFFSETS_BPS}", str(PREV_CLOSE_OFFSETS_BPS))

    memo_path = output_dir / "assumptions_memo.md"
    with open(memo_path, "w", encoding="utf-8") as f:
        f.write(memo)
    logger.info("仮定メモ出力: %s", memo_path)


def save_daily_results(all_results: list[dict], output_dir: Path) -> None:
    """各シナリオの日次リターン CSV を保存（主要シナリオのみ）。"""
    # ベースライン + 主系列 + 対照群 の renorm=False, slip=True のみ
    for res in all_results:
        sc = res["scenario"]
        if sc.renormalize or not sc.slip_on_entry:
            continue
        fname = f"daily_{sc.label}.csv"
        res["df_daily"].to_csv(output_dir / fname, encoding="utf-8-sig")


def save_adverse_selection(all_results: list[dict], output_dir: Path) -> None:
    """逆選択診断の結果を CSV で保存。"""
    rows = []
    for res in all_results:
        sc = res["scenario"]
        adv = res["adverse_selection"]
        if adv.empty:
            continue
        adv = adv.copy()
        adv["scenario"] = sc.label
        adv["basis"] = sc.basis
        adv["margin_bps"] = sc.margin_bps
        rows.append(adv)
    if rows:
        df = pd.concat(rows, ignore_index=True)
        df.to_csv(output_dir / "adverse_selection.csv", index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# 9. メイン
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="指値エントリーバックテスト検証")
    parser.add_argument("--start-date", default="2015-01-01", help="バックテスト開始日")
    parser.add_argument("--oos-start-date", default=OOS_DEFAULT_START, help="OOS 開始日")
    parser.add_argument("--force-refresh-highlow", action="store_true",
                        help="High/Low キャッシュを強制更新")
    args = parser.parse_args()

    print("=" * 60)
    print("指値エントリーバックテスト検証")
    print("=" * 60)

    # --- 出力ディレクトリ ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = _RESULTS_DIR / f"{ts}_limit_order_backtest"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"出力ディレクトリ: {output_dir}")

    # --- 1. データロード ---
    print("\n[1/5] データロード中...")
    df_exec = load_df_exec()
    print(f"  df_exec: {len(df_exec)} 日 ({df_exec.index[0].date()} 〜 {df_exec.index[-1].date()})")

    highlow = download_jp_highlow(
        cache_path=_HIGHLOW_CACHE_PATH,
        force_refresh=args.force_refresh_highlow,
    )
    ohlc = build_ohlc_arrays(df_exec, highlow)
    print(f"  High/Low: {ohlc['jp_high'].shape} (T x N_J)")

    # nan 状況確認
    nan_rate_h = np.isnan(ohlc["jp_high"]).mean()
    nan_rate_l = np.isnan(ohlc["jp_low"]).mean()
    print(f"  NaN率 High={nan_rate_h*100:.1f}%, Low={nan_rate_l*100:.1f}%")

    # --- 2. 戦略設定 ---
    config = StrategyConfig(
        k=STRATEGY_DEFAULTS["K"],
        lambda_reg=STRATEGY_DEFAULTS["lambda_reg"],
        q=STRATEGY_DEFAULTS["q"],
        weight_mode=STRATEGY_DEFAULTS["weight_mode"],
        dispersion_filter=STRATEGY_DEFAULTS["dispersion_filter"],
        dispersion_metric=STRATEGY_DEFAULTS.get("dispersion_metric", "long_short_mean_gap"),
        v3_mode=STRATEGY_DEFAULTS["v3_mode"],
        ewma_half_life=STRATEGY_DEFAULTS["ewma_half_life"],
        lambda_lw=STRATEGY_DEFAULTS["lambda_lw"],
        lw_target=STRATEGY_DEFAULTS["lw_target"],
        corr_window=STRATEGY_DEFAULTS["corr_window"],
        include_v4_prior=STRATEGY_DEFAULTS["include_v4_prior"],
        signal_mode=STRATEGY_DEFAULTS["signal_mode"],
        gap_open_coef=STRATEGY_DEFAULTS["gap_open_coef"],
        topix_beta_coef=STRATEGY_DEFAULTS.get("topix_beta_coef", 0.6),
        beta_window=STRATEGY_DEFAULTS.get("beta_window", 60),
        gamma=STRATEGY_DEFAULTS.get("gamma", 0.5),
        slippage_bps=STRATEGY_DEFAULTS.get("slippage_bps", DEFAULT_SLIPPAGE_BPS),
    )

    # --- 3. シグナル抽出 ---
    print(f"\n[2/5] シグナル抽出中 (start={args.start_date})...")
    df_signals = run_signal_extraction(df_exec, config, start_date=args.start_date)
    print(f"  シグナル行数: {len(df_signals)} ({df_signals['trade_date'].nunique()} 日 × {N_JP_ASSETS} 銘柄)")

    # --- 4. 全シナリオ実行 ---
    print("\n[3/5] 全シナリオ実行中...")
    all_results = run_all_scenarios(
        df_signals, ohlc, df_exec, oos_start=args.oos_start_date
    )

    # --- 5. レポート生成 ---
    print("\n[4/5] レポート生成中...")
    generate_summary_tables(all_results, output_dir)
    generate_plots(all_results, output_dir)
    generate_assumption_memo(output_dir, config, args.start_date, args.oos_start_date)
    save_daily_results(all_results, output_dir)
    save_adverse_selection(all_results, output_dir)

    # --- 6. マニフェスト ---
    manifest = {
        "run_timestamp": ts,
        "start_date": args.start_date,
        "oos_start_date": args.oos_start_date,
        "signal_mode": config.signal_mode,
        "gap_open_coef": config.gap_open_coef,
        "slippage_bps": float(config.slippage_bps),
        "theory_margins_bps": THEORY_MARGINS_BPS,
        "prev_close_offsets_bps": PREV_CLOSE_OFFSETS_BPS,
        "n_scenarios": len(all_results),
        "df_exec_rows": len(df_exec),
        "df_signals_rows": len(df_signals),
        "highlow_nan_rate_high": float(nan_rate_h),
        "highlow_nan_rate_low": float(nan_rate_l),
        "output_dir": str(output_dir),
    }
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # --- 7. サマリ出力 ---
    print("\n[5/5] 実行完了")
    print(f"\n{'='*60}")
    print("ベースライン vs 主要シナリオ (全期間, 元ウェイト版, slip=True)")
    print(f"{'='*60}")
    header = f"{'シナリオ':<35} {'AR%':>7} {'RISK%':>7} {'RR':>6} {'MDD%':>7} {'約定率%':>8} {'net>5%':>7}"
    print(header)
    print("-" * len(header))

    for res in all_results:
        sc = res["scenario"]
        if sc.renormalize or not sc.slip_on_entry:
            continue
        perf = res["perf_full"]
        fs = res["fill_stats"]
        nt = res["neutrality"]
        print(
            f"{sc.label:<35} "
            f"{perf.get('AR', np.nan)*100:>7.2f} "
            f"{perf.get('RISK', np.nan)*100:>7.2f} "
            f"{perf.get('R/R', np.nan):>6.3f} "
            f"{perf.get('MDD', np.nan)*100:>7.2f} "
            f"{fs.get('fill_rate_total', np.nan)*100:>8.1f} "
            f"{nt.get('net_abs_gt5pct_frac', np.nan)*100:>7.1f}"
        )

    print(f"\n出力ディレクトリ: {output_dir}")


if __name__ == "__main__":
    main()
