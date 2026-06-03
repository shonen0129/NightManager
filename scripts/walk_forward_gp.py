"""walk_forward_gp.py – GP 不確実性推定モジュール Walk-Forward バックテスト

概要
----
既存線形戦略（ベンチマーク）に対し、GP の予測分散を用いた確信度ベースの
サイジング（GP-Sizing）を walk-forward 形式で評価し、採用判定を出力する。

実行例
------
# デフォルト設定で実行
python scripts/walk_forward_gp.py

# 設定ファイルを指定
python scripts/walk_forward_gp.py --config configs/gp_uncertainty.yaml

# 高速テスト（小さい窓）
python scripts/walk_forward_gp.py --window 120 --oos-start 2022-01-01

比較指標
--------
- コスト控除後ネット R/R、年率リターン、最大ドローダウン
- レジーム別（高ボラ/低ボラ）パフォーマンス分解
- 確信度分位別（低/中/高 κ_t）リターン分解
- GP 不確実性較正テスト（coverage@80/90/95%）
- ARD 長さスケール可視化（どのファクターが非線形性に寄与するか）
- κ_t 時系列チャート
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── パス設定 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
SRC_DIR = SCRIPT_DIR.parent / "src"
CONFIG_DIR = SCRIPT_DIR.parent / "configs"
sys.path.insert(0, str(SRC_DIR))

# ── 既存インフラ ──────────────────────────────────────────────────────────────
from config import STRATEGY_DEFAULTS
from config import N_JP_ASSETS, N_US_ASSETS
from data_loader import download_data, preprocess_data
from domain.correction.evaluation import (
    CostModel,
    PerformanceMetrics,
    compute_net_returns,
    compute_performance_metrics,
)
from domain.correction.time_series_cv import TimeSeriesPurgeSplit, audit_no_leak
from domain.gp import (
    CalibrationResult,
    ConfidenceConfig,
    ConfidenceScorer,
    GPConfig,
    GPUncertaintyModule,
    coverage_test,
    coverage_test_by_sector,
    reliability_diagram,
)
from domain.gp.confidence import quantile_return_decomposition, regime_kappa_decomposition
from domain.models.types import StrategyConfig
from domain.signals import lead_lag as signals

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 採用判定（GPサイジング専用）
# ---------------------------------------------------------------------------


def evaluate_gp_adoption(
    linear_metrics: PerformanceMetrics,
    gp_metrics: PerformanceMetrics,
    calibration: CalibrationResult,
    ar_tolerance: float = 0.95,
    significance_level: float = 0.05,
) -> Dict:
    """GP-Sizing モジュールの採用判定。

    ADOPT 条件（すべて満たす必要あり）:
      1. OOS ネット R/R (GP) > ネット R/R (線形)
      2. OOS 年率 AR (GP) >= AR (線形) × ar_tolerance
         （DD 低減が主目的なので ±5% 許容）
      3. 80% 予測区間カバレッジが較正良好（GOOD または WARN 警告のみ）
      4. GP の MDD < 線形の MDD（高ボラ局面でのドローダウン低減を確認）

    Returns
    -------
    dict with keys:
        "decision": "ADOPT" or "REJECT"
        "reason": str
        "failures": list of str
    """
    failures = []

    # 条件 1: R/R 改善
    rr_lin = linear_metrics.rr
    rr_gp = gp_metrics.rr
    if not (np.isfinite(rr_gp) and np.isfinite(rr_lin) and rr_gp > rr_lin):
        failures.append(
            f"ネット R/R 未改善: GP={rr_gp:.3f} vs 線形={rr_lin:.3f}"
        )

    # 条件 2: AR 許容範囲
    ar_lin = linear_metrics.ar
    ar_gp = gp_metrics.ar
    threshold_ar = ar_lin * ar_tolerance
    if not (np.isfinite(ar_gp) and np.isfinite(ar_lin) and ar_gp >= threshold_ar):
        failures.append(
            f"年率リターン不足: GP={ar_gp*100:.2f}% < 線形 × {ar_tolerance:.2f} "
            f"= {threshold_ar*100:.2f}%"
        )

    # 条件 3: 較正品質
    if calibration.calibration_status == "REJECT":
        failures.append(
            f"較正不良 (REJECT): GP 予測分散を確信度として信頼できません。"
            f"最大カバレッジ誤差: {max(abs(e) for e in calibration.coverage_errors)*100:.1f}pp"
        )

    # 条件 4: MDD 改善（主目的）
    mdd_lin = linear_metrics.mdd
    mdd_gp = gp_metrics.mdd
    if np.isfinite(mdd_lin) and np.isfinite(mdd_gp) and mdd_gp <= mdd_lin:
        pass  # MDD 改善 OK (mdd は負なので <= が改善)
    elif np.isfinite(mdd_lin) and np.isfinite(mdd_gp):
        failures.append(
            f"MDD 未改善: GP={mdd_gp*100:.2f}% vs 線形={mdd_lin*100:.2f}% "
            f"(ドローダウン低減が GP-Sizing の主目的)"
        )

    if failures:
        reason = (
            "REJECT – GP-Sizing は OOS・コスト控除後でベンチマーク（線形）を上回りません。\n"
            "失敗条件:\n" + "\n".join(f"  • {f}" for f in failures)
        )
        decision = "REJECT"
    else:
        reason = (
            f"ADOPT – GP-Sizing は全採用条件を満たしています:\n"
            f"  • ネット R/R: {rr_gp:.3f} > {rr_lin:.3f}\n"
            f"  • 年率 AR: {ar_gp*100:.2f}% ≥ {threshold_ar*100:.2f}%\n"
            f"  • 較正: {calibration.calibration_status}\n"
            f"  • MDD: {mdd_gp*100:.2f}% ≤ {mdd_lin*100:.2f}%"
        )
        decision = "ADOPT"

    return {"decision": decision, "reason": reason, "failures": failures}


# ---------------------------------------------------------------------------
# Walk-Forward メインループ
# ---------------------------------------------------------------------------


def run_walk_forward_gp(
    df_exec: pd.DataFrame,
    gp_cfg: GPConfig,
    conf_cfg: ConfidenceConfig,
    strategy_cfg: StrategyConfig,
    cost_model: CostModel,
    start_date: str = "2015-01-01",
    oos_start_date: str = "2020-01-01",
    gap_purge: int = 1,
    embargo: int = 5,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Walk-Forward 形式で GP-Sizing を評価する。

    Returns
    -------
    dict with keys:
        "oos_returns_linear"  : pd.Series
        "oos_returns_gp"      : pd.Series
        "kappa_series"        : pd.Series
        "mu_panel"            : ndarray (T_oos, N_J)
        "sigma2_panel"        : ndarray (T_oos, N_J)
        "y_oos_panel"         : ndarray (T_oos, N_J)
        "signal_oos_panel"    : ndarray (T_oos, N_J)
        "ard_history"         : list of dict
    """
    N_J = N_JP_ASSETS
    N_U = N_US_ASSETS

    all_cc_cols = [
        c for c in df_exec.columns if c.startswith("us_cc_") or c.startswith("jp_cc_")
    ]
    jp_oc_cols = [c for c in df_exec.columns if c.startswith("jp_oc_")]
    gap_cols = [c for c in df_exec.columns if c.startswith("jp_gap_")]
    beta_cols = [c for c in df_exec.columns if c.startswith("jp_beta_")]

    all_returns = df_exec[all_cc_cols].values
    date_index = df_exec.index
    jp_oc = df_exec[jp_oc_cols].values if jp_oc_cols else np.zeros((len(df_exec), N_J))
    jp_gap = df_exec[gap_cols].values if len(gap_cols) == N_J else None
    jp_beta = df_exec[beta_cols].values if len(beta_cols) == N_J else None
    topix_night = (
        df_exec["topix_night_return"].values
        if "topix_night_return" in df_exec.columns
        else None
    )

    # 基準相関行列・事前サブスペース（既存パイプライン流用）
    c_full = signals.compute_baseline_correlation(all_returns, date_index.values, strategy_cfg.ewma_half_life)
    v0_static = signals.build_v3_static(N_U, N_J, strategy_cfg.include_v4_prior)
    base_vectors = signals.build_base_vectors(N_U, N_J)
    v1, v2 = base_vectors["v1"], base_vectors["v2"]

    start_idx = max(
        df_exec.index.searchsorted(pd.to_datetime(start_date)),
        strategy_cfg.corr_window + 1,
    )
    oos_start_idx = df_exec.index.searchsorted(pd.to_datetime(oos_start_date))

    logger.info(
        "Walk-Forward 設定: start_idx=%d, oos_start_idx=%d, total=%d",
        start_idx, oos_start_idx, len(df_exec),
    )

    # ── 全期間のシグナルと f_t を事前計算 ─────────────────────────────────
    logger.info("全期間のシグナル・ファクタースコアを事前計算中...")
    all_f_t = []
    all_z_lin = []
    all_signal = []
    all_y_oc = []

    dispersion_history: List[float] = []
    history_start = max(0, start_idx - 60)

    for idx in range(history_start, len(df_exec)):
        gap_t = None
        if strategy_cfg.signal_mode == "gap_residual" and jp_gap is not None:
            gap_t = np.nan_to_num(jp_gap[idx], nan=0.0)
        betas_t = np.asarray(jp_beta[idx], dtype=float) if jp_beta is not None else None
        topix_night_t = float(topix_night[idx]) if topix_night is not None else None

        sig_result = signals.compute_signal(
            all_returns, idx, N_U,
            strategy_cfg.corr_window, c_full, v0_static, v1, v2,
            strategy_cfg.k, strategy_cfg.lambda_reg, strategy_cfg.lambda_lw,
            strategy_cfg.lw_target, strategy_cfg.ewma_half_life,
            v3_dynamic=(strategy_cfg.v3_mode == "dynamic"),
            gap_override=gap_t, gap_open_coef=strategy_cfg.gap_open_coef,
            topix_beta_coef=strategy_cfg.topix_beta_coef,
            betas_t=betas_t, topix_night_t=topix_night_t,
        )

        if idx >= start_idx:
            f_t = np.asarray(sig_result["f_t"], dtype=float)
            z_hat = np.asarray(sig_result["z_hat_j_t1"], dtype=float)
            sig = np.asarray(sig_result["signal"], dtype=float)
            y_t = np.nan_to_num(jp_oc[idx], nan=0.0)

            all_f_t.append(f_t)
            all_z_lin.append(z_hat)
            all_signal.append(sig)
            all_y_oc.append(y_t)

            disp_ind = signals.compute_dispersion_indicator(sig, strategy_cfg.q, N_J, strategy_cfg.dispersion_metric)
            dispersion_history.append(disp_ind)

    T_total = len(all_f_t)
    f_matrix = np.stack(all_f_t)          # (T_total, K)
    z_lin_matrix = np.stack(all_z_lin)    # (T_total, N_J)
    signal_matrix = np.stack(all_signal)  # (T_total, N_J)
    y_matrix = np.stack(all_y_oc)         # (T_total, N_J)
    trade_dates = date_index[start_idx:]  # (T_total,)

    # リーク監査: signal_date (= US close = trade_date - 1bd) を近似
    signal_dates_approx = trade_dates - pd.Timedelta(days=1)
    audit_no_leak(signal_dates_approx, trade_dates, "walk_forward_gp")
    logger.info("リーク監査: 全サンプルでリークなし。")

    # ── Walk-Forward ループ ─────────────────────────────────────────────────
    # OOS 評価期間のインデックス（全期間の中での相対インデックス）
    oos_rel_start = oos_start_idx - start_idx
    oos_rel_start = max(0, min(oos_rel_start, T_total - 1))
    oos_dates = trade_dates[oos_rel_start:]

    # 結果格納
    linear_daily_returns = []
    gp_daily_returns = []
    kappa_list = []
    mu_list = []
    sigma2_list = []
    ard_history = []

    # 確信度スコアラー（EWMAスムージング用ステートフル）
    scorer = ConfidenceScorer(conf_cfg)

    logger.info(
        "Walk-Forward OOS評価開始: %s 〜 %s (%d 日)",
        oos_dates[0].date() if len(oos_dates) > 0 else "N/A",
        oos_dates[-1].date() if len(oos_dates) > 0 else "N/A",
        len(oos_dates),
    )

    window = gp_cfg.window_size
    gp_module = GPUncertaintyModule(gp_cfg)
    last_fit_idx = -1  # 最後に fit した時点（コスト削減のため毎月 refit）
    refit_interval = 21  # 約1ヶ月ごとに refit

    for oos_rel_i, t_date in enumerate(oos_dates):
        abs_rel_i = oos_rel_start + oos_rel_i  # T_total 内での絶対相対インデックス

        # ── GP の (re)fit ──────────────────────────────────────────────
        if oos_rel_i == 0 or (oos_rel_i - last_fit_idx) >= refit_interval:
            # ウィンドウ: abs_rel_i の直前 window 日分（将来情報含まず）
            win_end = abs_rel_i  # 現在日は含まない（target が未来だから）
            win_start = max(0, win_end - window)

            X_win = f_matrix[win_start:win_end]
            Z_win = z_lin_matrix[win_start:win_end]
            Y_win = y_matrix[win_start:win_end]
            D_win = trade_dates[win_start:win_end]
            S_win = signal_dates_approx[win_start:win_end]

            if len(X_win) >= 30:
                try:
                    gp_module.fit(X_win, Z_win, Y_win, D_win, signal_dates=S_win)
                    last_fit_idx = oos_rel_i

                    # ARD サマリーを記録
                    ard_summary = gp_module.get_ard_summary()
                    ard_history.append({"date": t_date, "ard": ard_summary})
                    logger.debug("GP refit: t=%s, window=%d日", t_date.date(), len(X_win))
                except Exception as e:
                    logger.warning("GP fit 失敗 (%s): %s", t_date.date(), e)

        # ── 現在日のシグナル・予測 ─────────────────────────────────────
        f_t = f_matrix[abs_rel_i]
        z_lin_t = z_lin_matrix[abs_rel_i]
        signal_t = signal_matrix[abs_rel_i]
        y_t = y_matrix[abs_rel_i]

        # 分散フィルタースケール（既存機能）
        disp_ind = signals.compute_dispersion_indicator(
            signal_t, strategy_cfg.q, N_J, strategy_cfg.dispersion_metric
        )
        hist_window = dispersion_history[max(0, abs_rel_i - 60): abs_rel_i]
        scale = signals.dispersion_scale(disp_ind, hist_window, strategy_cfg.dispersion_filter)

        # 既存ウェイト（線形・ベンチマーク）
        w_base = signals.build_weights(signal_t, strategy_cfg.q, N_J, strategy_cfg.weight_mode)
        w_base_scaled = w_base * scale  # 既存 dispersion フィルター適用

        # 線形戦略のリターン
        linear_ret = float(np.sum(w_base_scaled * y_t))

        # GP 予測
        if gp_module._is_fitted:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mu_t, sigma2_t = gp_module.predict_mean_var(f_t, z_lin_t)
        else:
            mu_t = z_lin_t.copy()
            sigma2_t = np.full(N_J, np.nan)

        # 確信度スコア κ_t
        kappa_t = scorer.score(sigma2_t, weights=w_base_scaled)

        # GP-Sizing ウェイト
        w_gp = scorer.apply_sizing(w_base_scaled, kappa_t)

        # GP 戦略のリターン
        gp_ret = float(np.sum(w_gp * y_t))

        # 記録
        linear_daily_returns.append({"date": t_date, "return": linear_ret})
        gp_daily_returns.append({"date": t_date, "return": gp_ret})
        kappa_list.append({"date": t_date, "kappa": kappa_t})
        mu_list.append(mu_t)
        sigma2_list.append(sigma2_t)

    # ── 結果の整形 ─────────────────────────────────────────────────────────
    linear_ret_series = pd.Series(
        [r["return"] for r in linear_daily_returns],
        index=pd.DatetimeIndex([r["date"] for r in linear_daily_returns]),
        name="linear",
    )
    gp_ret_series = pd.Series(
        [r["return"] for r in gp_daily_returns],
        index=pd.DatetimeIndex([r["date"] for r in gp_daily_returns]),
        name="gp_sizing",
    )
    kappa_series = pd.Series(
        [k["kappa"] for k in kappa_list],
        index=pd.DatetimeIndex([k["date"] for k in kappa_list]),
        name="kappa",
    )

    mu_panel = np.stack(mu_list) if mu_list else np.zeros((0, N_J))
    sigma2_panel = np.stack(sigma2_list) if sigma2_list else np.zeros((0, N_J))
    y_oos_panel = y_matrix[oos_rel_start:]

    return {
        "oos_returns_linear": linear_ret_series,
        "oos_returns_gp": gp_ret_series,
        "kappa_series": kappa_series,
        "mu_panel": mu_panel,
        "sigma2_panel": sigma2_panel,
        "y_oos_panel": y_oos_panel,
        "signal_oos_panel": signal_matrix[oos_rel_start:],
        "ard_history": ard_history,
        "oos_dates": oos_dates,
    }


# ---------------------------------------------------------------------------
# レポート生成
# ---------------------------------------------------------------------------


def generate_report(
    results: Dict,
    cost_model: CostModel,
    calibration_levels: List[float],
    calibration_warn_threshold: float,
    ar_tolerance: float,
    significance_level: float,
    output_dir: Path,
) -> Dict:
    """比較レポートを生成し、ファイルに出力する。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    lin_gross = results["oos_returns_linear"]
    gp_gross = results["oos_returns_gp"]
    kappa_series = results["kappa_series"]
    mu_panel = results["mu_panel"]
    sigma2_panel = results["sigma2_panel"]
    y_oos_panel = results["y_oos_panel"]

    # ── コスト控除 ─────────────────────────────────────────────────────────
    lin_net = compute_net_returns(lin_gross, cost_model=cost_model)
    gp_net = compute_net_returns(gp_gross, cost_model=cost_model)

    # ── パフォーマンス指標 ─────────────────────────────────────────────────
    m_lin = compute_performance_metrics(lin_net, label="線形ベンチマーク (κ=1.0固定)")
    m_gp = compute_performance_metrics(gp_net, label="GP-Sizing (κ_t動的)")

    # ── 不確実性較正テスト ─────────────────────────────────────────────────
    calibration = coverage_test(
        mu_panel, sigma2_panel, y_oos_panel,
        levels=calibration_levels,
        warn_threshold=calibration_warn_threshold,
    )
    calibration_by_sector = coverage_test_by_sector(
        mu_panel, sigma2_panel, y_oos_panel,
        level=0.80, warn_threshold=calibration_warn_threshold,
    )

    # ── 採用判定 ───────────────────────────────────────────────────────────
    decision = evaluate_gp_adoption(
        m_lin, m_gp, calibration,
        ar_tolerance=ar_tolerance,
        significance_level=significance_level,
    )

    # ── レジーム別分析 ─────────────────────────────────────────────────────
    # 20日ボラティリティをレジーム代理変数として使用
    vol_proxy = lin_gross.rolling(20).std() * np.sqrt(245)
    regime_df = regime_kappa_decomposition(kappa_series, vol_proxy, n_quantiles=3)

    # 確信度分位別リターン分解
    quantile_df = quantile_return_decomposition(gp_net, kappa_series, n_quantiles=3)

    # ── コンソール出力 ─────────────────────────────────────────────────────
    sep = "=" * 72
    print(f"\n{sep}")
    print("  GP 不確実性推定モジュール バックテスト レポート")
    print(sep)

    def _pct(v): return f"{v*100:.2f}%" if np.isfinite(v) else "N/A"
    def _fmt(v): return f"{v:.3f}" if np.isfinite(v) else "N/A"

    print(f"\n{'指標':<28} {'線形ベンチマーク':>18} {'GP-Sizing':>18} {'差分':>10}")
    print("-" * 76)
    rows = [
        ("年率リターン (AR, net)",    m_lin.ar,    m_gp.ar,    _pct),
        ("年率リスク (RISK)",          m_lin.risk,  m_gp.risk,  _pct),
        ("R/R (AR / RISK)",           m_lin.rr,    m_gp.rr,    _fmt),
        ("最大DD (MDD)",              m_lin.mdd,   m_gp.mdd,   _pct),
        ("Sharpe Ratio",              m_lin.sharpe, m_gp.sharpe, _fmt),
        ("OOS 日数",  float(m_lin.n_obs), float(m_gp.n_obs), _fmt),
    ]
    for label, v_lin, v_gp, fmt in rows:
        diff = v_gp - v_lin
        diff_str = ("+" if diff > 0 else "") + fmt(diff)
        print(f"{label:<28} {fmt(v_lin):>18} {fmt(v_gp):>18} {diff_str:>10}")

    print(f"\n{sep}")
    print(f"【採用判定】{decision['decision']}")
    print(decision['reason'])
    print(sep)
    print(f"\n【GP 較正テスト】")
    print(calibration)

    if len(regime_df) > 0:
        print(f"\n【レジーム別 κ_t 分布】")
        print(regime_df.to_string())

    if len(quantile_df) > 0:
        print(f"\n【確信度分位別リターン分解】")
        print(quantile_df.to_string())

    # ── ファイル出力 ───────────────────────────────────────────────────────
    # OOS リターン CSV
    ret_df = pd.DataFrame({
        "gross_linear": lin_gross,
        "gross_gp": gp_gross,
        "net_linear": lin_net,
        "net_gp": gp_net,
        "kappa": kappa_series,
    })
    ret_df.to_csv(output_dir / "oos_returns.csv", encoding="utf-8-sig")

    # 較正テスト結果
    cal_df = pd.DataFrame({
        "level": calibration.levels,
        "expected": calibration.expected_coverages,
        "actual": calibration.actual_coverages,
        "error": calibration.coverage_errors,
    })
    cal_df.to_csv(output_dir / "calibration_result.csv", index=False, encoding="utf-8-sig")

    # 業種別較正
    calibration_by_sector.to_csv(
        output_dir / "calibration_by_sector.csv", encoding="utf-8-sig"
    )

    # 採用判定 JSON
    import json
    with open(output_dir / "adoption_decision.json", "w", encoding="utf-8") as f:
        json.dump({
            "decision": decision["decision"],
            "reason": decision["reason"],
            "failures": decision["failures"],
            "metrics_linear": {
                "ar": m_lin.ar, "risk": m_lin.risk, "rr": m_lin.rr,
                "mdd": m_lin.mdd, "sharpe": m_lin.sharpe,
            },
            "metrics_gp": {
                "ar": m_gp.ar, "risk": m_gp.risk, "rr": m_gp.rr,
                "mdd": m_gp.mdd, "sharpe": m_gp.sharpe,
            },
            "calibration_status": calibration.calibration_status,
        }, f, indent=2, ensure_ascii=False)

    # チャート生成
    _generate_charts(
        lin_net, gp_net, kappa_series,
        mu_panel, sigma2_panel, y_oos_panel,
        results.get("ard_history", []),
        quantile_df, regime_df,
        output_dir,
    )

    return decision


# ---------------------------------------------------------------------------
# チャート生成
# ---------------------------------------------------------------------------


def _generate_charts(
    lin_net, gp_net, kappa_series,
    mu_panel, sigma2_panel, y_oos_panel,
    ard_history, quantile_df, regime_df,
    output_dir: Path,
):
    """全チャートを生成して保存する。"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        # チャート1: 累積リターン + ドローダウン + κ_t
        fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)

        (1 + lin_net).cumprod().plot(ax=axes[0], label="線形ベンチマーク (κ=1.0)", color="steelblue", linewidth=1.5)
        (1 + gp_net).cumprod().plot(ax=axes[0], label="GP-Sizing (κ_t 動的)", color="darkorange", linewidth=1.5, linestyle="--")
        axes[0].set_title("OOS 累積ネットリターン（取引コスト控除後）", fontsize=12)
        axes[0].set_ylabel("Wealth")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        def _dd(s): return ((1 + s).cumprod() / (1 + s).cumprod().cummax()) - 1
        _dd(lin_net).plot(ax=axes[1], color="steelblue", linewidth=1.0, label="線形")
        _dd(gp_net).plot(ax=axes[1], color="darkorange", linewidth=1.0, linestyle="--", label="GP-Sizing")
        axes[1].set_title("ドローダウン比較", fontsize=12)
        axes[1].set_ylabel("Drawdown")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

        kappa_series.plot(ax=axes[2], color="purple", linewidth=1.0, alpha=0.8, label="κ_t")
        axes[2].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        axes[2].axhline(0.5, color="orange", linestyle=":", linewidth=0.8)
        axes[2].fill_between(kappa_series.index, kappa_series.values, 1.0, where=kappa_series.values < 1.0, alpha=0.15, color="red", label="κ_t < 1.0")
        axes[2].set_title("確信度スコア κ_t の時系列", fontsize=12)
        axes[2].set_ylabel("κ_t")
        axes[2].set_ylim(0, 1.1)
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "gp_sizing_overview.png", dpi=150, bbox_inches="tight")
        plt.close()

        # チャート2: ARD 長さスケール（最後の fit 時点）
        if ard_history:
            last_ard = ard_history[-1]["ard"]
            valid_ard = [s for s in last_ard if "length_scales" in s]
            if valid_ard:
                # 全業種の平均長さスケール
                all_ls = np.stack([s["length_scales"] for s in valid_ard])
                mean_ls = all_ls.mean(axis=0)
                factor_names = valid_ard[0].get("factor_names", [f"f_{k}" for k in range(len(mean_ls))])

                fig, ax = plt.subplots(figsize=(8, 5))
                colors = ["steelblue" if ls == mean_ls.min() else "lightsteelblue" for ls in mean_ls]
                bars = ax.barh(factor_names, 1.0 / mean_ls, color=colors, edgecolor="white")
                ax.set_xlabel("非線形寄与重要度（1 / length_scale）")
                ax.set_title("ARD 長さスケール逆数（業種平均）\n小さい長さスケール = 高い非線形寄与", fontsize=11)
                ax.grid(axis="x", alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / "ard_importance.png", dpi=150, bbox_inches="tight")
                plt.close()

        # チャート3: 較正チャート（Reliability Diagram）
        fig = reliability_diagram(mu_panel, sigma2_panel, y_oos_panel, title="GP 予測区間 較正チャート")
        fig.savefig(output_dir / "calibration_reliability.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # チャート4: 確信度分位別リターン分解（棒グラフ）
        if len(quantile_df) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            x = range(len(quantile_df))
            ax.bar(x, quantile_df["annualized_return"] * 100, color=["#e74c3c", "#f39c12", "#2ecc71"])
            ax.set_xticks(x)
            ax.set_xticklabels([f"{idx}\n(κ≈{row['kappa_mean']:.2f})" for idx, row in quantile_df.iterrows()])
            ax.set_ylabel("年率リターン (%)")
            ax.set_title("確信度分位別 年率リターン分解\n(低κ_t=低確信度, 高κ_t=高確信度)", fontsize=11)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "kappa_quantile_returns.png", dpi=150, bbox_inches="tight")
            plt.close()

        logger.info("チャートを %s に保存しました", output_dir)

    except Exception as e:
        logger.warning("チャート生成に失敗しました: %s", e)


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="GP 不確実性推定モジュール Walk-Forward バックテスト")
    parser.add_argument("--config", default=str(CONFIG_DIR / "gp_uncertainty.yaml"), help="設定ファイルパス")
    parser.add_argument("--window", type=int, default=None, help="ローリング窓（営業日数）を上書き")
    parser.add_argument("--start-date", default=None, help="バックテスト開始日を上書き")
    parser.add_argument("--oos-start", default=None, help="OOS 開始日を上書き")
    parser.add_argument("--output-dir", default=None, help="出力ディレクトリを上書き")
    args = parser.parse_args()

    # ── 設定読み込み ─────────────────────────────────────────────────────
    import yaml
    config_path = Path(args.config)
    if not config_path.exists():
        logger.warning("設定ファイルが見つかりません: %s、デフォルト設定を使用します", config_path)
        cfg_dict = {}
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)

    # コマンドライン引数で上書き
    if args.window:
        cfg_dict.setdefault("fitting", {})["window_size"] = args.window

    start_date = args.start_date or cfg_dict.get("walk_forward", {}).get("start_date", "2015-01-01")
    oos_start_date = args.oos_start or cfg_dict.get("walk_forward", {}).get("oos_start_date", "2020-01-01")
    output_dir = Path(args.output_dir or cfg_dict.get("output", {}).get("output_dir", "results/gp_walkforward"))
    output_dir = SRC_DIR.parent / output_dir

    gp_cfg = GPConfig.from_config_dict(cfg_dict)
    conf_cfg = ConfidenceConfig.from_config_dict(cfg_dict)
    cost_cfg = cfg_dict.get("cost_model", {})
    cost_model = CostModel(
        commission_rate=float(cost_cfg.get("commission_rate", 0.0005)),
        spread_rate=float(cost_cfg.get("spread_rate", 0.0003)),
        impact_rate=float(cost_cfg.get("impact_rate", 0.0002)),
    )
    cal_cfg = cfg_dict.get("calibration", {})
    calibration_levels = list(cal_cfg.get("coverage_levels", [0.80, 0.90, 0.95]))
    calibration_warn_threshold = float(cal_cfg.get("warn_threshold", 0.05))
    thr_cfg = cfg_dict.get("thresholds", {})
    ar_tolerance = float(thr_cfg.get("ar_tolerance", 0.95))
    significance_level = float(thr_cfg.get("significance_level", 0.05))
    wf_cfg = cfg_dict.get("walk_forward", {})
    gap_purge = int(wf_cfg.get("gap_purge", 1))
    embargo = int(wf_cfg.get("embargo", 5))

    # 既存戦略設定（デフォルトを使用）
    strategy_cfg = StrategyConfig(
        k=STRATEGY_DEFAULTS["K"],
        lambda_reg=STRATEGY_DEFAULTS["lambda_reg"],
        q=STRATEGY_DEFAULTS["q"],
        weight_mode=STRATEGY_DEFAULTS["weight_mode"],
        dispersion_filter=STRATEGY_DEFAULTS["dispersion_filter"],
        dispersion_metric=STRATEGY_DEFAULTS.get("dispersion_metric", "long_short_mean_gap"),
        v3_mode=STRATEGY_DEFAULTS["v3_mode"],
        ewma_half_life=STRATEGY_DEFAULTS.get("ewma_half_life"),
        lambda_lw=STRATEGY_DEFAULTS.get("lambda_lw", 0.0),
        lw_target=STRATEGY_DEFAULTS.get("lw_target", "identity"),
        corr_window=STRATEGY_DEFAULTS.get("corr_window", 60),
        include_v4_prior=STRATEGY_DEFAULTS.get("include_v4_prior", True),
        signal_mode=STRATEGY_DEFAULTS.get("signal_mode", "gap_residual"),
        gap_open_coef=STRATEGY_DEFAULTS.get("gap_open_coef", 0.70),
        topix_beta_coef=STRATEGY_DEFAULTS.get("topix_beta_coef", 1.20),
        beta_window=STRATEGY_DEFAULTS.get("beta_window", 60),
        gamma=STRATEGY_DEFAULTS.get("gamma", 0.5),
    )

    # ── データ読み込み ─────────────────────────────────────────────────────
    logger.info("データ読み込み中...")
    data = download_data(beta_window=STRATEGY_DEFAULTS.get("beta_window", 60))
    df_exec = preprocess_data(data, beta_window=STRATEGY_DEFAULTS.get("beta_window", 60))
    logger.info("データ準備完了: %d 行", len(df_exec))

    # ── Walk-Forward 実行 ─────────────────────────────────────────────────
    logger.info("Walk-Forward バックテスト開始...")
    logger.info(
        "設定: window=%d日, kernel=%s, alpha_max=%.2f, ls_min=%.2f",
        gp_cfg.window_size, gp_cfg.kernel_cfg.nonlinear_type,
        gp_cfg.kernel_cfg.alpha_bounds[1], gp_cfg.kernel_cfg.length_scale_bounds[0],
    )

    wf_results = run_walk_forward_gp(
        df_exec=df_exec,
        gp_cfg=gp_cfg,
        conf_cfg=conf_cfg,
        strategy_cfg=strategy_cfg,
        cost_model=cost_model,
        start_date=start_date,
        oos_start_date=oos_start_date,
        gap_purge=gap_purge,
        embargo=embargo,
        output_dir=output_dir,
    )

    # ── レポート生成 ───────────────────────────────────────────────────────
    logger.info("レポート生成中...")
    decision = generate_report(
        results=wf_results,
        cost_model=cost_model,
        calibration_levels=calibration_levels,
        calibration_warn_threshold=calibration_warn_threshold,
        ar_tolerance=ar_tolerance,
        significance_level=significance_level,
        output_dir=output_dir,
    )

    print(f"\n結果を {output_dir} に保存しました。")
    return decision["decision"]


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result == "ADOPT" else 1)
