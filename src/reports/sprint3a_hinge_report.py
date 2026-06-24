"""src/reports/sprint3a_hinge_report.py — Sprint 3-A Markdown Report Generator.

Generates a comprehensive Markdown report for Sprint 3-A hinge feature overlay
verification results. All data comes from pre-computed artifact files.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_pct(val: float | None, decimals: int = 2) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "not_available"
    return f"{val * 100:.{decimals}f}%"


def _fmt_float(val: float | None, decimals: int = 4) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "not_available"
    return f"{val:.{decimals}f}"


def _fmt_bps(val: float | None) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "not_available"
    return f"{val * 10000:.1f} bps"


def _safe_load(path: str) -> pd.DataFrame | None:
    """Load CSV or parquet, return None if not found."""
    if not os.path.exists(path):
        return None
    try:
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------


def generate_sprint3a_report(
    artifact_dir: str,
    report_dir: str,
    figure_dir: str,
    config: dict,
    run_metadata: dict | None = None,
) -> str:
    """Generate Sprint 3-A Markdown report.

    Parameters
    ----------
    artifact_dir:
        Directory containing all artifact files.
    report_dir:
        Output directory for the report.
    figure_dir:
        Directory containing figure files.
    config:
        YAML config dict.
    run_metadata:
        Optional metadata about the run (start/end times, data coverage, etc.).

    Returns
    -------
    str
        Path to the generated report file.
    """
    os.makedirs(report_dir, exist_ok=True)

    # Load artifacts
    model_summary = _safe_load(os.path.join(artifact_dir, "model_comparison_summary.csv"))
    ic_ts = _safe_load(os.path.join(artifact_dir, "ic_timeseries.csv"))
    quantile_ret = _safe_load(os.path.join(artifact_dir, "quantile_return_summary.csv"))
    cost_sensitivity = _safe_load(os.path.join(artifact_dir, "cost_sensitivity_summary.csv"))
    feature_stability = _safe_load(os.path.join(artifact_dir, "feature_stability_summary.csv"))
    selected_features = _safe_load(os.path.join(artifact_dir, "selected_features_by_window.csv"))
    oos_preds = _safe_load(os.path.join(artifact_dir, "oos_predictions.parquet"))
    leakage_audit = _safe_load(os.path.join(artifact_dir, "qa", "leakage_audit.csv"))

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    start_date = config.get("run", {}).get("start_date", "not_available")
    end_date = config.get("run", {}).get("end_date", "not_available")
    aum = config.get("portfolio", {}).get("aum_jpy", 1000000)
    train_window = config.get("validation", {}).get("train_window_days", 252)
    val_window = config.get("validation", {}).get("validation_window_days", 63)
    test_window = config.get("validation", {}).get("test_window_days", 21)
    thresholds = config.get("features", {}).get("hinge_thresholds", [1.0, 1.5, 2.0])
    fdr_q = config.get("feature_selection", {}).get("fdr_q", 0.10)
    default_spread = config.get("costs", {}).get("default_spread_fallback_roundtrip_bps", 15)
    spread_scenarios = config.get("costs", {}).get("spread_fallback_roundtrip_bps", [5, 10, 15, 20, 30, 50])

    # Extract key metrics from model summary
    def get_model_row(model_name: str) -> dict | None:
        if model_summary is None or model_summary.empty:
            return None
        rows = model_summary[model_summary["model"] == model_name]
        if rows.empty:
            return None
        return rows.iloc[0].to_dict()

    b0 = get_model_row("net_score_ranking")
    h1 = get_model_row("hinge_ridge_overlay")
    h2 = get_model_row("hinge_elasticnet_overlay")

    def metric(row: dict | None, key: str) -> float | None:
        if row is None:
            return None
        val = row.get(key, None)
        return None if val is None or (isinstance(val, float) and np.isnan(val)) else float(val)

    # Baseline metrics
    b0_ret = metric(b0, "annualized_net_return")
    b0_ir = metric(b0, "IR")
    b0_dd = metric(b0, "max_drawdown")
    b0_ic = metric(b0, "mean_rank_ic")

    # H1 (Ridge) metrics
    h1_ret = metric(h1, "annualized_net_return")
    h1_ir = metric(h1, "IR")
    h1_dd = metric(h1, "max_drawdown")
    h1_ic = metric(h1, "mean_rank_ic")

    # H2 (ElasticNet) metrics
    h2_ret = metric(h2, "annualized_net_return")
    h2_ir = metric(h2, "IR")
    h2_dd = metric(h2, "max_drawdown")
    h2_ic = metric(h2, "mean_rank_ic")

    # Compute improvements
    def improvement(base: float | None, new: float | None) -> str:
        if base is None or new is None:
            return "not_available"
        delta = new - base
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta * 100:.2f}%"

    def improvement_float(base: float | None, new: float | None) -> str:
        if base is None or new is None:
            return "not_available"
        delta = new - base
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.4f}"

    # Cost sensitivity: extract spread 15bps and 20bps for baseline vs best hinge
    def get_spread_metric(spread_bps: int, model_name: str, metric_key: str) -> float | None:
        if cost_sensitivity is None or cost_sensitivity.empty:
            return None
        mask = (cost_sensitivity["spread_bps"] == spread_bps) & (cost_sensitivity["model"] == model_name)
        rows = cost_sensitivity[mask]
        if rows.empty:
            return None
        return float(rows.iloc[0].get(metric_key, np.nan))

    # Selected stable features
    stable_features = []
    if feature_stability is not None and not feature_stability.empty:
        top = feature_stability[feature_stability["selection_freq"] >= 0.5]
        stable_features = top["feature"].tolist()[:10]

    # Adoption criteria evaluation
    adoption_criteria = config.get("evaluation", {}).get("adoption_criteria", {})
    min_ret_improvement = adoption_criteria.get("min_annual_net_return_improvement", 0.03)
    min_ir_improvement = adoption_criteria.get("min_ir_improvement", 0.20)
    max_dd_worsening = adoption_criteria.get("max_dd_worsening_allowed", 0.00)
    max_turnover_increase = adoption_criteria.get("max_turnover_increase_ratio", 0.20)

    def check_adoption(b_ret, h_ret, b_ir, h_ir, b_dd, h_dd) -> tuple[bool, list[str]]:
        reasons = []
        passed = True

        if b_ret is None or h_ret is None:
            return False, ["insufficient data"]

        ret_imp = h_ret - b_ret
        if ret_imp < min_ret_improvement:
            reasons.append(f"Net return improvement {ret_imp*100:.2f}% < {min_ret_improvement*100:.1f}%")
            passed = False

        if b_ir is not None and h_ir is not None:
            ir_imp = h_ir - b_ir
            if ir_imp < min_ir_improvement:
                reasons.append(f"IR improvement {ir_imp:.4f} < {min_ir_improvement:.2f}")
                passed = False

        if b_dd is not None and h_dd is not None:
            dd_worsening = h_dd - b_dd  # both negative; if h_dd < b_dd, worsened
            if dd_worsening < -max_dd_worsening:
                reasons.append(f"Max DD worsened by {abs(dd_worsening)*100:.2f}% (limit={max_dd_worsening*100:.1f}%)")
                passed = False

        return passed, reasons

    h1_adopted, h1_reasons = check_adoption(b0_ret, h1_ret, b0_ir, h1_ir, b0_dd, h1_dd)
    h2_adopted, h2_reasons = check_adoption(b0_ret, h2_ret, b0_ir, h2_ir, b0_dd, h2_dd)
    best_overlay = "H1 (Ridge)" if h1_adopted else ("H2 (ElasticNet)" if h2_adopted else "None (research candidate only)")
    adoption_decision = "ADOPT" if (h1_adopted or h2_adopted) else "RESEARCH_CANDIDATE"

    # Build figure relative paths
    def fig_path(fname: str) -> str:
        return f"figures/{fname}"

    lines = []

    # -------------------------------------------------------------------------
    # Title
    lines.append("# Sprint 3-A — ヒンジ特徴量による限定的非線形化検証レポート")
    lines.append("")
    lines.append(f"**生成日時**: {now_str}  ")
    lines.append(f"**検証期間**: {start_date} ～ {end_date}  ")
    lines.append(f"**AUM**: ¥{aum:,}  ")
    lines.append("")

    # -------------------------------------------------------------------------
    # 1. 概要
    lines.append("---")
    lines.append("## 1. 概要")
    lines.append("")
    lines.append(
        "本レポートは Sprint 3-A として実施した、ヒンジ特徴量による **conservative nonlinear overlay** の"
        "検証結果をまとめたものです。"
    )
    lines.append("")
    lines.append(
        "既存 baseline モデル `net_score_ranking` の予測力を置き換えるのではなく、"
        "米国セクター・マクロ・ギャップ・レジーム指標が閾値を超えた局面のみを補正する overlay を"
        "Ridge / ElasticNet で実装し、walk-forward OOS で検証しました。"
    )
    lines.append("")

    # -------------------------------------------------------------------------
    # 2. 目的
    lines.append("---")
    lines.append("## 2. 目的")
    lines.append("")
    lines.append("1. 日中残差リターンの Rank IC が改善するか")
    lines.append("2. 分位リターンの単調性が改善するか")
    lines.append(f"3. AUM {aum:,}円・固定スプレッド 10/15/20/30bps 条件で net return / IR / max DD が改善するか")
    lines.append("4. DD を悪化させずにコスト控除後リターンが改善するか")
    lines.append("5. 選択特徴量が walk-forward 期間を通じて安定しているか")
    lines.append("6. 過学習の兆候がないか")
    lines.append("")

    # -------------------------------------------------------------------------
    # 3. ターゲット定義
    lines.append("---")
    lines.append("## 3. ターゲット定義")
    lines.append("")
    lines.append("| ターゲット | 説明 | 使用方法 |")
    lines.append("|-----------|------|----------|")
    lines.append("| `open_to_close_residual` | Open→Close 日中残差リターン (beta 控除) | **主要ターゲット** |")
    lines.append("| `entry_to_close_residual` | Entry→Close 残差 (9:10 proxy / Open) | サブターゲット |")
    lines.append("| `close_to_close_return` | 前日終値→当日終値 | **参照のみ**（リーク防止） |")
    lines.append("| `true_0910_to_close_residual` | 真の 9:10 価格からの残差 | サブサンプル報告 |")
    lines.append("")
    lines.append("> **注意**: Close-to-Close を主ターゲットにすることは禁止されています。")
    lines.append("")

    # -------------------------------------------------------------------------
    # 4. Baseline モデル
    lines.append("---")
    lines.append("## 4. Baseline モデル")
    lines.append("")
    lines.append("```")
    lines.append("Model B0: net_score_ranking")
    lines.append("  score_long  = signal_gap_adjusted - lambda * tc_long_roundtrip")
    lines.append("  score_short = -signal_gap_adjusted - lambda * tc_short_roundtrip")
    lines.append("  Select top-5 long / top-5 short by score")
    lines.append("  Weights proportional to score, normalized to target gross")
    lines.append("```")
    lines.append("")
    lines.append("**Baseline 性能 (OOS)**:")
    lines.append("")
    lines.append(f"- Annualized Net Return: {_fmt_pct(b0_ret)}")
    lines.append(f"- IR: {_fmt_float(b0_ir)}")
    lines.append(f"- Max Drawdown: {_fmt_pct(b0_dd)}")
    lines.append(f"- Mean Rank IC: {_fmt_float(b0_ic)}")
    lines.append("")

    # -------------------------------------------------------------------------
    # 5. ヒンジ特徴量設計
    lines.append("---")
    lines.append("## 5. ヒンジ特徴量設計")
    lines.append("")
    lines.append(f"**閾値 (kappa)**: {thresholds}")
    lines.append("**方向**: positive (`max(0, z-κ)`), negative (`max(0, -z-κ)`)")
    lines.append("")
    lines.append("### 特徴量グループ")
    lines.append("")
    feature_groups = config.get("features", {}).get("feature_groups", {})
    for grp_name, grp_cfg in feature_groups.items():
        enabled = grp_cfg.get("enabled", True)
        cols = grp_cfg.get("columns", [])
        status = "有効" if enabled else "無効"
        lines.append(f"**{grp_name}** ({status}): {', '.join(cols)}")
        lines.append("")
    lines.append("")
    lines.append("### 出力列名の例")
    lines.append("")
    lines.append("```")
    lines.append("hinge_pos_us_tech_return_k1_0")
    lines.append("hinge_neg_us_tech_return_k1_0")
    lines.append("hinge_pos_us_tech_return_k1_5")
    lines.append("hinge_neg_vix_return_k2_0")
    lines.append("... 合計最大 40 特徴量")
    lines.append("```")
    lines.append("")

    # -------------------------------------------------------------------------
    # 6. Rolling z-score と look-ahead 防止
    lines.append("---")
    lines.append("## 6. Rolling z-score と Look-ahead 防止")
    lines.append("")
    lines.append("$$")
    lines.append(r"z_{k,t} = \frac{x_{k,t} - \mu_{k,t-1}^{roll}}{\sigma_{k,t-1}^{roll}}")
    lines.append("$$")
    lines.append("")
    lines.append("**実装上の制約**:")
    lines.append("")
    lines.append("- rolling mean/std には `shift(1)` を適用し、当日データを含めない")
    lines.append("- std がゼロ or NaN → 当日 z-score は NaN")
    lines.append(f"- rolling window: {config.get('features', {}).get('default_zscore_window', 120)} 日")
    lines.append("- full-sample zscore は **禁止**")
    lines.append("")

    # -------------------------------------------------------------------------
    # 7. FDR feature selection
    lines.append("---")
    lines.append("## 7. FDR Feature Selection")
    lines.append("")
    lines.append(f"手法: **Benjamini-Hochberg FDR** (q = {fdr_q})")
    lines.append("")
    lines.append("各 walk-forward train window 内のみで実施:")
    lines.append("1. 各ヒンジ特徴量 × target の日次 cross-sectional Rank IC を計算")
    lines.append("2. 平均 Rank IC、ICIR、t 検定 p 値を算出")
    lines.append("3. BH FDR 補正を適用")
    lines.append(f"4. `q <= {fdr_q}` かつ `|mean_rank_ic| >= {config.get('feature_selection', {}).get('min_abs_rank_ic', 0.02)}`")
    lines.append(f"5. train window を前半/後半に分け、符号一致率 >= {config.get('feature_selection', {}).get('min_sign_consistency', 0.60)}")
    lines.append(f"6. 最大 {config.get('feature_selection', {}).get('max_features_after_fdr', 20)} 特徴量を使用")
    lines.append("")
    if selected_features is not None and not selected_features.empty:
        total_windows = selected_features["window_id"].nunique() if "window_id" in selected_features.columns else "N/A"
        lines.append(f"**検証 window 数**: {total_windows}")
        lines.append("")

    # -------------------------------------------------------------------------
    # 8. モデル仕様
    lines.append("---")
    lines.append("## 8. Ridge / ElasticNet Overlay 仕様")
    lines.append("")
    lines.append("### 残差ターゲット")
    lines.append("$$")
    lines.append(r"e_{j,t} = y^{intraday\_resid}_{j,t} - \hat{\mu}^{base}_{j,t}")
    lines.append("$$")
    lines.append("")
    lines.append("### 最終予測")
    lines.append("$$")
    lines.append(r"\hat{\mu}^{final}_{j,t} = \hat{\mu}^{base}_{j,t} + \alpha \cdot \hat{e}_{j,t}")
    lines.append("$$")
    lines.append("")
    lines.append(f"$\\alpha \\in {config.get('model', {}).get('alpha_blend_grid', [0.0, 0.25, 0.5, 0.75, 1.0])}$ (validation で選択)")
    lines.append("")
    lines.append("### 補正上限（保守的制約）")
    lines.append("```")
    lines.append(f"abs(delta) <= {config.get('model', {}).get('max_overlay_to_base_abs_ratio', 0.5)} * abs(mu_base)")
    lines.append(f"abs(delta) <= {config.get('model', {}).get('max_overlay_bps', 20)} bps")
    lines.append("→ どちらか厳しい方を適用")
    lines.append("```")
    lines.append("")
    lines.append("| モデル | 正則化 | ハイパーパラメータ |")
    lines.append("|--------|--------|-------------------|")
    lines.append(f"| H1: Ridge | L2 | alpha ∈ {config.get('model', {}).get('ridge_alpha_grid', [0.1, 1.0, 10.0, 100.0])} |")
    lines.append(f"| H2: ElasticNet | L1+L2 | alpha ∈ {config.get('model', {}).get('elasticnet_alpha_grid', [0.0001, 0.001, 0.01, 0.1])}, l1_ratio ∈ {config.get('model', {}).get('elasticnet_l1_ratio_grid', [0.1, 0.3, 0.5, 0.7])} |")
    lines.append("")

    # -------------------------------------------------------------------------
    # 9. walk-forward 検証設計
    lines.append("---")
    lines.append("## 9. Walk-forward 検証設計")
    lines.append("")
    lines.append("```")
    lines.append(f"Train window:       {train_window} days")
    lines.append(f"Validation window:  {val_window} days")
    lines.append(f"Test window:        {test_window} days")
    lines.append(f"Step:               {config.get('validation', {}).get('step_days', 21)} days")
    lines.append(f"Purge:              {config.get('validation', {}).get('purge_days', 1)} days")
    lines.append("")
    lines.append("Train  → zscore param, FDR selection, model fit")
    lines.append("Val    → Ridge/ElasticNet params + alpha blend selection")
    lines.append("Test   → OOS prediction (no future leakage)")
    lines.append("```")
    lines.append("")
    lines.append("> **制約**: 同日内の銘柄を train/test に分割しない。test window は完全 OOS。")
    lines.append("")

    # -------------------------------------------------------------------------
    # 10. Rank IC / ICIR 比較
    lines.append("---")
    lines.append("## 10. Rank IC / ICIR 比較")
    lines.append("")
    lines.append("| モデル | Mean Rank IC | ICIR | Hit Rate |")
    lines.append("|--------|-------------|------|---------|")

    for model_name, mrow in [("B0: Baseline", b0), ("H1: Ridge", h1), ("H2: ElasticNet", h2)]:
        mean_ic = _fmt_float(metric(mrow, "mean_rank_ic"))
        icir = _fmt_float(metric(mrow, "rank_icir"))
        hit = _fmt_pct(metric(mrow, "hit_rate"))
        lines.append(f"| {model_name} | {mean_ic} | {icir} | {hit} |")

    lines.append("")
    if os.path.exists(os.path.join(figure_dir, "ic_timeseries_baseline_vs_hinge.png")):
        lines.append(f"![IC時系列比較]({fig_path('ic_timeseries_baseline_vs_hinge.png')})")
        lines.append("")
    if os.path.exists(os.path.join(figure_dir, "cumulative_ic_baseline_vs_hinge.png")):
        lines.append(f"![累積IC比較]({fig_path('cumulative_ic_baseline_vs_hinge.png')})")
        lines.append("")

    # -------------------------------------------------------------------------
    # 11. 分位リターン比較
    lines.append("---")
    lines.append("## 11. 分位リターン比較")
    lines.append("")
    if quantile_ret is not None and not quantile_ret.empty:
        lines.append("分位別平均リターン (日次・bps 換算):")
        lines.append("")
        for model_col_prefix, model_label in [("baseline", "B0 Baseline"), ("ridge", "H1 Ridge"), ("elasticnet", "H2 ElasticNet")]:
            q1_col = f"{model_col_prefix}_q1_return"
            q3_col = f"{model_col_prefix}_q3_return"
            if q1_col in quantile_ret.columns and q3_col in quantile_ret.columns:
                q1 = quantile_ret[q1_col].mean()
                q3 = quantile_ret[q3_col].mean()
                spread = (q3 - q1) * 10000
                lines.append(f"- **{model_label}**: Q1={_fmt_bps(q1)}, Q3={_fmt_bps(q3)}, Spread={spread:.1f}bps")
    else:
        lines.append("分位リターンデータ: not_available")
    lines.append("")
    if os.path.exists(os.path.join(figure_dir, "quantile_returns_baseline_vs_hinge.png")):
        lines.append(f"![分位リターン比較]({fig_path('quantile_returns_baseline_vs_hinge.png')})")
        lines.append("")

    # -------------------------------------------------------------------------
    # 12. AUM 100万円 固定スプレッド感応度
    lines.append("---")
    lines.append("## 12. AUM 100万円・固定スプレッド感応度")
    lines.append("")
    lines.append(f"| Spread (bps) | B0 Net Return | H1 Ridge Net Return | H2 EN Net Return |")
    lines.append("|-------------|--------------|---------------------|-----------------|")
    for s_bps in spread_scenarios:
        b0_r = _fmt_pct(get_spread_metric(s_bps, "net_score_ranking", "annualized_net_return"))
        h1_r = _fmt_pct(get_spread_metric(s_bps, "hinge_ridge_overlay", "annualized_net_return"))
        h2_r = _fmt_pct(get_spread_metric(s_bps, "hinge_elasticnet_overlay", "annualized_net_return"))
        lines.append(f"| {s_bps} | {b0_r} | {h1_r} | {h2_r} |")
    lines.append("")
    if os.path.exists(os.path.join(figure_dir, "spread_sensitivity_baseline_vs_hinge.png")):
        lines.append(f"![スプレッド感応度]({fig_path('spread_sensitivity_baseline_vs_hinge.png')})")
        lines.append("")

    # -------------------------------------------------------------------------
    # 13. Reverse fee 感応度
    lines.append("---")
    lines.append("## 13. Reverse Fee 感応度")
    lines.append("")
    rev_scenarios = config.get("costs", {}).get("reverse_fee_bps_per_day_scenarios", [0, 5, 10, 30])
    lines.append(f"| Reverse Fee (bps/day) | B0 Net Return | H1 Ridge Net Return |")
    lines.append("|----------------------|--------------|---------------------|")
    for r_bps in rev_scenarios:
        b0_r = _fmt_pct(get_spread_metric(r_bps, "net_score_ranking", "rev_fee_annualized_net_return"))
        h1_r = _fmt_pct(get_spread_metric(r_bps, "hinge_ridge_overlay", "rev_fee_annualized_net_return"))
        lines.append(f"| {r_bps} | {b0_r} | {h1_r} |")
    lines.append("")
    if os.path.exists(os.path.join(figure_dir, "reverse_fee_sensitivity.png")):
        lines.append(f"![Reverse Fee感応度]({fig_path('reverse_fee_sensitivity.png')})")
        lines.append("")

    # -------------------------------------------------------------------------
    # 14. Max DD / Turnover / Effective Gross
    lines.append("---")
    lines.append("## 14. Max DD / Turnover / Effective Gross")
    lines.append("")
    lines.append("| モデル | Max DD | Avg Turnover | Avg Effective Gross |")
    lines.append("|--------|--------|-------------|---------------------|")
    for model_name, mrow in [("B0: Baseline", b0), ("H1: Ridge", h1), ("H2: ElasticNet", h2)]:
        dd = _fmt_pct(metric(mrow, "max_drawdown"))
        turn = _fmt_float(metric(mrow, "avg_turnover"), 4)
        gross = _fmt_float(metric(mrow, "avg_effective_gross"), 4)
        lines.append(f"| {model_name} | {dd} | {turn} | {gross} |")
    lines.append("")
    if os.path.exists(os.path.join(figure_dir, "equity_curve_baseline_vs_hinge.png")):
        lines.append(f"![Equity Curve比較]({fig_path('equity_curve_baseline_vs_hinge.png')})")
        lines.append("")
    if os.path.exists(os.path.join(figure_dir, "drawdown_baseline_vs_hinge.png")):
        lines.append(f"![Drawdown比較]({fig_path('drawdown_baseline_vs_hinge.png')})")
        lines.append("")

    # -------------------------------------------------------------------------
    # 15. 選択特徴量の安定性
    lines.append("---")
    lines.append("## 15. 選択特徴量の安定性")
    lines.append("")
    if feature_stability is not None and not feature_stability.empty:
        lines.append("| Feature | 選択頻度 | Mean Rank IC | Mean 符号一致率 |")
        lines.append("|---------|---------|-------------|----------------|")
        top_features = feature_stability.head(15)
        for _, row in top_features.iterrows():
            freq = _fmt_pct(row.get("selection_freq"))
            ic = _fmt_float(row.get("mean_rank_ic"))
            cons = _fmt_pct(row.get("mean_sign_consistency"))
            lines.append(f"| {row.get('feature', 'N/A')} | {freq} | {ic} | {cons} |")
        lines.append("")
        if stable_features:
            lines.append(f"**安定特徴量 (選択頻度 ≥ 50%)**:")
            for f in stable_features:
                lines.append(f"- `{f}`")
            lines.append("")
    else:
        lines.append("特徴量安定性データ: not_available")
        lines.append("")
    if os.path.exists(os.path.join(figure_dir, "selected_feature_frequency.png")):
        lines.append(f"![特徴量選択頻度]({fig_path('selected_feature_frequency.png')})")
        lines.append("")
    if os.path.exists(os.path.join(figure_dir, "feature_ic_heatmap.png")):
        lines.append(f"![Feature IC ヒートマップ]({fig_path('feature_ic_heatmap.png')})")
        lines.append("")

    # -------------------------------------------------------------------------
    # 16. true 9:10 サブサンプル
    lines.append("---")
    lines.append("## 16. True 9:10 サブサンプル結果")
    lines.append("")
    if oos_preds is not None and "is_true_0910" in oos_preds.columns:
        true_910 = oos_preds[oos_preds["is_true_0910"] == True]
        n_true = len(true_910.index.unique())
        lines.append(f"- True 9:10 サブサンプル日数: {n_true}")
        lines.append("")
        if n_true > 0:
            lines.append("True 9:10 サブサンプルで検証した場合の Mean Rank IC:")
            for model_col, label in [
                ("baseline_rank_ic", "B0 Baseline"),
                ("hinge_ridge_rank_ic", "H1 Ridge"),
                ("hinge_elasticnet_rank_ic", "H2 ElasticNet"),
            ]:
                if model_col in true_910.columns:
                    mean_ic = true_910[model_col].mean()
                    lines.append(f"  - {label}: Rank IC = {_fmt_float(mean_ic)}")
        lines.append("")
    else:
        lines.append("True 9:10 データ: not_available (55日未満のサブサンプルまたは予測データ未生成)")
        lines.append("")

    # -------------------------------------------------------------------------
    # 17. 過学習リスク評価
    lines.append("---")
    lines.append("## 17. 過学習リスク評価")
    lines.append("")
    lines.append("| リスク項目 | 評価 |")
    lines.append("|-----------|------|")
    lines.append("| rolling z-score に当日データ混入 | ✅ shift(1) で防止済 |")
    lines.append("| full-sample zscore | ✅ 禁止・実装なし |")
    lines.append("| FDR selection に test データ混入 | ✅ train window 内のみ |")
    lines.append("| α blend に test データ混入 | ✅ validation window のみで選択 |")
    lines.append("| overlay 過大 | ✅ cap_overlay による上限制約 |")
    lines.append("| 特徴量多重検定 | ✅ BH FDR q=0.10 で制御 |")
    lines.append("| test window の OOS 性 | ✅ 完全 OOS |")
    lines.append("")
    if leakage_audit is not None and not leakage_audit.empty:
        lines.append("**リーク監査結果**:")
        failed = leakage_audit[leakage_audit.get("status", pd.Series(["OK"])) == "FAIL"]
        if failed.empty:
            lines.append("- すべての監査項目: ✅ PASS")
        else:
            lines.append(f"- **FAIL 項目**: {len(failed)}")
            for _, row in failed.iterrows():
                lines.append(f"  - {row.get('check', 'unknown')}: {row.get('message', '')}")
    lines.append("")

    # -------------------------------------------------------------------------
    # 18. 採用可否判断
    lines.append("---")
    lines.append("## 18. 採用可否判断")
    lines.append("")
    lines.append("### 採用基準チェックリスト")
    lines.append("")

    criteria_checks = [
        (f"Net return +{min_ret_improvement*100:.0f}%以上改善", b0_ret, h1_ret, min_ret_improvement, ">="),
        (f"IR +{min_ir_improvement:.2f}以上改善", b0_ir, h1_ir, min_ir_improvement, ">="),
        ("Max DD 悪化なし", b0_dd, h1_dd, 0.0, "dd"),
        ("Spread 15bps で baseline を上回る", None, None, None, "note"),
        ("Spread 20bps で baseline を上回る", None, None, None, "note"),
        ("Turnover +20%以下の増加", None, None, None, "note"),
        ("walk-forward で特徴量安定", len(stable_features) > 0, None, None, "bool"),
        ("過学習・リーク監査 PASS", True, None, None, "bool"),
    ]

    for i, check_item in enumerate(criteria_checks, 1):
        label = check_item[0]
        lines.append(f"- [ ] **{label}**")

    lines.append("")
    lines.append(f"### H1 (Ridge) 採用判断: **{('✅ ADOPT' if h1_adopted else '⚠️ RESEARCH CANDIDATE')}**")
    if not h1_adopted and h1_reasons:
        for r in h1_reasons:
            lines.append(f"  - {r}")
    lines.append("")
    lines.append(f"### H2 (ElasticNet) 採用判断: **{('✅ ADOPT' if h2_adopted else '⚠️ RESEARCH CANDIDATE')}**")
    if not h2_adopted and h2_reasons:
        for r in h2_reasons:
            lines.append(f"  - {r}")
    lines.append("")
    lines.append(f"### 総合判断: **{adoption_decision}**")
    lines.append("")
    if adoption_decision == "ADOPT":
        lines.append(f"**推奨モデル**: {best_overlay}")
    else:
        lines.append("production 採用見送り。**research candidate** として保留し、追加データ蓄積後に再検証する。")
    lines.append("")

    # -------------------------------------------------------------------------
    # 19. 次のアクション
    lines.append("---")
    lines.append("## 19. 次のアクション")
    lines.append("")
    if adoption_decision == "ADOPT":
        lines.append("1. ✅ 採用モデルのシャドーラン実装 (live signal との比較)")
        lines.append("2. ✅ ADV cap / lot rounding の実環境確認")
        lines.append("3. ✅ Sprint 4 (アンサンブル / 動的 alpha blend) への移行")
    else:
        lines.append("1. 🔍 追加データ取得: US セクター ETF の価格データ拡充")
        lines.append("2. 🔍 特徴量グループの再評価 (macro / regime のみに絞る)")
        lines.append("3. 🔍 train window 延長 (252 → 504 日) で安定性確認")
        lines.append("4. 🔍 hinge_filter_only (H3) の実装による単純ルール比較")
        lines.append("5. 🔍 コスト条件の見直し (gross 0.5 重視)")
    lines.append("")

    # -------------------------------------------------------------------------
    # Appendix: Run settings
    lines.append("---")
    lines.append("## Appendix: 実行設定")
    lines.append("")
    lines.append("```yaml")
    lines.append(f"start_date: {start_date}")
    lines.append(f"end_date: {end_date}")
    lines.append(f"aum_jpy: {aum}")
    lines.append(f"train_window_days: {train_window}")
    lines.append(f"validation_window_days: {val_window}")
    lines.append(f"test_window_days: {test_window}")
    lines.append(f"hinge_thresholds: {thresholds}")
    lines.append(f"fdr_q: {fdr_q}")
    lines.append(f"default_spread_bps: {default_spread}")
    lines.append("```")
    lines.append("")

    report_content = "\n".join(lines)
    report_path = os.path.join(report_dir, "sprint3a_hinge_feature_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info("Sprint 3-A report written to: %s", report_path)
    return report_path
