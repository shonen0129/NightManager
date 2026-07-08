"""src/reports/sprint3b_hinge_interaction_report.py — Sprint 3-B Report Generator.

Generates the markdown report for Sprint 3-B: Asset-specific Hinge Interaction
による限定的非線形化再検証.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _fmt(val, fmt=".4f", fallback="not_available"):
    """Format a value or return fallback string."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return fallback
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return fallback


def _pct(val, fallback="not_available"):
    """Format as percentage."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return fallback
    try:
        return f"{float(val) * 100:.2f}%"
    except (TypeError, ValueError):
        return fallback


def generate_sprint3b_report(
    artifact_dir: str,
    report_dir: str,
    figure_dir: str,
    config: dict,
    run_metadata: dict,
) -> str:
    """Generate Sprint 3-B markdown report.

    Parameters
    ----------
    artifact_dir, report_dir, figure_dir:
        Paths to artifact, report, and figure directories.
    config:
        Loaded YAML config dict.
    run_metadata:
        Dict with keys: start_date, end_date, mode.

    Returns
    -------
    str
        Path to the generated report file.
    """
    os.makedirs(report_dir, exist_ok=True)

    # Load artifacts
    model_comparison = _load_csv(artifact_dir, "model_comparison_summary.csv")
    cost_sensitivity = _load_csv(artifact_dir, "cost_sensitivity_summary.csv")
    feature_stability = _load_csv(artifact_dir, "feature_stability_summary.csv")
    rank_change_audit = _load_csv(artifact_dir, "qa/rank_change_audit.csv")
    within_date_std = _load_csv(artifact_dir, "qa/within_date_feature_std.csv")
    leakage_audit = _load_csv(artifact_dir, "qa/leakage_audit.csv")
    fdr_audit = _load_csv(artifact_dir, "qa/fdr_audit.csv")
    selected_features = _load_csv(artifact_dir, "selected_features_by_window.csv")

    start_date = run_metadata.get("start_date", "N/A")
    end_date = run_metadata.get("end_date", "N/A")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    portfolio_cfg = config.get("portfolio", {})
    aum = portfolio_cfg.get("aum_jpy", 1_000_000)
    val_cfg = config.get("validation", {})

    # --- Build report ---
    lines = []
    lines.append("# Sprint 3-B — Asset-specific Hinge Interaction による限定的非線形化再検証レポート")
    lines.append("")
    lines.append(f"**生成日時**: {now}  ")
    lines.append(f"**検証期間**: {start_date} ～ {end_date}  ")
    lines.append(f"**AUM**: ¥{aum:,}  ")
    lines.append("")
    lines.append("---")

    # 1. 概要
    lines.append("## 1. 概要")
    lines.append("")
    lines.append(
        "Sprint 3-B は Sprint 3-A で発見された「全銘柄共通特徴量がクロスセクション順位を変えない」"
        "問題を解決するため、ヒンジ特徴量を銘柄別 exposure / rolling beta / base signal と"
        "交差させた asset-specific な非線形補正を検証します。"
    )
    lines.append("")

    # 2. Sprint 3-A の問題点
    lines.append("## 2. Sprint 3-A の問題点")
    lines.append("")
    lines.append("Sprint 3-A で確認された問題点:")
    lines.append("")
    lines.append("| 問題点 | 詳細 |")
    lines.append("|--------|------|")
    lines.append("| Feature selection 0% | 175 walk-forward window 全てで選択特徴量 = 0 |")
    lines.append("| 全銘柄共通特徴量 | マクロ・レジーム系特徴量は日付単位で全銘柄共通 → cross-section 順位不変 |")
    lines.append("| within-date CS std = 0 | 特徴量が同日内で全銘柄同一のため Rank IC を計測できない |")
    lines.append("| Mean Rank IC 完全一致 | Baseline と Overlay の Mean Rank IC が同一 (0.2036) |")
    lines.append("")

    # 3. Sprint 3-B の目的
    lines.append("## 3. Sprint 3-B の目的")
    lines.append("")
    lines.append(
        "ヒンジ特徴量 × 銘柄別 exposure の交差積により、asset-specific な非線形補正を生成する。"
    )
    lines.append("")
    lines.append("$$")
    lines.append(r"\hat{\delta}^{interaction}_{j,t} = f(Hinge(z_{k,t}) \times Exposure_{j,k,t-1})")
    lines.append("$$")
    lines.append("")

    # 4. ターゲット定義
    lines.append("## 4. ターゲット定義")
    lines.append("")
    lines.append("| ターゲット | 説明 | 使用方法 |")
    lines.append("|-----------|------|----------|")
    lines.append("| `open_to_close_residual` | Open→Close 日中残差リターン | **主要ターゲット** |")
    lines.append("| `entry_to_close_residual` | Entry→Close 残差 (9:10 proxy) | サブターゲット |")
    lines.append("| `true_0910_to_close_residual` | 真の 9:10 価格からの残差 | サブサンプル報告 |")
    lines.append("| `close_to_close_return` | 前日終値→当日終値 | **参照のみ** |")
    lines.append("")

    # 5. Asset exposure 設計
    lines.append("## 5. Asset Exposure 設計")
    lines.append("")
    lines.append("### 5.1 Rolling Beta Exposure")
    lines.append("")
    lines.append("各銘柄 j とマクロ/US セクター特徴量 k について、t-1 までで rolling beta を推定:")
    lines.append("")
    lines.append("$$")
    lines.append(r"\hat{\beta}^{ridge}_{j,k,t} = \frac{\sum_{s=t-w}^{t-1} x_{k,s} y_{j,s}}{\sum_{s=t-w}^{t-1} x_{k,s}^2 + \lambda}")
    lines.append("$$")
    lines.append("")
    lines.append("- Ridge shrinkage λ = 10.0")
    lines.append("- window: 120 / 252 日")
    lines.append("- min_obs: 60 日")
    lines.append("- lag_days = 1 (look-ahead 防止)")
    lines.append("")
    lines.append("### 5.2 Static Sector Exposure")
    lines.append("")
    lines.append("JP ETF の米国セクター exposure を `configs/sector_exposure_map.yaml` で定義。")
    lines.append("")

    # 6. Rolling beta exposure
    lines.append("## 6. Rolling Beta Exposure")
    lines.append("")
    lines.append("**窓**: 120 日 / 252 日  ")
    lines.append("**最小観測数**: 60 日  ")
    lines.append("**lag_days**: 1 (t-1 まで)  ")
    lines.append("**Cross-sectional z-score**: 適用済み  ")
    lines.append("")

    # 7. Static sector map
    lines.append("## 7. Static Sector Exposure Map")
    lines.append("")
    lines.append("設定ファイル: `configs/sector_exposure_map.yaml`")
    lines.append("")
    lines.append("| セクター | 対応 US ETF |")
    lines.append("|---------|------------|")
    lines.append("| us_tech | QQQ |")
    lines.append("| us_semiconductor | SOXX |")
    lines.append("| us_energy | XLE |")
    lines.append("| us_financial | XLF |")
    lines.append("| us_industrial | XLI |")
    lines.append("| us_healthcare | XLV |")
    lines.append("| us_smallcap | IWM |")
    lines.append("")

    # 8. Hinge interaction 特徴量設計
    lines.append("## 8. Hinge Interaction 特徴量設計")
    lines.append("")
    lines.append("| グループ | 数式 | 特徴量種別 |")
    lines.append("|---------|------|----------|")
    lines.append("| G1/G2: macro × asset_beta | `hinge(z_macro) × beta_{j,macro}` | asset-specific ✅ |")
    lines.append("| G3/G4: sector × sector_exposure | `hinge(z_sector) × exposure_{j,sector}` | asset-specific ✅ |")
    lines.append("| G5/G6: regime × base_signal | `hinge(z_regime) × signal_{j,t}` | asset-specific ✅ |")
    lines.append("| G7/G8: gap asset-specific | `hinge(gap_{j,t})` | asset-specific ✅ |")
    lines.append("")
    lines.append("**ヒンジ閾値**: κ ∈ {1.0, 1.5, 2.0}  ")
    lines.append("**方向**: positive (`max(0, z - κ)`), negative (`max(0, -z - κ)`)  ")
    lines.append("")

    # 9. within-date feature std QA
    lines.append("## 9. Within-Date Cross-Sectional Std QA")
    lines.append("")
    if within_date_std is not None and not within_date_std.empty:
        use_mask = within_date_std.get("use_for_model", pd.Series([False] * len(within_date_std)))
        n_pass = int(use_mask.sum())
        n_fail = int((~use_mask).sum())
        lines.append(f"- 合格特徴量 (mean_within_date_std > 0): **{n_pass}**")
        lines.append(f"- 不合格特徴量 (全銘柄共通): **{n_fail}**")
        lines.append("")
        lines.append("詳細: `artifacts/sprint3b_hinge_interactions/qa/within_date_feature_std.csv`")
    else:
        lines.append("within-date std QA: not_available")
    lines.append("")

    # 10. FDR feature selection
    lines.append("## 10. FDR Feature Selection")
    lines.append("")
    feat_sel_cfg = config.get("feature_selection", {})
    lines.append(f"- 手法: Benjamini-Hochberg FDR (q = {feat_sel_cfg.get('fdr_q', 0.10)})")
    lines.append(f"- min_abs_rank_ic: {feat_sel_cfg.get('min_abs_rank_ic', 0.015)}")
    lines.append(f"- min_sign_consistency: {feat_sel_cfg.get('min_sign_consistency', 0.55)}")
    lines.append(f"- max_features: {feat_sel_cfg.get('max_features_after_fdr', 25)}")
    lines.append("- require_nonzero_within_date_std: true (**Sprint 3-A から追加**)")
    lines.append("")
    if fdr_audit is not None and not fdr_audit.empty:
        lines.append("| チェック | ステータス | メッセージ |")
        lines.append("|---------|-----------|----------|")
        for _, row in fdr_audit.iterrows():
            lines.append(f"| {row.get('check', '')} | {row.get('status', '')} | {row.get('message', '')} |")
    lines.append("")

    # 11. Ridge / ElasticNet overlay 仕様
    lines.append("## 11. Ridge / ElasticNet Overlay 仕様")
    lines.append("")
    model_cfg = config.get("model", {})
    lines.append("| モデル | 正則化 | ハイパーパラメータ |")
    lines.append("|--------|--------|-------------------|")
    lines.append(f"| Ridge | L2 | alpha ∈ {model_cfg.get('ridge_alpha_grid', [])} |")
    en_alphas = model_cfg.get('elasticnet_alpha_grid', [])
    en_l1s = model_cfg.get('elasticnet_l1_ratio_grid', [])
    lines.append(f"| ElasticNet | L1+L2 | alpha ∈ {en_alphas}, l1_ratio ∈ {en_l1s} |")
    lines.append("")
    lines.append("**Overlay cap**:")
    lines.append("```")
    lines.append("abs(delta_interaction) <= 0.5 * abs(mu_base)")
    lines.append("abs(delta_interaction) <= 20 bps")
    lines.append("```")
    lines.append("")

    # 12. Walk-forward 検証設計
    lines.append("## 12. Walk-forward 検証設計")
    lines.append("")
    lines.append("```")
    lines.append(f"Train window:       {val_cfg.get('train_window_days', 252)} days")
    lines.append(f"Validation window:  {val_cfg.get('validation_window_days', 63)} days")
    lines.append(f"Test window:        {val_cfg.get('test_window_days', 21)} days")
    lines.append(f"Step:               {val_cfg.get('step_days', 21)} days")
    lines.append(f"Purge:              {val_cfg.get('purge_days', 1)} days")
    lines.append("")
    lines.append("Train  → rolling zscore / rolling beta / FDR selection / model fit")
    lines.append("Val    → hyperparams + alpha blend selection")
    lines.append("Test   → OOS prediction (no future leakage)")
    lines.append("```")
    lines.append("")

    # 13. Ranking change QA
    lines.append("## 13. Ranking Change QA")
    lines.append("")
    lines.append("> **Sprint 3-A の問題点**: Overlay が実際にランキングを変えているか確認")
    lines.append("")
    if rank_change_audit is not None and not rank_change_audit.empty:
        lines.append("| モデル | Mean Spearman Rank Corr | Name Change Rate | Overlay Nonzero Rate |")
        lines.append("|--------|------------------------|-----------------|---------------------|")
        for _, row in rank_change_audit.iterrows():
            model = row.get("model", "")
            rho = _fmt(row.get("mean_spearman_rank_corr_mu_base_vs_final"), ".4f")
            name_chg = _pct(row.get("mean_selected_name_change_rate"))
            nz_rate = _pct(row.get("overlay_nonzero_rate"))
            lines.append(f"| {model} | {rho} | {name_chg} | {nz_rate} |")
    else:
        lines.append("Rank change audit: not_available")
    lines.append("")

    # 14. Rank IC / ICIR 比較
    lines.append("## 14. Rank IC / ICIR 比較")
    lines.append("")
    if model_comparison is not None and not model_comparison.empty:
        lines.append("| モデル | Mean Rank IC | ICIR | Hit Rate |")
        lines.append("|--------|-------------|------|---------  |")
        for _, row in model_comparison.iterrows():
            mn = row.get("model", "")
            ic = _fmt(row.get("mean_rank_ic"), ".4f")
            icir = _fmt(row.get("rank_icir"), ".4f")
            hr = _pct(row.get("hit_rate"))
            lines.append(f"| {mn} | {ic} | {icir} | {hr} |")
    else:
        lines.append("Rank IC 比較データ: not_available")
    lines.append("")
    lines.append("![IC時系列比較](figures/ic_timeseries_baseline_vs_interactions.png)")
    lines.append("")
    lines.append("![累積IC比較](figures/cumulative_ic_baseline_vs_interactions.png)")
    lines.append("")

    # 15. 分位リターン比較
    lines.append("## 15. 分位リターン比較")
    lines.append("")
    lines.append("![分位リターン比較](figures/quantile_returns_baseline_vs_interactions.png)")
    lines.append("")

    # 16. AUM100万円・固定スプレッド感応度
    lines.append("## 16. AUM100万円・固定スプレッド感応度")
    lines.append("")
    if cost_sensitivity is not None and not cost_sensitivity.empty and "spread_bps" in cost_sensitivity.columns:
        # Pivot by model and spread
        models_in_cost = cost_sensitivity["model"].unique().tolist() if "model" in cost_sensitivity.columns else []
        header = "| Spread (bps) | " + " | ".join(models_in_cost) + " |"
        sep = "|---|" + "---|" * len(models_in_cost)
        lines.append(header)
        lines.append(sep)
        spreads = sorted(cost_sensitivity["spread_bps"].unique())
        for s in spreads:
            row_vals = [str(int(s))]
            for m in models_in_cost:
                sub = cost_sensitivity[(cost_sensitivity["model"] == m) & (cost_sensitivity["spread_bps"] == s)]
                if sub.empty:
                    row_vals.append("not_available")
                else:
                    val = sub.iloc[0].get("annualized_net_return", np.nan)
                    row_vals.append(_pct(val))
            lines.append("| " + " | ".join(row_vals) + " |")
    else:
        lines.append("スプレッド感応度データ: not_available")
    lines.append("")
    lines.append("![スプレッド感応度](figures/spread_sensitivity_baseline_vs_interactions.png)")
    lines.append("")

    # 17. Reverse fee 感応度
    lines.append("## 17. Reverse Fee 感応度")
    lines.append("")
    lines.append("| Reverse Fee (bps/day) | B0 Net Return | Best Interaction Model |")
    lines.append("|----------------------|--------------|------------------------|")
    rev_fees = config.get("costs", {}).get("reverse_fee_bps_per_day_scenarios", [0, 5, 10, 30])
    for rf in rev_fees:
        lines.append(f"| {rf} | not_available | not_available |")
    lines.append("")
    lines.append("![Reverse Fee感応度](figures/reverse_fee_sensitivity.png)")
    lines.append("")

    # 18. Max DD / Turnover
    lines.append("## 18. Max DD / Turnover / Effective Gross")
    lines.append("")
    if model_comparison is not None and not model_comparison.empty:
        lines.append("| モデル | Max DD | Avg Turnover | Avg Effective Gross |")
        lines.append("|--------|--------|-------------|---------------------|")
        for _, row in model_comparison.iterrows():
            mn = row.get("model", "")
            dd = _pct(row.get("max_drawdown"))
            to = _fmt(row.get("turnover"), ".4f")
            eg = _fmt(row.get("effective_gross"), ".4f")
            lines.append(f"| {mn} | {dd} | {to} | {eg} |")
    else:
        lines.append("Max DD / Turnover データ: not_available")
    lines.append("")
    lines.append("![Equity Curve比較](figures/equity_curve_baseline_vs_interactions.png)")
    lines.append("")
    lines.append("![Drawdown比較](figures/drawdown_baseline_vs_interactions.png)")
    lines.append("")

    # 19. 選択特徴量の安定性
    lines.append("## 19. 選択特徴量の安定性")
    lines.append("")
    if feature_stability is not None and not feature_stability.empty:
        top = feature_stability.head(15)
        lines.append("| Feature | 選択頻度 | Mean Rank IC | Mean 符号一致率 |")
        lines.append("|---------|---------|-------------|----------------|")
        for _, row in top.iterrows():
            feat = row.get("feature", "")
            freq = _pct(row.get("selection_freq"))
            mic = _fmt(row.get("mean_rank_ic"), ".4f")
            sc = _fmt(row.get("mean_sign_consistency"), ".3f")
            lines.append(f"| {feat} | {freq} | {mic} | {sc} |")
    else:
        lines.append("特徴量安定性データ: not_available")
    lines.append("")
    lines.append("![特徴量選択頻度](figures/selected_feature_frequency.png)")
    lines.append("")
    lines.append("![Feature IC ヒートマップ](figures/feature_ic_heatmap.png)")
    lines.append("")

    # 20. True 9:10 サブサンプル
    lines.append("## 20. True 9:10 サブサンプル")
    lines.append("")
    lines.append("True 9:10 データは 55 日前後のみのため、別サブサンプルとして報告。")
    lines.append("")
    lines.append("| 指標 | B0 Baseline | Best Interaction |")
    lines.append("|------|------------|-----------------|")
    lines.append("| Rank IC (true 9:10 subsample) | not_available | not_available |")
    lines.append("")

    # 21. 過学習・リーク監査
    lines.append("## 21. 過学習・リーク監査")
    lines.append("")
    if leakage_audit is not None and not leakage_audit.empty:
        lines.append("| チェック | ステータス | 詳細 |")
        lines.append("|---------|-----------|------|")
        for _, row in leakage_audit.iterrows():
            chk = row.get("check", "")
            status = row.get("status", "")
            msg = row.get("message", "")
            lines.append(f"| {chk} | {status} | {msg} |")
    else:
        lines.append("リーク監査データ: not_available")
    lines.append("")

    # 22. Sprint 2-C / 3-A との比較
    lines.append("## 22. Sprint 2-C / 3-A との比較")
    lines.append("")
    lines.append("| Sprint | Rank IC | ICIR | Net Return | Max DD |")
    lines.append("|--------|--------|------|------------|--------|")
    lines.append("| Sprint 2-C (baseline) | 0.2036 | 10.9607 | 15.91% | -12.17% |")
    lines.append("| Sprint 3-A (hinge overlay) | 0.2036 | 10.9607 | 16.19% | -12.33% |")
    if model_comparison is not None and not model_comparison.empty:
        best_row = model_comparison[model_comparison["model"] != "net_score_ranking"]
        if not best_row.empty:
            best_row = best_row.sort_values("IR", ascending=False).iloc[0]
            mn = best_row.get("model", "Sprint 3-B best")
            ic = _fmt(best_row.get("mean_rank_ic"), ".4f")
            icir = _fmt(best_row.get("rank_icir"), ".4f")
            nr = _pct(best_row.get("annualized_net_return"))
            dd = _pct(best_row.get("max_drawdown"))
            lines.append(f"| Sprint 3-B ({mn}) | {ic} | {icir} | {nr} | {dd} |")
        else:
            lines.append("| Sprint 3-B | not_available | not_available | not_available | not_available |")
    else:
        lines.append("| Sprint 3-B | not_available | not_available | not_available | not_available |")
    lines.append("")

    # 23. 採用可否判断
    lines.append("## 23. 採用可否判断")
    lines.append("")
    lines.append("### 採用基準チェックリスト")
    lines.append("")
    adoption_criteria = config.get("evaluation", {}).get("adoption_criteria", {})

    # Get baseline and best model
    baseline_row = None
    best_interaction_row = None
    if model_comparison is not None and not model_comparison.empty:
        bl = model_comparison[model_comparison["model"] == "net_score_ranking"]
        baseline_row = bl.iloc[0] if not bl.empty else None
        others = model_comparison[model_comparison["model"] != "net_score_ranking"]
        if not others.empty:
            best_interaction_row = others.sort_values("IR", ascending=False).iloc[0]

    def check_criterion(name, passed, detail=""):
        mark = "✅" if passed else "❌"
        return f"- [{mark}] **{name}** {detail}"

    if baseline_row is not None and best_interaction_row is not None:
        base_nr = float(baseline_row.get("annualized_net_return", np.nan))
        best_nr = float(best_interaction_row.get("annualized_net_return", np.nan))
        base_ir = float(baseline_row.get("IR", np.nan))
        best_ir = float(best_interaction_row.get("IR", np.nan))
        base_dd = float(baseline_row.get("max_drawdown", np.nan))
        best_dd = float(best_interaction_row.get("max_drawdown", np.nan))

        min_nr_imp = adoption_criteria.get("min_annual_net_return_improvement", 0.03)
        min_ir_imp = adoption_criteria.get("min_ir_improvement", 0.20)
        max_dd_worse = adoption_criteria.get("max_dd_worsening_allowed", 0.0)

        nr_imp = best_nr - base_nr if not (np.isnan(best_nr) or np.isnan(base_nr)) else np.nan
        ir_imp = best_ir - base_ir if not (np.isnan(best_ir) or np.isnan(base_ir)) else np.nan
        dd_worse = (abs(best_dd) - abs(base_dd)) if not (np.isnan(best_dd) or np.isnan(base_dd)) else np.nan

        lines.append(check_criterion(
            f"Net return +{min_nr_imp*100:.0f}%以上改善",
            not np.isnan(nr_imp) and nr_imp >= min_nr_imp,
            f"実績: {_pct(nr_imp)} improvement"
        ))
        lines.append(check_criterion(
            f"IR +{min_ir_imp:.2f}以上改善",
            not np.isnan(ir_imp) and ir_imp >= min_ir_imp,
            f"実績: {_fmt(ir_imp, '.4f')} improvement"
        ))
        lines.append(check_criterion(
            "Max DD 悪化なし",
            not np.isnan(dd_worse) and dd_worse <= max_dd_worse,
            f"実績: DD worsening = {_pct(dd_worse)}"
        ))
    else:
        lines.append("- 採用基準チェック: not_available (モデル比較データなし)")

    lines.append("")

    # Rank change specific criteria
    if rank_change_audit is not None and not rank_change_audit.empty:
        best_rc = rank_change_audit.nsmallest(1, "mean_spearman_rank_corr_mu_base_vs_final") if "mean_spearman_rank_corr_mu_base_vs_final" in rank_change_audit.columns else rank_change_audit.head(1)
        if not best_rc.empty:
            max_rho_thresh = adoption_criteria.get("max_rank_corr_to_baseline", 0.995)
            min_name_chg = adoption_criteria.get("min_selected_name_change_rate", 0.05)
            min_nz = adoption_criteria.get("min_overlay_nonzero_rate", 0.10)

            rho = float(best_rc.iloc[0].get("mean_spearman_rank_corr_mu_base_vs_final", np.nan))
            nchg = float(best_rc.iloc[0].get("mean_selected_name_change_rate", np.nan))
            nz = float(best_rc.iloc[0].get("overlay_nonzero_rate", np.nan))

            lines.append(check_criterion(
                f"Spearman rank corr < {max_rho_thresh}",
                not np.isnan(rho) and rho < max_rho_thresh,
                f"実績: {_fmt(rho, '.4f')}"
            ))
            lines.append(check_criterion(
                f"Selected name change rate >= {min_name_chg*100:.0f}%",
                not np.isnan(nchg) and nchg >= min_name_chg,
                f"実績: {_pct(nchg)}"
            ))
            lines.append(check_criterion(
                f"Overlay nonzero rate >= {min_nz*100:.0f}%",
                not np.isnan(nz) and nz >= min_nz,
                f"実績: {_pct(nz)}"
            ))
    lines.append("")

    # Adoption decision
    lines.append("### 総合採用判断")
    lines.append("")
    lines.append("> **判断**: 上記チェックリストに基づき判断。"
                 "全条件を満たす場合 **PRODUCTION CANDIDATE**、"
                 "一部改善あり **RESEARCH CANDIDATE**、"
                 "改善なし **REJECT**。")
    lines.append("")

    # 24. 次のアクション
    lines.append("## 24. 次のアクション")
    lines.append("")
    lines.append("1. 🔍 rolling beta の window 延長 (252 → 504 日) で安定性確認")
    lines.append("2. 🔍 gap asset-specific 特徴量の銘柄別 z-score 計算の精緻化")
    lines.append("3. 🔍 追加銘柄データ取得による sector exposure map の精度向上")
    lines.append("4. 🔍 Sprint 4 への連携: 採用モデルを production pipeline に統合")
    lines.append("")

    # Appendix
    lines.append("---")
    lines.append("## Appendix: 実行設定")
    lines.append("")
    lines.append("```yaml")
    lines.append(f"start_date: {start_date}")
    lines.append(f"end_date: {end_date}")
    lines.append(f"aum_jpy: {aum}")
    lines.append(f"train_window_days: {val_cfg.get('train_window_days', 252)}")
    lines.append(f"validation_window_days: {val_cfg.get('validation_window_days', 63)}")
    lines.append(f"test_window_days: {val_cfg.get('test_window_days', 21)}")
    feat_cfg = config.get("features", {})
    lines.append(f"hinge_thresholds: {feat_cfg.get('hinge_thresholds', [1.0, 1.5, 2.0])}")
    lines.append(f"fdr_q: {config.get('feature_selection', {}).get('fdr_q', 0.10)}")
    lines.append(f"default_spread_bps: {config.get('costs', {}).get('default_spread_fallback_roundtrip_bps', 15)}")
    lines.append("```")
    lines.append("")

    report_text = "\n".join(lines)

    report_path = os.path.join(report_dir, "sprint3b_hinge_interaction_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info("Sprint 3-B report written to: %s", report_path)
    return report_path


def _load_csv(artifact_dir: str, filename: str) -> pd.DataFrame | None:
    """Load a CSV artifact if it exists."""
    path = os.path.join(artifact_dir, filename)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            return df if not df.empty else None
        except Exception:
            return None
    return None
