"""walk_forward_correction.py – Walk-forward validation for the nonlinear correction layer.

このスクリプトは以下を実施します:
1. 既存パイプラインから f_t, z_lin, y (OC リターン) を抽出
2. Walk-forward (拡大窓または固定窓) で補正層を学習・評価
3. 線形ベンチマーク vs 補正層の OOS ネット R/R を比較
4. 採用判定関数を呼び出し、結果をレポート出力

実行方法:
    cd src && python ../scripts/walk_forward_correction.py

オプション:
    --config         設定ファイルパス (default: configs/nonlinear_correction.yaml)
    --output_dir     レポート出力ディレクトリ
    --oos_start      OOS 開始日 (YYYY-MM-DD)
    --n_splits       Walk-forward フォールド数
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── パス設定（src/ を Python path に追加） ──────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import yaml

from config import STRATEGY_DEFAULTS, N_US_ASSETS, N_JP_ASSETS
from data_loader import download_data, preprocess_data
from domain.signals import lead_lag as signals
from domain.models.types import StrategyConfig
from domain.correction import (
    NonlinearCorrectionLayer,
    TimeSeriesPurgeSplit,
    audit_no_leak,
    CostModel,
    evaluate_correction_adoption,
)
from domain.correction.evaluation import (
    compute_net_returns,
    compute_performance_metrics,
)
from domain.correction.nonlinear_layer import GBTHyperparams
from domain.correction.feature_builder import FeatureFlags

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# データ抽出ヘルパー
# ─────────────────────────────────────────────────────────────────────────────


def extract_panel_data(
    df_exec: pd.DataFrame,
    start_date: str,
    cfg: StrategyConfig,
) -> dict:
    """既存パイプラインを実行して f_t, z_lin, y のパネルを抽出する。

    Parameters
    ----------
    df_exec : pd.DataFrame
        download_data / preprocess_data で得られる全データフレーム。
    start_date : str
        抽出開始日（例: "2015-01-01"）。
    cfg : StrategyConfig
        既存戦略設定。

    Returns
    -------
    dict with keys:
        dates       : pd.DatetimeIndex (T,)  – trade dates (JP OC return dates)
        signal_dates: pd.DatetimeIndex (T,)  – US close dates
        f_matrix    : np.ndarray (T, K)
        z_lin_matrix: np.ndarray (T, N_J)
        y_matrix    : np.ndarray (T, N_J)   – realized OC returns
    """
    from backtest.runner import run_backtest_with_config

    us_cols = [c for c in df_exec.columns if c.startswith("us_cc_")]
    jp_cc_cols = [c for c in df_exec.columns if c.startswith("jp_cc_")]
    jp_oc_cols = [c for c in df_exec.columns if c.startswith("jp_oc_")]
    jp_gap_cols = [c for c in df_exec.columns if c.startswith("jp_gap_")]
    jp_beta_cols = [c for c in df_exec.columns if c.startswith("jp_beta_")]
    topix_night_col = "topix_night_return" if "topix_night_return" in df_exec.columns else None
    all_cc_cols = us_cols + jp_cc_cols

    # Static components
    df_base = df_exec[(df_exec.index >= "2010-01-01") & (df_exec.index <= "2014-12-31")]
    base_returns = df_base[all_cc_cols].values
    _, _, C_full = signals.compute_correlation(base_returns, cfg.ewma_half_life)

    V_0 = signals.build_v3_static(N_US_ASSETS, N_JP_ASSETS, cfg.include_v4_prior)
    C_0 = signals.build_c0_from_v0(V_0, C_full)
    base_vectors = signals.build_base_vectors(N_US_ASSETS, N_JP_ASSETS)
    v1, v2 = base_vectors["v1"], base_vectors["v2"]

    all_returns = df_exec[all_cc_cols].values
    df_start = df_exec[df_exec.index >= start_date]
    start_idx = df_exec.index.get_loc(df_start.index[0])

    f_list, z_lin_list, y_list, date_list = [], [], [], []

    for i in range(start_idx, len(df_exec)):
        # y: realized OC return on trade date i (this is what we predict at i-1)
        y_row = df_exec[jp_oc_cols].iloc[i].values.astype(float)
        if np.all(np.isnan(y_row)):
            continue

        # signal at i-1 → predicts trade date i
        sig_i = i  # compute_signal uses data up to (and including) row i for the signal

        gap_arr = None
        if cfg.signal_mode == "gap_residual" and len(jp_gap_cols) == N_JP_ASSETS:
            gap_arr = np.nan_to_num(
                df_exec[jp_gap_cols].iloc[sig_i].values, nan=0.0
            ).astype(float)

        betas_t = None
        topix_night_t = None
        if jp_beta_cols and len(jp_beta_cols) == N_JP_ASSETS:
            betas_t = df_exec[jp_beta_cols].iloc[sig_i].values.astype(float)
        if topix_night_col is not None:
            topix_night_t = float(df_exec[topix_night_col].iloc[sig_i])

        sig_result = signals.compute_signal(
            all_returns,
            sig_i,
            N_US_ASSETS,
            cfg.corr_window,
            C_full,
            V_0,
            v1, v2,
            cfg.k,
            cfg.lambda_reg,
            cfg.lambda_lw,
            cfg.lw_target,
            cfg.ewma_half_life,
            v3_dynamic=False,
            gap_override=gap_arr,
            gap_open_coef=cfg.gap_open_coef,
            topix_beta_coef=cfg.topix_beta_coef,
            betas_t=betas_t,
            topix_night_t=topix_night_t,
        )

        f_list.append(sig_result["f_t"])
        z_lin_list.append(sig_result["signal"])
        y_list.append(y_row)
        date_list.append(df_exec.index[i])

    trade_dates = pd.DatetimeIndex(date_list)
    # signal_dates: US close = trade_date - 1 business day
    signal_dates = trade_dates - pd.tseries.offsets.BDay(1)

    return {
        "dates": trade_dates,
        "signal_dates": signal_dates,
        "f_matrix": np.stack(f_list, axis=0).astype(np.float32),
        "z_lin_matrix": np.stack(z_lin_list, axis=0).astype(float),
        "y_matrix": np.stack(y_list, axis=0).astype(float),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward 本体
# ─────────────────────────────────────────────────────────────────────────────


def run_walk_forward(
    panel: dict,
    layer: NonlinearCorrectionLayer,
    cv_splitter: TimeSeriesPurgeSplit,
    cost_model: CostModel,
    n_sectors: int = N_JP_ASSETS,
    q: float = 0.3,
) -> pd.DataFrame:
    """Walk-forward を実行し、各フォールドの日次リターンを返す。

    Returns
    -------
    pd.DataFrame with columns:
        linear_return, corrected_return, trade_date
    """
    from domain.signals.lead_lag import build_weights

    dates = panel["dates"]
    f_mat = panel["f_matrix"]
    z_lin_mat = panel["z_lin_matrix"]
    y_mat = panel["y_matrix"]

    results_linear = []
    results_corrected = []
    result_dates = []
    n_trials_total = 0

    for fold_i, (train_idx, val_idx) in enumerate(cv_splitter.split(dates)):
        logger.info(
            "Fold %d: train=%d..%d (%d samples), val=%d..%d (%d samples)",
            fold_i,
            train_idx[0], train_idx[-1], len(train_idx),
            val_idx[0], val_idx[-1], len(val_idx),
        )

        # ── Train ────────────────────────────────────────────────────────
        train_dates = dates[train_idx]
        signal_dates_train = panel["signal_dates"][train_idx]

        audit_no_leak(
            signal_dates=signal_dates_train,
            target_dates=train_dates,
            sample_label=f"Fold {fold_i} train",
        )

        try:
            layer.fit(
                f_matrix=f_mat[train_idx],
                z_lin_matrix=z_lin_mat[train_idx],
                y_matrix=y_mat[train_idx],
                dates=train_dates,
                signal_dates=signal_dates_train,
                n_trials_recorded=n_trials_total + 1,
            )
        except Exception as e:
            logger.warning("Fold %d training failed: %s", fold_i, e)
            continue

        n_trials_total += 1

        # ── Validate ─────────────────────────────────────────────────────
        val_dates = dates[val_idx]
        signal_dates_val = panel["signal_dates"][val_idx]
        audit_no_leak(
            signal_dates=signal_dates_val,
            target_dates=val_dates,
            sample_label=f"Fold {fold_i} val",
        )

        for t_local, t_global in enumerate(val_idx):
            f_t = f_mat[t_global]
            z_lin_t = z_lin_mat[t_global]
            y_t = y_mat[t_global]
            date_t = dates[t_global]

            # Linear-only strategy return
            w_lin = build_weights(z_lin_t, q, n_sectors, weight_mode="signal")
            ret_lin = float(np.sum(w_lin * y_t))

            # Corrected strategy return
            if layer._is_fitted:
                z_final_t = layer.predict(f_t, z_lin_t)
                w_cor = build_weights(z_final_t, q, n_sectors, weight_mode="signal")
                ret_cor = float(np.sum(w_cor * y_t))
            else:
                ret_cor = ret_lin

            results_linear.append(ret_lin)
            results_corrected.append(ret_cor)
            result_dates.append(date_t)

    if not result_dates:
        raise RuntimeError("Walk-forward produced no results. Check data and fold sizes.")

    df_results = pd.DataFrame(
        {
            "linear_return": results_linear,
            "corrected_return": results_corrected,
        },
        index=pd.DatetimeIndex(result_dates),
    )
    df_results.index.name = "trade_date"
    return df_results, n_trials_total


# ─────────────────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward validation for the nonlinear correction layer"
    )
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIR.parent / "configs" / "nonlinear_correction.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument("--output_dir", default=None, help="Output directory for reports")
    parser.add_argument("--oos_start", default=None, help="OOS start date (YYYY-MM-DD)")
    parser.add_argument("--n_splits", type=int, default=None, help="Walk-forward splits")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    wf_cfg = config.get("walk_forward", {})
    start_date = wf_cfg.get("start_date", "2015-01-01")
    oos_start = args.oos_start or wf_cfg.get("oos_start_date", "2020-01-01")
    output_dir = Path(args.output_dir or wf_cfg.get("output_dir", "results/correction_walkforward"))
    output_dir.mkdir(parents=True, exist_ok=True)

    cv_cfg = config.get("cv", {})
    n_splits = args.n_splits or cv_cfg.get("n_splits", 5)

    cost_cfg = config.get("cost_model", {})
    cost_model = CostModel(
        commission_rate=cost_cfg.get("commission_rate", 0.0005),
        spread_rate=cost_cfg.get("spread_rate", 0.0003),
        impact_rate=cost_cfg.get("impact_rate", 0.0002),
    )

    # ── Load data ────────────────────────────────────────────────────────
    logger.info("Downloading and preprocessing data...")
    data = download_data(beta_window=STRATEGY_DEFAULTS["beta_window"])
    df_exec = preprocess_data(data, beta_window=STRATEGY_DEFAULTS["beta_window"])

    cfg = StrategyConfig(
        k=STRATEGY_DEFAULTS["K"],
        lambda_reg=STRATEGY_DEFAULTS["lambda_reg"],
        q=STRATEGY_DEFAULTS["q"],
        weight_mode=STRATEGY_DEFAULTS["weight_mode"],
        dispersion_filter=STRATEGY_DEFAULTS["dispersion_filter"],
        ewma_half_life=STRATEGY_DEFAULTS["ewma_half_life"],
        lambda_lw=STRATEGY_DEFAULTS["lambda_lw"],
        lw_target=STRATEGY_DEFAULTS["lw_target"],
        corr_window=STRATEGY_DEFAULTS["corr_window"],
        include_v4_prior=STRATEGY_DEFAULTS["include_v4_prior"],
        signal_mode=STRATEGY_DEFAULTS["signal_mode"],
        gap_open_coef=STRATEGY_DEFAULTS["gap_open_coef"],
        topix_beta_coef=STRATEGY_DEFAULTS["topix_beta_coef"],
        beta_window=STRATEGY_DEFAULTS["beta_window"],
    )

    # ── Extract panel ────────────────────────────────────────────────────
    logger.info("Extracting f_t / z_lin / y panel from %s ...", start_date)
    panel = extract_panel_data(df_exec, start_date=start_date, cfg=cfg)
    logger.info("Panel shape: f=%s, z_lin=%s, y=%s",
                panel["f_matrix"].shape, panel["z_lin_matrix"].shape, panel["y_matrix"].shape)

    # ── Build correction layer ───────────────────────────────────────────
    layer = NonlinearCorrectionLayer.from_config(config)
    logger.info("Layer config: %s", layer)

    # ── CV splitter ──────────────────────────────────────────────────────
    cv_splitter = TimeSeriesPurgeSplit(
        n_splits=n_splits,
        train_size=cv_cfg.get("train_size", None),
        gap_purge=cv_cfg.get("gap_purge", 1),
        embargo=cv_cfg.get("embargo", 5),
        min_train_size=cv_cfg.get("min_train_size", 120),
    )

    # ── Walk-forward ─────────────────────────────────────────────────────
    logger.info("Starting walk-forward validation...")
    df_wf, n_trials = run_walk_forward(
        panel=panel,
        layer=layer,
        cv_splitter=cv_splitter,
        cost_model=cost_model,
        n_sectors=N_JP_ASSETS,
        q=cfg.q,
    )

    # ── OOS filtering ────────────────────────────────────────────────────
    df_oos = df_wf[df_wf.index >= oos_start]
    logger.info("OOS period: %s to %s (%d days)", df_oos.index.min().date(),
                df_oos.index.max().date(), len(df_oos))

    # ── Net returns ──────────────────────────────────────────────────────
    net_linear = compute_net_returns(df_oos["linear_return"], cost_model=cost_model)
    net_corrected = compute_net_returns(df_oos["corrected_return"], cost_model=cost_model)

    # ── Compute metrics ──────────────────────────────────────────────────
    sig_thresh = config.get("thresholds", {}).get("significance_level", 0.05)
    m_lin = compute_performance_metrics(net_linear, label="Linear Baseline", n_trials=1)
    m_cor = compute_performance_metrics(net_corrected, label="Linear + GBT Correction",
                                        n_trials=n_trials)

    # ── Adoption gate ────────────────────────────────────────────────────
    decision = evaluate_correction_adoption(
        linear_metrics=m_lin,
        corrected_metrics=m_cor,
        significance_level=sig_thresh,
    )

    # ── Output ───────────────────────────────────────────────────────────
    print(str(decision))

    # Save returns CSV
    df_wf.to_csv(output_dir / "walk_forward_returns.csv", encoding="utf-8-sig")
    df_oos["net_linear"] = net_linear
    df_oos["net_corrected"] = net_corrected
    df_oos.to_csv(output_dir / "oos_net_returns.csv", encoding="utf-8-sig")

    # Save metrics JSON
    import json
    import dataclasses

    metrics_dict = {
        "linear": {k: (v if not isinstance(v, float) or (v == v) else None)
                   for k, v in m_lin.__dict__.items() if k != "extra"},
        "corrected": {k: (v if not isinstance(v, float) or (v == v) else None)
                      for k, v in m_cor.__dict__.items() if k != "extra"},
        "decision": decision.decision,
        "reason": decision.reason,
        "n_trials": n_trials,
        "oos_start": oos_start,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Results saved to %s", output_dir)

    # Plot cumulative returns
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        (1 + net_linear).cumprod().plot(ax=axes[0], label="Linear", linewidth=1.2)
        (1 + net_corrected).cumprod().plot(ax=axes[0], label="Linear + GBT", linewidth=1.2)
        axes[0].set_title("OOS Cumulative Net Returns")
        axes[0].set_ylabel("Wealth")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        def _dd(s): return ((1 + s).cumprod() / (1 + s).cumprod().cummax()) - 1

        _dd(net_linear).plot(ax=axes[1], label="Linear", linewidth=1.0)
        _dd(net_corrected).plot(ax=axes[1], label="Linear + GBT", linewidth=1.0)
        axes[1].set_title("Drawdown")
        axes[1].set_ylabel("Drawdown")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "oos_comparison.png", dpi=150)
        plt.close()
        logger.info("Chart saved: %s/oos_comparison.png", output_dir)
    except Exception as e:
        logger.warning("Chart generation failed: %s", e)


if __name__ == "__main__":
    main()
