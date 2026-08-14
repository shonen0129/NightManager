"""Pydantic schemas for validated configuration variables.

Single source of truth for all configuration types.
All modules should import StrategyConfig / RiskConfig from here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field, model_validator

from leadlag.config.paths import live, results


class StrategyConfig(BaseModel):
    """Strategy parameters validation schema.

    Covers model hyperparameters, signal construction settings,
    portfolio construction settings, and execution-layer parameters
    (start_date, risk thresholds) that are used by production runners.
    """
    model_config = {"frozen": True}

    model_name: str = Field(default="sector_relative_ensemble", description="モデル識別名")
    k: int = Field(default=6, ge=1, description="固有ベクトル空間の次元数 K")
    lambda_reg: float = Field(default=0.75, ge=0.0, le=1.0, description="相関行列レギュラリゼーション強度（第2段階: priorへの縮小）")
    q: float = Field(default=0.3, ge=0.0, le=1.0, description="ロング/ショート選択比率 (各サイド q×N_JP 銘柄)")
    weight_mode: str = Field(default="signal", description="ウェイト構築モード (signal | equal | rank)")
    dispersion_filter: bool = Field(default=False, description="分散フィルター有効フラグ")
    dispersion_metric: str = Field(default="long_short_mean_gap", description="分散指標の種類")
    v3_mode: str = Field(default="static", description="事前部分空間モード (static | dynamic)")
    ewma_half_life: int = Field(default=45, ge=1, description="EWMA 半減期 (日数)")
    lambda_lw: float = Field(default=0.5, ge=0.0, le=1.0, description="Ledoit-Wolf 縮小強度（第1段階: equicorrelationへの縮小）")
    lw_target: str = Field(default="equicorrelation", description="Ledoit-Wolf ターゲット行列")
    min_raw_weight: float = Field(default=0.0, ge=0.0, le=1.0, description="生の相関行列が最終結果に占める最低重量（ガードレール、0=無効）")
    corr_window: int = Field(default=60, ge=1, description="相関計算ローリング窓 (日数)")
    include_v4_prior: bool = Field(default=True, description="v4 事前ベクトル (Market-Factor) を含めるか")
    signal_mode: str = Field(default="gap_residual", description="シグナルモード (gap_residual | raw)")
    gap_open_coef: float = Field(default=0.70, description="ギャップ調整係数 (idiosyncratic ギャップへの感応度)")
    topix_beta_coef: float = Field(
        default=0.6,
        description=(
            "TOPIX ベータ係数（ギャップ残差補正時のTOPIXベータ係数）。"
            "バックテスト検証により 0.6 が 1.20 より優れたパフォーマンスを示すため、"
            "0.6 を正本として採用。"
        ),
    )
    beta_window: int = Field(default=60, ge=1, description="ローリング OLS ベータ推定窓 (日数)")
    beta_ewma_halflife: float | None = Field(
        default=None,
        ge=1.0,
        description="EWMA 加重ベータ推定の半減期 (日数)。None の場合は等重ローリング推定 (従来動作)",
    )
    beta_shrinkage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="ベータの 1.0 へのベイズ縮小強度 (0=縮小なし, 1=完全に1.0)",
    )
    beta_winsor_sigma: float | None = Field(
        default=None,
        ge=1.0,
        description="ベータ推定前のローリングウィンソライズ sigma 数 (例: 3.0)。None=ウィンソライズなし",
    )
    gamma: float = Field(default=0.5, description="US 残差化ブレンド係数")
    slippage_bps: float = Field(default=5.0, ge=0.0, description="片道スリッページ (basis points)")
    vol_adjusted_target: bool = Field(default=True, description="ボラティリティ調整ターゲット有効フラグ")

    # --- Copula correlation blending ---
    copula_enabled: bool = Field(default=False, description="t-copula相関ブレンド有効フラグ")
    copula_blend_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Copula相関の固定ブレンド重み (0=Pearsonのみ, 1=Copulaのみ)")
    copula_dynamic_blend: bool = Field(default=True, description="ストレス期にCopula重みを動的増加")
    copula_stress_threshold: float = Field(default=1.5, ge=1.0, description="ストレス判定のボラティリティ比率閾値")
    copula_nu_init: float = Field(default=5.0, ge=2.1, le=30.0, description="t-copula自由度の初期値")
    copula_marginal_method: str = Field(default="empirical", description="周辺分布推定法 (empirical | skewt)")

    # --- Covariance-aware weight optimization ---
    minvar_enabled: bool = Field(default=False, description="共分散対応最小分散weight最適化有効フラグ")
    minvar_alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="最小分散ブレンド係数 (0=signal比例, 1=純最小分散)")

    overnight_alpha_long: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="ロングポジションのオーバーナイト持ち越し比率 (0=日次全額決済, 1=全額持ち越し)",
    )
    overnight_alpha_short: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="ショートポジションのオーバーナイト持ち越し比率 (0=日次全額決済, 1=全額持ち越し)",
    )
    buy_interest_annual: float = Field(default=0.025, ge=0.0, description="ロング資金調達コスト (年率)")
    borrow_fee_annual: float = Field(default=0.0115, ge=0.0, description="空売り貸株コスト (年率)")
    reverse_fee_bps: float = Field(default=2.0, ge=0.0, description="逆日歩 (bps/day, ショート側のみ)")
    side_leverage: float = Field(
        default=1.5,
        ge=0.0,
        description="信用取引のロング+ショート合計レバレッジ倍率（gross notional = weights × side_leverage）",
    )

    # Production runner parameters (start_date + risk thresholds)
    # NOTE: risk thresholds are duplicated here for backward compat with production runners
    # that pass a single StrategyConfig to both strategy and risk layers.
    # The canonical risk-only type is RiskConfig; use AppConfig.risk in new code.
    start_date: str = Field(default="2015-01-05", description="バックテスト開始日")
    var_confidence: float = Field(default=0.99, ge=0.0, le=1.0, description="VaR 信頼水準")
    var_window: int = Field(default=250, ge=1, description="VaR/ES 計算ウィンドウ (日数)")
    var_method: str = Field(default="historical", description="VaR 計算手法: historical or cornish_fisher")
    var_warning: float = Field(default=0.02, ge=0.0, le=1.0, description="VaR 警告閾値")
    var_stop: float = Field(default=0.03, ge=0.0, le=1.0, description="VaR 停止閾値")
    es_warning: float = Field(default=0.025, ge=0.0, le=1.0, description="ES 警告閾値")
    es_stop: float = Field(default=0.04, ge=0.0, le=1.0, description="ES 停止閾値")
    daily_loss_warning: float = Field(default=0.015, ge=0.0, le=1.0, description="日次損失警告閾値")
    daily_loss_stop: float = Field(default=0.025, ge=0.0, le=1.0, description="日次損失停止閾値")
    monthly_loss_stop: float = Field(default=0.05, ge=0.0, le=1.0, description="月次損失停止閾値")
    max_net_exposure: float = Field(default=0.05, ge=0.0, le=1.0, description="最大ネット露出比率")
    max_gross_exposure: float = Field(default=2.0, ge=0.0, description="最大グロス露出比率")


class RiskConfig(BaseModel):
    """Risk management parameters validation schema.

    Canonical risk-only configuration type.
    Use AppConfig.risk when constructing the full application config.
    """
    model_config = {"frozen": True}

    var_confidence: float = Field(default=0.99, ge=0.0, le=1.0, description="VaR 信頼水準")
    var_window: int = Field(default=250, ge=1, description="VaR/ES 計算ウィンドウ (日数)")
    var_method: str = Field(default="historical", description="VaR 計算手法: historical or cornish_fisher")
    var_warning: float = Field(default=0.02, ge=0.0, le=1.0, description="VaR 警告閾値")
    var_stop: float = Field(default=0.03, ge=0.0, le=1.0, description="VaR 停止閾値")
    es_warning: float = Field(default=0.025, ge=0.0, le=1.0, description="ES 警告閾値")
    es_stop: float = Field(default=0.04, ge=0.0, le=1.0, description="ES 停止閾値")
    daily_loss_warning: float = Field(default=0.015, ge=0.0, le=1.0, description="日次損失警告閾値")
    daily_loss_stop: float = Field(default=0.025, ge=0.0, le=1.0, description="日次損失停止閾値")
    monthly_loss_stop: float = Field(default=0.05, ge=0.0, le=1.0, description="月次損失停止閾値")
    max_net_exposure: float = Field(default=0.05, ge=0.0, le=1.0, description="最大ネット露出比率")
    max_gross_exposure: float = Field(default=2.0, ge=0.0, description="最大グロス露出比率")


class KabuApiConfig(BaseModel):
    """kabuステーション API configuration."""
    model_config = {"frozen": True}

    api_url: str = Field(default="http://localhost:18080/kabusapi", description="API ベース URL")
    api_token: str = Field(default="", description="API トークン")
    api_password: str = Field(default="", description="API パスワード（トークン自動更新用）")
    request_timeout: int = Field(default=10, ge=1, description="リクエストタイムアウト (秒)")
    margin_trade_type: int = Field(default=3, description="信用取引区分 (1=制度, 2=一般, 3=日計)")
    account_type: int = Field(default=4, description="口座区分 (2=一般, 4=特定, 12=法人)")


class TachibanaApiConfig(BaseModel):
    """立花証券 API configuration."""
    model_config = {"frozen": True}

    api_url: str = Field(default="https://kabuka.e-shiten.jp/e_api_v4r9", description="立花API ベース URL")
    auth_id: str = Field(default="", description="認証ID (sAuthId)")
    private_key_path: str = Field(default="", description="秘密鍵ファイルパス (.pem)")
    second_password: str = Field(default="", description="第二パスワード (取引パスワード)")
    request_timeout: int = Field(default=10, ge=1, description="リクエストタイムアウト (秒)")
    margin_trade_type: int = Field(default=3, description="信用取引区分 (1=制度, 2=一般, 3=日計)")
    account_type: int = Field(default=4, description="口座区分 (2=一般, 4=特定, 12=法人)")


class MLOrderOverlayConfig(BaseModel):
    """ML order overlay configuration."""
    model_config = {"frozen": True}

    enabled: bool = Field(default=False, description="ML order overlay を有効化")
    model_dir: str = Field(default="", description="overlay モデルディレクトリ")
    use_ticker: bool = Field(default=True, description="ticker 特徴量を使用")
    use_classification: bool = Field(default=False, description="分類モデルを使用")
    per_ticker_interactions: bool = Field(default=True, description="ticker × score / gap 交互作用を使用")


class BLPXConfig(BaseModel):
    """BLPX signal model parameters.

    These values are consumed by ``leadlag.models.blpx.ProductionBLPXModel``.
    Defaults are aligned with ``configs/production/production.yaml``.
    """
    model_config = {"frozen": True, "extra": "forbid"}

    # Identity
    param_set: str = Field(default="default", description="BLPX parameter set name")
    model_name: str = Field(default="ProductionBLPXModel", description="Model identifier for diagnostics")

    # Universe / meta
    n_u: int = Field(default=15, ge=1, description="Number of US tickers")
    n_j: int = Field(default=17, ge=1, description="Number of JP tickers")

    # PCA / core
    k: int = Field(default=6, ge=1, description="Number of retained eigenvectors")
    q: float = Field(default=0.3, ge=0.0, le=1.0, description="Long/short selection fraction")
    weight_mode: str = Field(default="signal", description="Weight construction mode")
    normalization: str = Field(default="zscore", description="Signal normalization method")
    rank: str = Field(default="full", description="Correlation rank scheme")

    # Window / EWMA
    ewma_halflife: int = Field(default=120, ge=1, description="Baseline correlation EWMA half-life")
    blp_ewma_halflife: float = Field(default=120.0, ge=1.0, description="BLP window correlation EWMA half-life")
    blp_window: int = Field(default=504, ge=1, description="BLP rolling window (days)")
    corr_window: int = Field(default=60, ge=1, description="PCA correlation rolling window")
    corr_min_periods: int = Field(default=60, ge=1, description="Minimum periods for correlation")
    beta_window: int = Field(default=60, ge=1, description="TOPIX residualization beta window")
    beta_floor: float = Field(default=0.0, ge=0.0, description="Minimum beta value")
    prior_variant: str | None = Field(default=None, description="Prior variant override")

    # Shrinkage
    lambda_reg: float = Field(default=0.75, ge=0.0, le=1.0)
    lambda_lw: float = Field(default=0.5, ge=0.0, le=1.0)
    lw_target: str = Field(default="equicorrelation")
    include_v4_prior: bool = Field(default=True)
    min_raw_weight: float = Field(default=0.0, ge=0.0, le=1.0)

    # Ensemble / signal component weights
    raw_pca_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    residual_pca_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    raw_blpx_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    residual_blpx_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    p4_weight: float = Field(default=0.0, ge=0.0, le=1.0)

    # BLP structured prior / Tikhonov
    rho: float = Field(default=0.01, ge=0.0)
    alpha_xx: float = Field(default=0.20, ge=0.0, le=1.0)
    alpha_yx: float = Field(default=0.15, ge=0.0, le=1.0)
    alpha_yy: float = Field(default=0.50, ge=0.0, le=1.0)
    lambda_pca: float = Field(default=0.10, ge=0.0)
    lambda_sector: float = Field(default=0.60, ge=0.0)
    beta_conf: float = Field(default=0.25, ge=0.0)
    frobenius_scale_priors: bool = Field(default=False)
    winsor_sigma: float | None = Field(default=3.0, ge=0.0)
    sector_eta: float = Field(default=0.5, ge=0.0)
    sector_gamma: float = Field(default=4.0, ge=0.0)

    # Target / gap adjustment
    target: str = Field(default="topix_residual")
    use_raw_target: bool = Field(default=False)
    gap_open_coef: float = Field(default=0.70, ge=0.0)
    gap_open_coef_neg: float | None = Field(default=0.60, ge=0.0)
    topix_beta_coef: float = Field(default=0.60, ge=0.0)
    topix_beta_coef_neg: float | None = Field(default=0.60, ge=0.0)
    vol_adjusted_target: bool = Field(default=False)
    execution_target_cost_adjustment: str = Field(default="none")

    # Asymmetry
    asymmetry_mode: str = Field(default="scalar")
    asymmetry_delta: float = Field(default=0.30, ge=0.0)
    asymmetry_post_gap_delta: float = Field(default=0.0, ge=0.0)
    asymmetry_post_gap_mode: str = Field(default="signal_split")

    # Macro
    macro_confidence_enabled: bool = Field(default=True)
    macro_kappa_enabled: bool = Field(default=True)
    macro_direction_enabled: bool = Field(default=True)
    macro_kappas: tuple[float, float, float] = Field(default=(3.0, 0.5, 0.5))
    macro_surprise_halflife_mean: float = Field(default=20.0, ge=1.0)
    macro_surprise_halflife_vol: float = Field(default=60.0, ge=1.0)
    macro_sigma_yy_inflation_enabled: bool = Field(default=False)

    # Copula
    copula_enabled: bool = Field(default=True)
    copula_blend_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    copula_dynamic_blend: bool = Field(default=True)
    copula_stress_threshold: float = Field(default=1.5, ge=0.0)
    copula_nu_init: float = Field(default=5.0, ge=2.0)
    copula_marginal_method: str = Field(default="empirical")

    # Min-variance (used by BLPX build_weights path)
    minvar_enabled: bool = Field(default=False)
    minvar_alpha: float = Field(default=0.5, ge=0.0, le=1.0)


class CostConfig(BaseModel):
    """Cost and financing parameters shared by production and backtest."""
    model_config = {"frozen": True, "extra": "forbid"}

    cost_bps_per_gross: float = Field(default=10.0, ge=0.0)
    slippage_bps_per_side: float = Field(default=5.0, ge=0.0)
    overnight_alpha_long: float = Field(default=0.75, ge=0.0, le=1.0)
    overnight_alpha_short: float = Field(default=0.5, ge=0.0, le=1.0)
    buy_interest_annual: float = Field(default=0.025, ge=0.0)
    borrow_fee_annual: float = Field(default=0.0115, ge=0.0)
    reverse_fee_bps: float = Field(default=2.0, ge=0.0)
    side_leverage: float = Field(default=1.5, ge=0.0)


_BLPX_PREFIX = "blpx_"
_COSTS_FLAT_FIELDS = {
    "slippage_bps_per_side",
    "cost_bps_per_gross",
    "overnight_alpha_long",
    "overnight_alpha_short",
    "buy_interest_annual",
    "borrow_fee_annual",
    "reverse_fee_bps",
    "side_leverage",
}


def _map_flat_to_nested(raw: dict[str, Any]) -> dict[str, Any]:
    """Move flat ``blpx_*`` and known cost keys into nested ``blpx``/``costs`` dicts.

    Flat keys take precedence over values inside the nested sections.
    This makes both the new flat YAML layout and the existing nested YAML
    layout validate through the same Pydantic schema.
    """
    out = dict(raw)
    blpx = dict(out.pop("blpx", {}) or {})
    costs = dict(out.pop("costs", {}) or {})

    for k, v in list(out.items()):
        if k.startswith(_BLPX_PREFIX):
            blpx[k[len(_BLPX_PREFIX):]] = v
            del out[k]
        elif k in _COSTS_FLAT_FIELDS:
            costs[k] = v
            del out[k]

    if blpx:
        out["blpx"] = blpx
    if costs:
        out["costs"] = costs
    return out


class ProductionV2RunConfig(BaseModel):
    """Runtime parameters for the v2 daily production pipeline.

    Parsed from the YAML ``portfolio:``, ``gross_scaling:``, ``costs:``,
    ``fallback:``, ``blpx:``, ``residualization:``, ``features:``,
    ``ml_order_overlay:``, ``gap_distribution:``, and other V2 sections
    via ``_flatten_nested_yaml``.  Acts as the single source of truth for
    all v2 pipeline constants.
    """
    model_config = {"frozen": True, "extra": "forbid"}

    # --- Portfolio construction ---
    long_count: int = Field(default=5, ge=1, description="ロング選択銘柄数")
    short_count: int = Field(default=5, ge=1, description="ショート選択銘柄数")
    baseline_gross: float = Field(default=2.0, ge=0.0, description="pre-gross 基準グロスエクスポージャー")

    # --- Cost model (flat alias, canonical value lives in ``costs``) ---
    cost_bps_per_gross: float = Field(
        default=10.0, ge=0.0,
        description="ex-ante コスト (bps/unit gross)。IR 計算のみに使用。実取引コストではない。"
    )

    # --- RuleD dynamic gross scaling ---
    pit_rolling_window: int = Field(default=252, ge=1, description="PIT 三分位ビニング用ローリング窓（営業日）")
    tertile_low_pct: float = Field(default=33.3333, ge=0.0, le=100.0, description="低閾値パーセンタイル")
    tertile_high_pct: float = Field(default=66.6667, ge=0.0, le=100.0, description="高閾値パーセンタイル")
    mult_low: float = Field(default=0.75, ge=0.0, le=1.0, description="Low ビン グロス乗数")
    mult_mid: float = Field(default=1.00, ge=0.0, le=1.0, description="Medium ビン グロス乗数")
    mult_high: float = Field(default=1.00, ge=0.0, le=1.0, description="High ビン グロス乗数")
    fallback_multiplier: float = Field(default=1.00, ge=0.0, le=1.0, description="PIT 履歴不足時のフォールバック乗数")

    # --- Fallback behavior ---
    fallback_on_gap_data_missing: bool = Field(default=True, description="gap data 欠損時に flat position (w_final=0) を返す")
    fallback_on_audit_failure: bool = Field(default=True, description="数値監査失敗時に flat position (w_final=0) を返す")
    ondemand_fallback_enabled: bool = Field(default=True, description="gap ファイル不在時に on-demand BLPX 計算でフォールバックする")
    shadow_ondemand_validation: bool = Field(default=False, description="file cache 読み込み時に on-demand 計算と shadow 比較を実行する")

    # --- Phase 2A: Multi-Horizon Signal Blending ---
    mh_blend_enabled: bool = Field(default=False, description="マルチホライズンブレンド有効フラグ")
    mh_horizons: tuple[int, ...] = Field(default=(1, 3, 5), description="ブレンド対象ホライズン（日）")
    mh_weights: tuple[float, ...] = Field(default=(0.8, 0.1, 0.1), description="各ホライズンのブレンド重み")
    mh_mu_file_pattern_h: str = Field(
        default="matrices/mu_gap_h{h}_{date}.npy",
        description="マルチホライズンブレンド用 mu_gap ファイルパターン",
    )
    mh_omega_file_pattern_h: str = Field(
        default="matrices/omega_gap_h{h}_{date}.npy",
        description="マルチホライズンブレンド用 omega_gap ファイルパターン",
    )

    # --- Phase 2D: Cross-Sectional Rank Reversal Overlay ---
    cs_overlay_enabled: bool = Field(default=False, description="CS特徴量オーバーレイ有効フラグ")
    cs_overlay_weight: float = Field(default=0.05, ge=0.0, description="ランク反転オーバーレイ重み")
    cs_rank_reversal_file_pattern: str = Field(
        default="matrices/rank_reversal_{date}.npy",
        description="ランク反転オーバーレイ用ファイルパターン",
    )

    # --- MinVar weight optimization ---
    minvar_enabled: bool = Field(default=False, description="共分散対応最小分散weight最適化有効フラグ")
    minvar_alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="最小分散ブレンド係数 (0=signal比例, 1=純最小分散)")

    # --- Macro Factor-Kappa (Omega_gap inflation) ---
    macro_kappa_enabled: bool = Field(default=False, description="マクロサプライズによるOmega_gap膨張有効フラグ")
    macro_kappas: tuple[float, float, float] = Field(default=(3.0, 0.5, 0.5), description="因子別kappa [USDJPY, CLF, TNX]")
    macro_surprise_halflife_mean: float = Field(default=20.0, ge=1.0, description="EWMA平均推定半減期")
    macro_surprise_halflife_vol: float = Field(default=60.0, ge=1.0, description="EWMAボラティリティ推定半減期")

    # --- Macro Directional Adjustment ---
    macro_direction_enabled: bool = Field(default=False, description="符号付きマクロサプライズによる方向調整有効フラグ")

    # --- Gap distribution / ranking ---
    gap_input_dir: Path | None = Field(default=None, description="gap 調整済み分布ディレクトリ")
    mu_file_pattern: str = Field(default="matrices/mu_gap_{date}.npy", description="mu_gap ファイルパターン")
    omega_file_pattern: str = Field(default="matrices/omega_gap_{date}.npy", description="omega_gap ファイルパターン")
    sigma_floor: float = Field(default=1.0e-6, gt=0.0, description="mu_over_sigma ゼロ除算防止フロア")

    # --- Residualization ---
    residualization_enabled_for_p3: bool = Field(default=True, description="JP residualization (P3) 有効フラグ")
    residualization_beta_window: int = Field(default=60, ge=1, description="TOPIX residualization beta 窓")
    residualization_beta_winsor_sigma: float = Field(default=3.0, ge=0.0, description="beta 推定前のウィンソライズ sigma")
    residualization_beta_shrinkage: float = Field(default=0.05, ge=0.0, le=1.0, description="beta の 1.0 へのベイズ縮小強度")

    # --- Fractional Differentiation ---
    frac_diff_enabled: bool = Field(default=False, description="分数階差分有効フラグ")
    frac_diff_d: float = Field(default=0.1, ge=0.0, le=1.0, description="分数階差分次数")
    frac_diff_threshold: float = Field(default=1.0e-5, gt=0.0, description="二項展開の重み打ち切り閾値")
    frac_diff_window: int = Field(default=100, ge=1, description="分数階差分の最大ルックバック")
    frac_diff_normalize: str | None = Field(default=None, description="重み正規化方法")

    # --- ML Order Overlay ---
    ml_overlay_enabled: bool = Field(default=False, description="ML order overlay を有効化")
    ml_overlay_model_dir: str = Field(default="", description="overlay モデルディレクトリ")
    ml_overlay_use_ticker: bool = Field(default=True, description="ticker 特徴量を使用")
    ml_overlay_use_classification: bool = Field(default=False, description="分類モデルを使用")
    ml_overlay_per_ticker_interactions: bool = Field(default=True, description="ticker × score / gap 交互作用を使用")

    # --- Nested BLPX and cost sub-models ---
    blpx: BLPXConfig = Field(default_factory=BLPXConfig, description="BLPX シグナルパラメータ")
    costs: CostConfig = Field(default_factory=CostConfig, description="コスト・ファイナンスパラメータ")

    #: Top-level YAML sections that this validator flattens into flat fields and sub-models.
    _NESTED_SECTIONS: ClassVar[tuple[str, ...]] = (
        "portfolio", "gross_scaling", "costs", "fallback",
        "multi_horizon_blend", "cs_feature_overlay", "blpx",
        "residualization", "features", "ml_order_overlay",
        "gap_distribution", "execution", "ranking", "signal_components",
    )

    @model_validator(mode="before")
    @classmethod
    def _flatten_nested_yaml(cls, data: Any) -> Any:
        """Flatten the nested production YAML structure into flat fields and sub-models.

        Only keys explicitly present in the YAML are mapped; everything else
        falls back to the ``Field`` defaults above (single source of truth).
        Explicit top-level (flat) keys take precedence over nested sections.
        """
        if not isinstance(data, dict):
            return data

        data = _map_flat_to_nested(data)

        portfolio = data.get("portfolio") or {}
        gross_scaling = data.get("gross_scaling") or {}
        costs = data.get("costs") or {}
        fallback = data.get("fallback") or {}
        mh = data.get("multi_horizon_blend") or {}
        cs = data.get("cs_feature_overlay") or {}
        blpx = data.get("blpx") or {}
        residualization = data.get("residualization") or {}
        features = data.get("features") or {}
        frac_diff = features.get("fractional_diff") or {}
        ml = data.get("ml_order_overlay") or {}
        gap_dist = data.get("gap_distribution") or {}
        execution = data.get("execution") or {}
        ranking = data.get("ranking") or {}
        signal_components = data.get("signal_components") or {}
        multipliers = (gross_scaling.get("multipliers") or {}) if isinstance(gross_scaling, dict) else {}

        if not any(
            [
                portfolio, gross_scaling, costs, fallback, mh, cs, blpx,
                residualization, features, ml, gap_dist, execution, ranking,
                signal_components,
            ]
        ):
            return data

        flat = {k: v for k, v in data.items() if k not in cls._NESTED_SECTIONS}

        # Build nested blpx sub-model (do not emit flat blpx_* aliases)
        blpx_cfg = dict(blpx) if isinstance(blpx, dict) else {}
        if residualization and isinstance(residualization, dict):
            if residualization.get("beta_window") is not None and "beta_window" not in blpx_cfg:
                blpx_cfg["beta_window"] = residualization["beta_window"]
            if residualization.get("topix_beta_coef") is not None and "topix_beta_coef" not in blpx_cfg:
                blpx_cfg["topix_beta_coef"] = residualization["topix_beta_coef"]
        if signal_components and isinstance(signal_components, dict):
            for comp, weight_key in (
                ("raw_pca", "raw_pca_weight"),
                ("residual_pca", "residual_pca_weight"),
                ("raw_blpx", "raw_blpx_weight"),
                ("residual_blpx", "residual_blpx_weight"),
            ):
                comp_cfg = signal_components.get(comp)
                if isinstance(comp_cfg, dict) and weight_key not in blpx_cfg:
                    enabled = bool(comp_cfg.get("enabled", False))
                    blpx_cfg[weight_key] = float(comp_cfg.get("weight", 1.0 if enabled else 0.0)) if enabled else 0.0
        if blpx_cfg and "blpx" not in flat:
            flat["blpx"] = blpx_cfg

        # Build nested costs sub-model
        costs_cfg = dict(costs) if isinstance(costs, dict) else {}
        if execution and isinstance(execution, dict) and execution.get("side_leverage") is not None:
            costs_cfg.setdefault("side_leverage", execution["side_leverage"])
        if costs_cfg and "costs" not in flat:
            flat["costs"] = costs_cfg

        candidates = {
            "long_count": portfolio.get("long_count"),
            "short_count": portfolio.get("short_count"),
            "baseline_gross": gross_scaling.get("baseline_gross"),
            "cost_bps_per_gross": costs.get("cost_bps_per_gross"),
            "pit_rolling_window": gross_scaling.get("pit_rolling_window"),
            "tertile_low_pct": gross_scaling.get("tertile_low_pct"),
            "tertile_high_pct": gross_scaling.get("tertile_high_pct"),
            "mult_low": multipliers.get("Low"),
            "mult_mid": multipliers.get("Medium"),
            "mult_high": multipliers.get("High"),
            "fallback_multiplier": gross_scaling.get("fallback_multiplier"),
            "fallback_on_gap_data_missing": fallback.get("fallback_on_gap_data_missing"),
            "fallback_on_audit_failure": fallback.get("fallback_on_audit_failure"),
            "ondemand_fallback_enabled": fallback.get("ondemand_fallback_enabled"),
            "shadow_ondemand_validation": fallback.get("shadow_ondemand_validation"),
            "mh_blend_enabled": mh.get("enabled"),
            "mh_horizons": mh.get("horizons"),
            "mh_weights": mh.get("weights"),
            "mh_mu_file_pattern_h": mh.get("mu_file_pattern_h"),
            "mh_omega_file_pattern_h": mh.get("omega_file_pattern_h"),
            "cs_overlay_enabled": cs.get("enabled"),
            "cs_overlay_weight": cs.get("weight"),
            "cs_rank_reversal_file_pattern": cs.get("rank_reversal_file_pattern"),
            "minvar_enabled": portfolio.get("minvar_enabled"),
            "minvar_alpha": portfolio.get("minvar_alpha"),
            # Macro keys: production.yaml places them under blpx: (not portfolio:).
            # portfolio: takes precedence if both are present.
            "macro_kappa_enabled": portfolio.get("macro_kappa_enabled", blpx.get("macro_kappa_enabled")),
            "macro_kappas": portfolio.get("macro_kappas", blpx.get("macro_kappas")),
            "macro_surprise_halflife_mean": portfolio.get(
                "macro_surprise_halflife_mean", blpx.get("macro_surprise_halflife_mean")
            ),
            "macro_surprise_halflife_vol": portfolio.get(
                "macro_surprise_halflife_vol", blpx.get("macro_surprise_halflife_vol")
            ),
            "macro_direction_enabled": portfolio.get(
                "macro_direction_enabled", blpx.get("macro_direction_enabled")
            ),
            "gap_input_dir": gap_dist.get("dir"),
            "mu_file_pattern": gap_dist.get("mu_file_pattern"),
            "omega_file_pattern": gap_dist.get("omega_file_pattern"),
            "sigma_floor": ranking.get("sigma_floor"),
            "residualization_enabled_for_p3": residualization.get("enabled_for_p3"),
            "residualization_beta_window": residualization.get("beta_window"),
            "residualization_beta_winsor_sigma": residualization.get("beta_winsor_sigma"),
            "residualization_beta_shrinkage": residualization.get("beta_shrinkage"),
            "frac_diff_enabled": frac_diff.get("enabled"),
            "frac_diff_d": frac_diff.get("d"),
            "frac_diff_threshold": frac_diff.get("threshold"),
            "frac_diff_window": frac_diff.get("window"),
            "frac_diff_normalize": frac_diff.get("normalize"),
            "ml_overlay_enabled": ml.get("enabled"),
            "ml_overlay_model_dir": ml.get("model_dir"),
            "ml_overlay_use_ticker": ml.get("use_ticker"),
            "ml_overlay_use_classification": ml.get("use_classification"),
            "ml_overlay_per_ticker_interactions": ml.get("per_ticker_interactions"),
        }
        for key, value in candidates.items():
            if value is not None and key not in flat:
                flat[key] = value
        return flat


class AppConfig(BaseModel):
    """Full application configuration.

    The canonical top-level config assembled from YAML + environment variables.
    Instantiated via ``execution.config.load_config_from_yaml()``.
    """
    model_config = {"frozen": True}

    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    kabu: KabuApiConfig = Field(default_factory=KabuApiConfig)
    tachibana: TachibanaApiConfig = Field(default_factory=TachibanaApiConfig)
    broker_provider: str = Field(default="kabu", description="使用するブローカープロバイダー ('kabu' | 'tachibana' | 'dry_run')")
    output_base_dir: str = Field(
        default_factory=lambda: str(results("sector_relative_ensemble")),
        description="バックテスト出力ルート",
    )
    output_live_dir: str = Field(
        default_factory=lambda: str(live("sector_relative_ensemble")),
        description="本番ライブ出力ルート",
    )
    run_audit: bool = Field(default=True, description="実行後に ComplianceAuditor を走らせるか")
    gap_distribution_dir: str = Field(default="", description="gap 調整分布ディレクトリ（相対パス可）")
    ml_order_overlay: MLOrderOverlayConfig = Field(default_factory=MLOrderOverlayConfig)
    v2: ProductionV2RunConfig = Field(
        default_factory=ProductionV2RunConfig,
        description="V2 本番ポートフォリオ生成パラメータ",
    )


