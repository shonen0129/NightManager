"""Pydantic schemas for validated configuration variables.

Single source of truth for all configuration types.
All modules should import StrategyConfig / RiskConfig from here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

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

# Legacy top-level keys that are consumed by the BLPX sub-model but may
# appear unprefixed in older flat files or nested residualization sections.
_BLPX_ALIASES: dict[str, str] = {
    "topix_beta_coef": "topix_beta_coef",
    "residualization_topix_beta_coef": "topix_beta_coef",
    "residualization_beta_window": "beta_window",
}


def _map_flat_to_nested(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize both flat and nested production YAML to ``ProductionV2RunConfig`` input.

    The returned dict contains flat V2 fields plus nested ``blpx`` and
    ``costs`` sub-dicts.  Unknown keys are filtered out so the result
    validates cleanly against ``ProductionV2RunConfig`` (``extra=forbid``).

    Flat keys take precedence over values inside nested sections.
    """
    if not isinstance(raw, dict):
        return raw

    # Runtime references are safe because this function is called after the
    # module has finished loading.
    v2_keys = set(ProductionV2RunConfig.model_fields) - {"blpx", "costs"}
    blpx_keys = set(BLPXConfig.model_fields)
    costs_keys = set(CostConfig.model_fields)

    out: dict[str, Any] = {}
    blpx: dict[str, Any] = {}
    costs: dict[str, Any] = {}

    # First pass: separate flat blpx_*, cost keys, and other flat keys.
    for k, v in raw.items():
        if k.startswith(_BLPX_PREFIX):
            key = k[len(_BLPX_PREFIX):]
            if key in blpx_keys:
                blpx[key] = v
        elif k in _COSTS_FLAT_FIELDS:
            if k in costs_keys:
                costs[k] = v
        else:
            out[k] = v

    def _pop_section(name: str) -> dict[str, Any]:
        sec = out.pop(name, None)
        return dict(sec) if isinstance(sec, dict) else {}

    # Pull nested sections (they are removed from ``out``).
    portfolio = _pop_section("portfolio")
    gross_scaling = _pop_section("gross_scaling")
    costs_section = _pop_section("costs")
    fallback = _pop_section("fallback")
    mh = _pop_section("multi_horizon_blend")
    cs = _pop_section("cs_feature_overlay")
    blpx_section = _pop_section("blpx")
    residualization = _pop_section("residualization")
    features = _pop_section("features")
    ml = _pop_section("ml_order_overlay")
    gap_dist = _pop_section("gap_distribution")
    execution = _pop_section("execution")
    ranking = _pop_section("ranking")
    signal_components = _pop_section("signal_components")

    # Merge nested blpx/costs with flat-prefixed values (flat wins).
    if blpx_section:
        for k, v in blpx_section.items():
            blpx.setdefault(k, v)
    if costs_section:
        for k, v in costs_section.items():
            costs.setdefault(k, v)

    # Flatten gross scaling multipliers.
    multipliers = gross_scaling.get("multipliers", {}) if isinstance(gross_scaling, dict) else {}

    # Section-to-flat mappings.  Each tuple is (section, source_key, target_key).
    # Some target keys are in sub-models and will be routed there by the
    # per-key routing below.
    section_mappings: list[tuple[dict[str, Any], str, str]] = [
        (portfolio, "long_count", "long_count"),
        (portfolio, "short_count", "short_count"),
        (portfolio, "minvar_enabled", "minvar_enabled"),
        (portfolio, "minvar_alpha", "minvar_alpha"),
        (portfolio, "macro_kappa_enabled", "macro_kappa_enabled"),
        (portfolio, "macro_kappas", "macro_kappas"),
        (portfolio, "macro_surprise_halflife_mean", "macro_surprise_halflife_mean"),
        (portfolio, "macro_surprise_halflife_vol", "macro_surprise_halflife_vol"),
        (portfolio, "macro_direction_enabled", "macro_direction_enabled"),
        (gross_scaling, "baseline_gross", "baseline_gross"),
        (gross_scaling, "pit_rolling_window", "pit_rolling_window"),
        (gross_scaling, "tertile_low_pct", "tertile_low_pct"),
        (gross_scaling, "tertile_high_pct", "tertile_high_pct"),
        (multipliers, "Low", "mult_low"),
        (multipliers, "Medium", "mult_mid"),
        (multipliers, "High", "mult_high"),
        (gross_scaling, "fallback_multiplier", "fallback_multiplier"),
        (fallback, "fallback_on_gap_data_missing", "fallback_on_gap_data_missing"),
        (fallback, "fallback_on_audit_failure", "fallback_on_audit_failure"),
        (fallback, "ondemand_fallback_enabled", "ondemand_fallback_enabled"),
        (fallback, "shadow_ondemand_validation", "shadow_ondemand_validation"),
        (mh, "enabled", "mh_blend_enabled"),
        (mh, "horizons", "mh_horizons"),
        (mh, "weights", "mh_weights"),
        (mh, "mu_file_pattern_h", "mh_mu_file_pattern_h"),
        (mh, "omega_file_pattern_h", "mh_omega_file_pattern_h"),
        (cs, "enabled", "cs_overlay_enabled"),
        (cs, "weight", "cs_overlay_weight"),
        (cs, "rank_reversal_file_pattern", "cs_rank_reversal_file_pattern"),
        (residualization, "enabled_for_p3", "residualization_enabled_for_p3"),
        (residualization, "beta_window", "residualization_beta_window"),
        (residualization, "winsor_sigma", "residualization_beta_winsor_sigma"),
        (residualization, "shrinkage", "residualization_beta_shrinkage"),
        (residualization, "topix_beta_coef", "residualization_topix_beta_coef"),
        (gap_dist, "dir", "gap_input_dir"),
        (gap_dist, "mu_file_pattern", "mu_file_pattern"),
        (gap_dist, "omega_file_pattern", "omega_file_pattern"),
        (ranking, "mode", "ranking_mode"),
        (ranking, "sigma_floor", "sigma_floor"),
        (ml, "enabled", "ml_overlay_enabled"),
        (ml, "model_dir", "ml_overlay_model_dir"),
        (ml, "use_ticker", "ml_overlay_use_ticker"),
        (ml, "use_classification", "ml_overlay_use_classification"),
        (ml, "per_ticker_interactions", "ml_overlay_per_ticker_interactions"),
        (ml, "p_trade_ema_span", "ml_overlay_p_trade_ema_span"),
        (ml, "fallback_to_baseline", "ml_overlay_fallback_to_baseline"),
    ]

    # fractional_diff sub-section.
    frac_diff = features.get("fractional_diff", {}) if isinstance(features, dict) else {}
    section_mappings.extend([
        (frac_diff, "enabled", "frac_diff_enabled"),
        (frac_diff, "d", "frac_diff_d"),
        (frac_diff, "threshold", "frac_diff_threshold"),
        (frac_diff, "window", "frac_diff_window"),
        (frac_diff, "normalize", "frac_diff_normalize"),
    ])

    for section, source_key, target_key in section_mappings:
        if source_key in section and target_key not in out:
            out[target_key] = section[source_key]

    # Signal component weights -> blpx sub-model.
    if signal_components and isinstance(signal_components, dict):
        for comp, weight_key in (
            ("raw_pca", "raw_pca_weight"),
            ("residual_pca", "residual_pca_weight"),
            ("raw_blpx", "raw_blpx_weight"),
            ("residual_blpx", "residual_blpx_weight"),
        ):
            comp_cfg = signal_components.get(comp)
            if isinstance(comp_cfg, dict) and weight_key not in blpx:
                enabled = bool(comp_cfg.get("enabled", False))
                blpx[weight_key] = float(comp_cfg.get("weight", 1.0 if enabled else 0.0)) if enabled else 0.0

    # execution.side_leverage -> costs.
    if execution and isinstance(execution, dict) and "side_leverage" in execution:
        costs.setdefault("side_leverage", execution["side_leverage"])

    # Residualization -> blpx aliases for the BLPX model.
    if "beta_window" not in blpx and "residualization_beta_window" in out:
        blpx["beta_window"] = out["residualization_beta_window"]
    for alias, blpx_key in _BLPX_ALIASES.items():
        if blpx_key not in blpx and alias in out:
            if blpx_key in blpx_keys:
                blpx[blpx_key] = out.pop(alias)

    # Copy any overlapping sub-model keys to the top-level aliases that
    # ``ProductionV2RunConfig`` also accepts (e.g. macro_kappas).
    for k, v in list(blpx.items()):
        if k in v2_keys and k not in out:
            out[k] = v
    for k, v in list(costs.items()):
        if k in v2_keys and k not in out:
            out[k] = v

    # Filter each layer to allowed keys.
    out = {k: v for k, v in out.items() if k in v2_keys}
    if blpx:
        blpx = {k: v for k, v in blpx.items() if k in blpx_keys}
        if blpx:
            out["blpx"] = blpx
    if costs:
        costs = {k: v for k, v in costs.items() if k in costs_keys}
        if costs:
            out["costs"] = costs

    return out


class ProductionV2RunConfig(BaseModel):
    """Runtime parameters for the v2 daily production pipeline.

    Parsed from flat or nested YAML via ``_map_flat_to_nested``.  Acts as the
    single source of truth for all v2 pipeline constants.
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

    # --- Class helpers ---
    # ``_map_flat_to_nested`` in this module handles both nested and flat YAML
    # normalization; the model itself no longer needs a pre-validation hook.


class NextGenConfig(BaseModel):
    """Next-Gen convex portfolio optimizer configuration."""

    model_config = {"frozen": True}

    lambda_risk: float = Field(default=3.0, ge=0.0, description="リスク回避係数")
    cost_bps: float = Field(default=5.0, ge=0.0, description="取引コスト bps")
    turnover_penalty: float = Field(default=0.0001, ge=0.0, description="ターンオーバーペナルティ")
    max_single_weight: float = Field(default=0.25, gt=0.0, le=1.0, description="銘柄別最大ウェイト")
    gross_target: float = Field(default=2.0, gt=0.0, description="目標グロスエクスポージャー")
    min_weight_threshold: float = Field(default=1e-4, ge=0.0, description="ゼロ切り捨て閾値")
    solver_tol: float = Field(default=1e-7, gt=0.0, description="SLSQP 停止許容値")
    max_iter: int = Field(default=100, ge=1, description="SLSQP 最大反復回数")
    smooth_eps: float = Field(default=1e-4, gt=0.0, description="Pseudo-Huber 平滑化パラメータ")


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
    nextgen: NextGenConfig = Field(
        default_factory=NextGenConfig,
        description="Next-Gen convex optimizer parameters",
    )


