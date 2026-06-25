"""Pydantic schemas for validated configuration variables.

Single source of truth for all configuration types.
All modules should import StrategyConfig / RiskConfig from here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class StrategyConfig(BaseModel):
    """Strategy parameters validation schema.

    Covers model hyperparameters, signal construction settings,
    portfolio construction settings, and execution-layer parameters
    (start_date, risk thresholds) that are used by production runners.
    """
    model_config = {"frozen": True}

    model_name: str = Field(default="sector_relative_ensemble", description="モデル識別名")
    k: int = Field(default=6, ge=1, description="固有ベクトル空間の次元数 K")
    lambda_reg: float = Field(default=0.75, ge=0.0, le=1.0, description="相関行列レギュラリゼーション強度")
    q: float = Field(default=0.3, ge=0.0, le=1.0, description="ロング/ショート選択比率 (各サイド q×N_JP 銘柄)")
    weight_mode: str = Field(default="signal", description="ウェイト構築モード (signal | equal | rank)")
    dispersion_filter: bool = Field(default=False, description="分散フィルター有効フラグ")
    dispersion_metric: str = Field(default="long_short_mean_gap", description="分散指標の種類")
    v3_mode: str = Field(default="static", description="事前部分空間モード (static | dynamic)")
    ewma_half_life: int = Field(default=45, ge=1, description="EWMA 半減期 (日数)")
    lambda_lw: float = Field(default=0.5, ge=0.0, le=1.0, description="Ledoit-Wolf 縮小強度")
    lw_target: str = Field(default="equicorrelation", description="Ledoit-Wolf ターゲット行列")
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

    # Production runner parameters (start_date + risk thresholds)
    # NOTE: risk thresholds are duplicated here for backward compat with production runners
    # that pass a single StrategyConfig to both strategy and risk layers.
    # The canonical risk-only type is RiskConfig; use AppConfig.risk in new code.
    start_date: str = Field(default="2015-01-01", description="バックテスト開始日")
    var_confidence: float = Field(default=0.99, ge=0.0, le=1.0, description="VaR 信頼水準")
    var_window: int = Field(default=250, ge=1, description="VaR/ES 計算ウィンドウ (日数)")
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
    output_base_dir: str = Field(default="results/sector_relative_ensemble", description="バックテスト出力ルート")
    output_live_dir: str = Field(default="live/sector_relative_ensemble", description="本番ライブ出力ルート")
    run_audit: bool = Field(default=True, description="実行後に ComplianceAuditor を走らせるか")


class ProductionV2RunConfig(BaseModel):
    """Runtime parameters for the v2 daily production pipeline.

    Parsed from the YAML ``portfolio:``, ``gross_scaling:``, ``costs:``,
    and ``fallback:`` sections via ``models.production_v2.parse_run_config(cfg)``.
    Acts as the single source of truth for all v2 pipeline constants — replacing
    the module-level literals that previously lived in ``tools/run_daily_production_v2.py``.
    """
    model_config = {"frozen": True}

    # --- Portfolio construction ---
    long_count: int = Field(default=5, ge=1, description="ロング選択銘柄数")
    short_count: int = Field(default=5, ge=1, description="ショート選択銘柄数")
    baseline_gross: float = Field(default=2.0, ge=0.0, description="pre-gross 基準グロスエクスポージャー")

    # --- Cost model ---
    cost_bps_per_gross: float = Field(
        default=10.0, ge=0.0,
        description="ex-ante コスト (bps/unit gross)。IR 計算のみに使用。実取引コストではない。"
    )

    # --- RuleD dynamic gross scaling ---
    pit_rolling_window: int = Field(default=252, ge=1, description="PIT 三分位ビニング用ローリング窓（営業日）")
    tertile_low_pct: float = Field(default=33.3333, ge=0.0, le=100.0, description="低閾値パーセンタイル")
    tertile_high_pct: float = Field(default=66.6667, ge=0.0, le=100.0, description="高閾値パーセンタイル")
    mult_low: float = Field(default=0.75, ge=0.0, description="Low ビン グロス乗数")
    mult_mid: float = Field(default=1.00, ge=0.0, description="Medium ビン グロス乗数")
    mult_high: float = Field(default=1.00, ge=0.0, description="High ビン グロス乗数")
    fallback_multiplier: float = Field(default=1.00, ge=0.0, description="PIT 履歴不足時のフォールバック乗数")

    # --- Fallback behavior ---
    fallback_on_gap_data_missing: bool = Field(default=True, description="gap data 欠損時に v1 フォールバック")
    fallback_on_audit_failure: bool = Field(default=True, description="数値監査失敗時に v1 フォールバック")
