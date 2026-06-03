"""Production/runtime configuration shared by strategy and production runner."""

import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# .envファイルの読み込み（プロジェクトルートまたはsrc/config.pyからの相対パス）
# プロジェクト構造に応じて.envファイルを探す
_env_paths = [
    Path(__file__).parent.parent / ".env",  # プロジェクトルート
    Path(__file__).parent / ".env",  # srcディレクトリ内
    Path(__file__).parent / "kabu_auto_login" / ".env",  # kabu_auto_login env
]
for _env_path in _env_paths:
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=False)


# 定数
DEFAULT_START_DATE = "2015-01-01"


def _normalize_kabu_api_url(api_url: str) -> str:
    """Ensure the API base URL includes /kabusapi."""
    if not api_url:
        return api_url
    normalized = api_url.rstrip("/")
    if normalized.endswith("/kabusapi"):
        return normalized
    return f"{normalized}/kabusapi"

# アセット数（US ETF 15銘柄 + TOPIX-17セクターETF 17銘柄）
N_US_ASSETS = 15
N_JP_ASSETS = 17
N_TOTAL_ASSETS = N_US_ASSETS + N_JP_ASSETS

STRATEGY_DEFAULTS = {
    "K": 6,
    "lambda_reg": 0.75,
    "q": 0.3,
    "weight_mode": "signal",
    "dispersion_filter": False,
    "dispersion_metric": "long_short_mean_gap",
    "v3_mode": "static",
    "ewma_half_life": 45,
    "lambda_lw": 0.5,
    "lw_target": "equicorrelation",
    "corr_window": 60,
    "include_v4_prior": True,
    "signal_mode": "gap_residual",
    "gap_open_coef": 0.70,
    "topix_beta_coef": 0.6,
    "beta_window": 60,
    "gap_noise_std": 0.0,
    "gap_noise_seed": 42,
    "gamma": 0.5,
    # 片道スリッページ (basis points).
    # TOPIX-17 ETFの厕付き成行取引におけるスプレッド+約定不利の合計推定値。
    # 往復コスト = 2 × slippage_bps/10000 × gross_exposure_per_day
    "slippage_bps": 5.0,
    "vol_adjusted_target": True,
}

DISPERSION_HISTORY_WINDOW = 60

PRODUCTION_DEFAULTS = {
    "start_date": DEFAULT_START_DATE,
    "k": STRATEGY_DEFAULTS["K"],
    "lambda_reg": STRATEGY_DEFAULTS["lambda_reg"],
    "q": STRATEGY_DEFAULTS["q"],
    "weight_mode": STRATEGY_DEFAULTS["weight_mode"],
    "dispersion_filter": STRATEGY_DEFAULTS["dispersion_filter"],
    "dispersion_metric": STRATEGY_DEFAULTS["dispersion_metric"],
    "v3_mode": STRATEGY_DEFAULTS["v3_mode"],
    "ewma_half_life": STRATEGY_DEFAULTS["ewma_half_life"],
    "lambda_lw": STRATEGY_DEFAULTS["lambda_lw"],
    "lw_target": STRATEGY_DEFAULTS["lw_target"],
    "corr_window": STRATEGY_DEFAULTS["corr_window"],
    "include_v4_prior": STRATEGY_DEFAULTS["include_v4_prior"],
    "signal_mode": STRATEGY_DEFAULTS["signal_mode"],
    "gap_open_coef": STRATEGY_DEFAULTS["gap_open_coef"],
    "topix_beta_coef": STRATEGY_DEFAULTS["topix_beta_coef"],
    "beta_window": STRATEGY_DEFAULTS["beta_window"],
    "gamma": STRATEGY_DEFAULTS.get("gamma", 0.5),
    "gap_noise_std": STRATEGY_DEFAULTS["gap_noise_std"],
    "gap_noise_seed": STRATEGY_DEFAULTS["gap_noise_seed"],
    "slippage_bps": STRATEGY_DEFAULTS["slippage_bps"],
    "vol_adjusted_target": STRATEGY_DEFAULTS["vol_adjusted_target"],
}

# kabuステーション API設定
# 環境変数から読み込む: KABU_API_URL, KABU_API_TOKEN, KABU_API_PASSWORD
KABU_API_CONFIG = {
    "api_url": _normalize_kabu_api_url(
        os.environ.get("KABU_API_URL", "http://localhost:18080")
    ),
    "api_token": os.environ.get("KABU_API_TOKEN", ""),
    "request_timeout": int(os.environ.get("KABU_REQUEST_TIMEOUT", "10")),
    # 信用取引設定: 1=制度信用, 2=一般信用(長期), 3=一般信用(デイトレ)
    "margin_trade_type": int(os.environ.get("KABU_MARGIN_TRADE_TYPE", "3")),
    # 口座種別: 2=一般, 4=特定, 12=法人
    "account_type": int(os.environ.get("KABU_ACCOUNT_TYPE", "4")),
}


def validate_kabu_config(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """kabuステーションAPI設定の検証

    Args:
        config: 検証する設定 (Noneの場合はKABU_API_CONFIGを使用)

    Returns:
        検証済みの設定dict

    Raises:
        ValueError: 設定が不正な場合
    """
    if config is None:
        config = KABU_API_CONFIG.copy()

    # API URLの検証
    api_url = _normalize_kabu_api_url(config.get("api_url", ""))
    config["api_url"] = api_url
    if not api_url:
        raise ValueError(
            "KABU_API_URL is required. "
            "Set KABU_API_URL environment variable or provide via command line."
        )
    if not api_url.startswith(("http://", "https://")):
        raise ValueError(
            f"KABU_API_URL must start with http:// or https://, got: {api_url}"
        )

    # APIトークンの検証
    # KABU_API_TOKEN が空でも、KABU_API_PASSWORD があれば
    # 実行時に /token でトークン発行するため許可する。
    api_token = config.get("api_token", "")
    api_password = config.get("api_password", os.environ.get("KABU_API_PASSWORD", ""))
    if not api_token and not api_password:
        raise ValueError(
            "Either KABU_API_TOKEN or KABU_API_PASSWORD is required. "
            "Set KABU_API_TOKEN directly, or set KABU_API_PASSWORD "
            "to issue token via /token endpoint."
        )

    # リクエストタイムアウトの検証
    timeout = config.get("request_timeout", 10)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(
            f"KABU_REQUEST_TIMEOUT must be a positive number, got: {timeout}"
        )

    # 信用取引タイプの検証
    margin_type = config.get("margin_trade_type", 3)
    valid_margin_types = (1, 2, 3)  # 1=制度, 2=一般長期, 3=一般デイトレ
    if margin_type not in valid_margin_types:
        raise ValueError(
            f"KABU_MARGIN_TRADE_TYPE must be one of {valid_margin_types}, got: {margin_type}. "
            "1=制度信用, 2=一般信用(長期), 3=一般信用(デイトレ)"
        )

    # 口座種別の検証
    account_type = config.get("account_type", 4)
    valid_account_types = (2, 4, 12)  # 2=一般, 4=特定, 12=法人
    if account_type not in valid_account_types:
        raise ValueError(
            f"KABU_ACCOUNT_TYPE must be one of {valid_account_types}, got: {account_type}. "
            "2=一般口座, 4=特定口座, 12=法人口座"
        )

    return config


def get_validated_kabu_config() -> Dict[str, Any]:
    """環境変数からkabuステーションAPI設定を読み込み、検証して返す

    Returns:
        検証済みの設定dict

    Raises:
        ValueError: 設定が不正な場合
    """
    config = {
        "api_url": _normalize_kabu_api_url(
            os.environ.get("KABU_API_URL", "http://localhost:18080")
        ),
        "api_token": os.environ.get("KABU_API_TOKEN", ""),
        "request_timeout": int(os.environ.get("KABU_REQUEST_TIMEOUT", "10")),
        "margin_trade_type": int(os.environ.get("KABU_MARGIN_TRADE_TYPE", "3")),
        "account_type": int(os.environ.get("KABU_ACCOUNT_TYPE", "4")),
    }
    return validate_kabu_config(config)
