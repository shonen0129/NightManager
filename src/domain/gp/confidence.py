"""domain.gp.confidence – 確信度スコア κ_t の算出とサイジング接続

確信度スコア κ_t の役割
------------------------
GP の予測分散 σ²_{j,t+1} を「その日の確信度」として、
ポートフォリオのグロスエクスポージャーを動的に調整する。

    w_final = κ_t × w_base

κ_t が低い（= GP が高不確実と判断）日はポジションを縮小し、
ドローダウンを抑制する。方向判断（ロング/ショート）は既存シグナルのまま。

κ_t の算出方法は設定で切替可能:
- 業種横断集約: "mean" / "max" / "portfolio_weighted"
- 分散→確信度写像: "inv_exp" / "percentile_rank"
- EWMA スムージング: smoothing_halflife で平滑化強度を制御

ベイズ的ウェイト縮小（オプション）
-------------------------------------
``apply_bayesian_weight_shrinkage()`` は μ/σ（シャープ比的な信号対雑音比）
で各業種のウェイトを個別に縮小する。確信度の低い業種ほど薄くなる。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# ConfidenceConfig
# ---------------------------------------------------------------------------


@dataclass
class ConfidenceConfig:
    """確信度スコア算出の設定。

    Attributes
    ----------
    aggregation : str
        業種横断集約方法。"mean", "max", "portfolio_weighted" のいずれか。
    mapping : str
        分散→確信度の写像関数。"inv_exp" または "percentile_rank"。
    tau : float
        inv_exp 写像の減衰係数 τ。κ = exp(-τ × σ²)。
        大きいほど高分散日の確信度が急激に低下する。
    smoothing_halflife : int
        κ_t の EWMA スムージング半減期（営業日単位）。
        0 でスムージングなし（生の κ_t をそのまま使用）。
    min_kappa : float
        κ_t の最小値。0 除去のための下限。デフォルト 0.1。
    bayesian_weight : bool
        True のとき apply_bayesian_weight_shrinkage() をデフォルト動作に含める。
    percentile_window : int
        percentile_rank モード用の過去σ²の参照窓。
    """

    aggregation: str = "mean"
    mapping: str = "inv_exp"
    tau: float = 5.0
    smoothing_halflife: int = 5
    min_kappa: float = 0.1
    bayesian_weight: bool = False
    percentile_window: int = 60

    def __post_init__(self) -> None:
        assert self.aggregation in {"mean", "max", "portfolio_weighted"}, (
            f"aggregation は 'mean', 'max', 'portfolio_weighted' のいずれか: "
            f"got {self.aggregation}"
        )
        assert self.mapping in {"inv_exp", "percentile_rank"}, (
            f"mapping は 'inv_exp' または 'percentile_rank': got {self.mapping}"
        )
        assert 0.0 <= self.min_kappa < 1.0, (
            f"min_kappa は [0, 1) の範囲内: got {self.min_kappa}"
        )
        assert self.tau > 0.0, f"tau は正数: got {self.tau}"

    @classmethod
    def from_config_dict(cls, d: dict) -> "ConfidenceConfig":
        conf = d.get("confidence", d)
        return cls(
            aggregation=str(conf.get("aggregation", "mean")),
            mapping=str(conf.get("mapping", "inv_exp")),
            tau=float(conf.get("tau", 5.0)),
            smoothing_halflife=int(conf.get("smoothing_halflife", 5)),
            min_kappa=float(conf.get("min_kappa", 0.1)),
            bayesian_weight=bool(conf.get("bayesian_weight", False)),
            percentile_window=int(conf.get("percentile_window", 60)),
        )


# ---------------------------------------------------------------------------
# ConfidenceScorer: ステートフルな確信度スコア計算クラス
# ---------------------------------------------------------------------------


class ConfidenceScorer:
    """GP 予測分散から確信度スコア κ_t を計算するステートフルなクラス。

    EWMA スムージングと percentile_rank モードのために
    過去の σ² と κ_t 履歴を保持する。

    Parameters
    ----------
    cfg : ConfidenceConfig
        確信度算出の設定。

    Examples
    --------
    >>> scorer = ConfidenceScorer(ConfidenceConfig(tau=5.0))
    >>> kappa = scorer.score(sigma2_all, weights=w_base)
    >>> w_final = scorer.apply_sizing(w_base, kappa)
    """

    def __init__(self, cfg: Optional[ConfidenceConfig] = None) -> None:
        self.cfg = cfg if cfg is not None else ConfidenceConfig()
        self._sigma2_history: List[float] = []
        self._kappa_history: List[float] = []

    # ------------------------------------------------------------------
    # score: σ² → κ_t の変換
    # ------------------------------------------------------------------

    def score(
        self,
        sigma2_all: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """全業種の GP 予測分散から確信度スコア κ_t を算出する。

        Parameters
        ----------
        sigma2_all : ndarray, shape (N_J,)
            各業種の GP 予測分散。NaN 業種は除外される。
        weights : ndarray or None, shape (N_J,)
            ポートフォリオウェイト（aggregation='portfolio_weighted' 時に使用）。

        Returns
        -------
        kappa : float
            確信度スコア κ_t ∈ [min_kappa, 1.0]。
        """
        sigma2_arr = np.asarray(sigma2_all, dtype=float)
        valid = np.isfinite(sigma2_arr) & (sigma2_arr >= 0)

        if not valid.any():
            return 1.0  # 情報なし → フルエクスポージャー

        sigma2_valid = sigma2_arr[valid]

        # ── 業種横断集約 ─────────────────────────────────────────────
        sigma2_agg = self._aggregate(sigma2_valid, sigma2_arr, weights)

        # ── 写像: σ² → κ_raw ─────────────────────────────────────────
        kappa_raw = self._map_to_kappa(sigma2_agg)

        # ── EWMA スムージング ─────────────────────────────────────────
        kappa_smooth = self._smooth(kappa_raw)

        # ── 下限クリップ ──────────────────────────────────────────────
        kappa = float(np.clip(kappa_smooth, self.cfg.min_kappa, 1.0))

        # 履歴保存
        self._sigma2_history.append(float(sigma2_agg))
        self._kappa_history.append(kappa)

        return kappa

    # ------------------------------------------------------------------
    # apply_sizing
    # ------------------------------------------------------------------

    def apply_sizing(
        self,
        weights: np.ndarray,
        kappa: float,
    ) -> np.ndarray:
        """w_final = κ_t × w_base を適用する。

        Parameters
        ----------
        weights : ndarray, shape (N_J,)
            既存戦略の基準ウェイト w_base。
        kappa : float
            確信度スコア κ_t ∈ [0, 1]。

        Returns
        -------
        w_final : ndarray, shape (N_J,)
            確信度調整済み最終ウェイト。
        """
        kappa = float(np.clip(kappa, 0.0, 1.0))
        return np.asarray(weights, dtype=float) * kappa

    # ------------------------------------------------------------------
    # apply_bayesian_weight_shrinkage（オプション）
    # ------------------------------------------------------------------

    def apply_bayesian_weight_shrinkage(
        self,
        weights: np.ndarray,
        mu: np.ndarray,
        sigma2: np.ndarray,
        scale: float = 1.0,
    ) -> np.ndarray:
        """μ/σ によるベイズ的ウェイト縮小（業種別個別縮小）。

        各業種の「シグナル対雑音比」|μ_j / σ_j| でウェイトを縮小する。
        不確実な業種（高σ²）はウェイトが薄くなり、
        確信のある業種（低σ²）はウェイトが厚くなる。
        ウェイトの符号（ロング/ショート方向）は保持される。

        Parameters
        ----------
        weights : ndarray, shape (N_J,)
            基準ウェイト w_base。
        mu : ndarray, shape (N_J,)
            GP 予測平均 μ_j（z_lin + GP 残差予測）。
        sigma2 : ndarray, shape (N_J,)
            GP 予測分散 σ²_j。
        scale : float
            縮小強度。1.0 = 標準的な縮小。0 = 縮小なし（恒等）。

        Returns
        -------
        w_shrunk : ndarray, shape (N_J,)
            縮小済みウェイト。
        """
        w = np.asarray(weights, dtype=float)
        mu_arr = np.asarray(mu, dtype=float)
        sigma_arr = np.sqrt(np.maximum(np.asarray(sigma2, dtype=float), 1e-12))

        snr = np.abs(mu_arr) / sigma_arr  # |μ/σ|

        # SNR → [0, 1] 正規化: tanh(scale × snr / snr_max)
        snr_max = float(np.nanmax(snr))
        if snr_max < 1e-8:
            return w  # SNR が全体的に極小 → 縮小なし

        shrinkage = np.where(
            np.isfinite(snr),
            np.tanh(scale * snr / snr_max),
            1.0,
        )

        return w * shrinkage

    # ------------------------------------------------------------------
    # 履歴アクセサ
    # ------------------------------------------------------------------

    def get_kappa_series(
        self,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> pd.Series:
        """蓄積された κ_t の時系列を Series で返す。"""
        idx = dates if dates is not None else range(len(self._kappa_history))
        return pd.Series(self._kappa_history, index=idx, name="kappa", dtype=float)

    def get_sigma2_series(
        self,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> pd.Series:
        """蓄積された集約済み σ²_agg の時系列を Series で返す。"""
        idx = dates if dates is not None else range(len(self._sigma2_history))
        return pd.Series(
            self._sigma2_history, index=idx, name="sigma2_agg", dtype=float
        )

    def reset(self) -> None:
        """履歴をリセット（walk-forward の各フォールドで呼び出す）。"""
        self._sigma2_history.clear()
        self._kappa_history.clear()

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        sigma2_valid: np.ndarray,
        sigma2_all: np.ndarray,
        weights: Optional[np.ndarray],
    ) -> float:
        """業種横断集約を実行する。"""
        if self.cfg.aggregation == "mean":
            return float(np.mean(sigma2_valid))
        elif self.cfg.aggregation == "max":
            return float(np.max(sigma2_valid))
        elif self.cfg.aggregation == "portfolio_weighted":
            if weights is not None:
                valid = np.isfinite(sigma2_all) & (sigma2_all >= 0)
                w = np.abs(np.asarray(weights, dtype=float)[valid])
                w_sum = float(w.sum())
                if w_sum > 1e-12:
                    return float(np.sum(w * sigma2_valid) / w_sum)
            return float(np.mean(sigma2_valid))
        else:
            return float(np.mean(sigma2_valid))  # フォールバック

    def _map_to_kappa(self, sigma2_agg: float) -> float:
        """集約済み σ² → κ_raw の写像。"""
        if self.cfg.mapping == "inv_exp":
            return float(np.exp(-self.cfg.tau * sigma2_agg))
        elif self.cfg.mapping == "percentile_rank":
            window = self.cfg.percentile_window
            if len(self._sigma2_history) >= 20:
                hist = np.array(self._sigma2_history[-window:])
                # 現在の σ² が過去に対してどの分位か（高分位 = 高分散 = 低確信度）
                p = float(np.mean(hist <= sigma2_agg))
                return 1.0 - p  # 高分位ほど低確信度
            else:
                # 履歴不足: inv_exp でフォールバック
                return float(np.exp(-self.cfg.tau * sigma2_agg))
        return float(np.exp(-self.cfg.tau * sigma2_agg))

    def _smooth(self, kappa_raw: float) -> float:
        """EWMA スムージングを適用する。"""
        if self.cfg.smoothing_halflife <= 0 or not self._kappa_history:
            return kappa_raw
        alpha = 1.0 - 0.5 ** (1.0 / max(self.cfg.smoothing_halflife, 1))
        prev = self._kappa_history[-1]
        return alpha * kappa_raw + (1.0 - alpha) * prev


# ---------------------------------------------------------------------------
# スタンドアロン関数（後方互換 / シンプルな用途向け）
# ---------------------------------------------------------------------------


def compute_confidence_from_sigma2(
    sigma2_all: np.ndarray,
    tau: float = 5.0,
    aggregation: str = "mean",
    min_kappa: float = 0.1,
    weights: Optional[np.ndarray] = None,
) -> float:
    """GP 予測分散から確信度スコア κ_t を算出するスタンドアロン関数。

    ステートフルなスムージングが不要な場合に使用。

    Parameters
    ----------
    sigma2_all : ndarray, shape (N_J,)
        各業種の GP 予測分散。
    tau : float
        inv_exp 減衰係数。
    aggregation : str
        "mean", "max", "portfolio_weighted" のいずれか。
    min_kappa : float
        κ_t の下限値。
    weights : ndarray or None
        aggregation='portfolio_weighted' 時のウェイト。

    Returns
    -------
    kappa : float
        確信度スコア ∈ [min_kappa, 1.0]。
    """
    scorer = ConfidenceScorer(
        ConfidenceConfig(
            aggregation=aggregation,
            tau=tau,
            smoothing_halflife=0,
            min_kappa=min_kappa,
        )
    )
    return scorer.score(sigma2_all, weights=weights)


def regime_kappa_decomposition(
    kappa_series: pd.Series,
    vol_series: pd.Series,
    n_quantiles: int = 3,
) -> pd.DataFrame:
    """ボラティリティレジーム別の κ_t 分布を集計する。

    Parameters
    ----------
    kappa_series : pd.Series
        κ_t の時系列（DatetimeIndex 付き）。
    vol_series : pd.Series
        ボラティリティ代理変数（例: 20日リターン標準偏差、DatetimeIndex 付き）。
    n_quantiles : int
        ボラティリティの分位数。3 なら 低/中/高 の3レジーム。

    Returns
    -------
    pd.DataFrame
        レジーム別の κ_t の統計量（mean, std, min, max, count）。
    """
    df = pd.DataFrame({"kappa": kappa_series, "vol": vol_series}).dropna()
    if len(df) < n_quantiles * 5:
        return pd.DataFrame()

    df["regime"] = pd.qcut(
        df["vol"],
        q=n_quantiles,
        labels=[f"vol_q{i+1}" for i in range(n_quantiles)],
    )
    return df.groupby("regime")["kappa"].agg(["mean", "std", "min", "max", "count"])


def quantile_return_decomposition(
    returns: pd.Series,
    kappa_series: pd.Series,
    n_quantiles: int = 3,
) -> pd.DataFrame:
    """確信度分位別（低/中/高 κ_t）のリターン分解。

    Parameters
    ----------
    returns : pd.Series
        日次リターン（net）。
    kappa_series : pd.Series
        κ_t の時系列。
    n_quantiles : int
        分位数。3 なら 低確信度/中/高確信度 の3グループ。

    Returns
    -------
    pd.DataFrame
        分位別の年率リターン、ボラティリティ、R/R、サンプル数。
    """
    df = pd.DataFrame({"ret": returns, "kappa": kappa_series}).dropna()
    if len(df) < n_quantiles * 5:
        return pd.DataFrame()

    df["kappa_quantile"] = pd.qcut(
        df["kappa"],
        q=n_quantiles,
        labels=[f"kappa_q{i+1}" for i in range(n_quantiles)],
    )

    rows = []
    for group, sub in df.groupby("kappa_quantile"):
        r = sub["ret"]
        ar = float(r.mean() * 245)
        risk = float(r.std(ddof=1) * np.sqrt(245))
        rr = ar / risk if risk > 1e-8 else float("nan")
        rows.append(
            {
                "kappa_quantile": group,
                "kappa_mean": float(sub["kappa"].mean()),
                "n_days": len(r),
                "annualized_return": ar,
                "annualized_vol": risk,
                "rr": rr,
            }
        )
    return pd.DataFrame(rows).set_index("kappa_quantile")
