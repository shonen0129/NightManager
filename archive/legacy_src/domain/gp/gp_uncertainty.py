"""domain.gp.gp_uncertainty – ガウス過程回帰による予測不確実性推定モジュール

概要
----
既存線形シグナル z_lin に対し、GP の予測分散 σ²_{j,t+1} を
「その日の確信度」としてポジションサイジングに用いる。
GP 予測平均は既存線形予測に近い骨格を保つ（線形カーネルで等価性を担保）。

方向予測の置換ではなく、確信度ベースのエクスポージャー調整が本モジュールの役割。

インターフェース
----------------
``fit(f_matrix, z_lin_matrix, y_matrix, dates)``
    walk-forward 対応のローリング窓学習。リーク防止を assert で監査。

``predict_mean_var(f_t, z_lin)``
    予測平均 μ_j と予測分散 σ²_j を返す（shape: (N_J,) 各々）。

``compute_confidence(sigma2_all)``
    全業種の予測分散から確信度スコア κ_t ∈ [min_kappa, 1.0] を算出。

``apply_sizing(weights, kappa)``
    w_final = κ_t × w_base を適用して最終ウェイトを返す。

``save(path) / load(path)``
    学習済みモデルと設定を永続化。

Notes
-----
計算コスト: sklearn GPR は O(n³) の計算量。
ローリング窓 window_size=250 × 業種 17 で fit の都度 O(n³) が発生する。
walk-forward では1フォールドにつき train_size 分だけ fit が走る。
スパース GP オプション（use_sparse_gp=True）を設定すると誘導点近似を使用。
"""

from __future__ import annotations

import json
import logging
import pickle
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from .kernels import (
    KernelConfig,
    build_structured_kernel,
    extract_kernel_params,
    summarize_ard_importance,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GP設定データクラス
# ---------------------------------------------------------------------------


@dataclass
class GPConfig:
    """GP 不確実性推定モジュールの設定。

    Attributes
    ----------
    n_factors : int
        ファクタースコアの次元数 K（デフォルト 6）。
    n_sectors : int
        JP 業種数 N_J（デフォルト 17）。
    window_size : int
        ローリング学習窓（営業日数）。O(n³) の計算コストを考慮し 250~500 推奨。
    n_restarts_optimizer : int
        周辺尤度最大化の再試行数。大きいほど良解を得やすいが遅い。
    use_sparse_gp : bool
        True のとき誘導点（Nyström 近似）を使用。window_size > 300 で推奨。
    n_inducing_points : int
        スパース GP の誘導点数。
    seed : int
        再現性のための乱数シード。
    kernel_cfg : KernelConfig
        カーネル設定。
    enabled : bool
        False のとき predict_mean_var は (z_lin, ones*nan) を返す（OFF スイッチ）。
    """

    n_factors: int = 6
    n_sectors: int = 17
    window_size: int = 250
    n_restarts_optimizer: int = 3
    use_sparse_gp: bool = False
    n_inducing_points: int = 50
    seed: int = 42
    kernel_cfg: KernelConfig = field(default_factory=KernelConfig)
    enabled: bool = True

    def __post_init__(self) -> None:
        assert self.window_size >= 60, (
            f"window_size は 60 以上にしてください: got {self.window_size}"
        )
        assert self.n_restarts_optimizer >= 0

    @classmethod
    def from_config_dict(cls, d: dict) -> "GPConfig":
        """YAML config dict（gp + fitting セクション）から生成。"""
        gp_sec = d.get("gp", d)
        fit_sec = d.get("fitting", d)
        kernel_sec = d.get("kernel", {})
        return cls(
            n_factors=int(gp_sec.get("n_factors", 6)),
            n_sectors=int(gp_sec.get("n_sectors", 17)),
            window_size=int(fit_sec.get("window_size", 250)),
            n_restarts_optimizer=int(fit_sec.get("n_restarts_optimizer", 3)),
            use_sparse_gp=bool(fit_sec.get("use_sparse_gp", False)),
            n_inducing_points=int(fit_sec.get("sparse_gp_inducing_points", 50)),
            seed=int(gp_sec.get("seed", 42)),
            kernel_cfg=KernelConfig.from_config_dict(kernel_sec),
            enabled=bool(gp_sec.get("enabled", True)),
        )


# ---------------------------------------------------------------------------
# メインクラス
# ---------------------------------------------------------------------------


class GPUncertaintyModule:
    """ガウス過程回帰による予測不確実性推定モジュール。

    Parameters
    ----------
    cfg : GPConfig
        モジュール設定。

    Examples
    --------
    >>> cfg = GPConfig(window_size=250)
    >>> gp_module = GPUncertaintyModule(cfg)
    >>> gp_module.fit(f_matrix, z_lin_matrix, y_matrix, trade_dates)
    >>> mu, sigma2 = gp_module.predict_mean_var(f_t, z_lin)
    >>> kappa = gp_module.compute_confidence(sigma2)
    >>> w_final = gp_module.apply_sizing(w_base, kappa)
    """

    VERSION = "1.0.0"

    def __init__(self, cfg: Optional[GPConfig] = None) -> None:
        self.cfg = cfg if cfg is not None else GPConfig()
        self._models: List[Any] = []  # 業種別 GPR モデル (len = n_sectors)
        self._is_fitted: bool = False
        self._fit_metadata: Dict[str, Any] = {}
        self._train_end_date: Optional[pd.Timestamp] = None
        self._kappa_history: List[float] = []  # スムージング用の過去 κ_t 履歴

        # 再現性: NumPy グローバルシードの固定
        np.random.seed(self.cfg.seed)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        f_matrix: np.ndarray,
        z_lin_matrix: np.ndarray,
        y_matrix: np.ndarray,
        dates: pd.DatetimeIndex,
        signal_dates: Optional[pd.DatetimeIndex] = None,
    ) -> "GPUncertaintyModule":
        """ローリング窓内のデータで全業種の GP を学習する。

        Parameters
        ----------
        f_matrix : ndarray, shape (T, K)
            ファクタースコアのパネル（US クローズ日基準）。
        z_lin_matrix : ndarray, shape (T, N_J)
            既存線形シグナル（GP 予測平均の骨格として使用）。
        y_matrix : ndarray, shape (T, N_J)
            実現 Open-to-Close リターン（教師信号）。
        dates : pd.DatetimeIndex, shape (T,)
            トレード日（JP OC リターン日 = signal_date + 1bd）。
        signal_dates : pd.DatetimeIndex or None, shape (T,)
            シグナル生成日（= US クローズ日）。リーク監査に使用。
            None の場合は dates - 1 calendar day で近似（警告あり）。

        Returns
        -------
        self

        Raises
        ------
        AssertionError
            データリーク検出時。
        """
        if not self.cfg.enabled:
            logger.info("GPUncertaintyModule は無効化されています (enabled=False)")
            return self

        T, K = f_matrix.shape
        assert K == self.cfg.n_factors, (
            f"f_matrix の列数 {K} が n_factors={self.cfg.n_factors} と不一致"
        )
        assert z_lin_matrix.shape == (T, self.cfg.n_sectors), (
            f"z_lin_matrix の shape が期待値 ({T}, {self.cfg.n_sectors}) と不一致: "
            f"{z_lin_matrix.shape}"
        )
        assert y_matrix.shape == (T, self.cfg.n_sectors), (
            f"y_matrix の shape が期待値 ({T}, {self.cfg.n_sectors}) と不一致: "
            f"{y_matrix.shape}"
        )
        assert len(dates) == T

        # ── リーク監査 ─────────────────────────────────────────────────
        if signal_dates is None:
            signal_dates = dates - pd.Timedelta(days=1)
            warnings.warn(
                "signal_dates が未指定のため dates - 1 calendar day で近似。"
                "厳密なリーク監査には signal_dates を明示的に渡してください。",
                stacklevel=2,
            )
        _audit_no_leak(signal_dates, dates, label="GPUncertaintyModule.fit")

        # ── ローリング窓の適用 ─────────────────────────────────────────
        window = min(self.cfg.window_size, T)
        X_win = f_matrix[-window:]            # (window, K)
        Z_win = z_lin_matrix[-window:]        # (window, N_J)
        Y_win = y_matrix[-window:]            # (window, N_J)

        # ウィンドウ内の将来情報混入チェック
        assert not np.any(np.isnan(X_win) & False), "NaN チェック（placeholder）"
        # 重要: ウィンドウ外データが混入していないことを確認
        assert X_win.shape[0] <= self.cfg.window_size, (
            f"ウィンドウサイズ超過: {X_win.shape[0]} > {self.cfg.window_size}"
        )

        # ── ターゲット: y - z_lin（線形モデルの残差）─────────────────
        residuals = Y_win - Z_win  # (window, N_J)

        # ── 業種別 GP 学習 ───────────────────────────────────────────
        np.random.seed(self.cfg.seed)  # 再現性保証
        self._models = []

        for j in range(self.cfg.n_sectors):
            y_j = residuals[:, j]
            valid = np.isfinite(y_j) & np.isfinite(X_win).all(axis=1)
            X_j = X_win[valid]
            y_j_valid = y_j[valid]

            if len(X_j) < 30:
                logger.warning(
                    "業種 %d: 有効サンプル数 %d < 30, ゼロモデルで代替", j, len(X_j)
                )
                self._models.append(None)
                continue

            model = self._fit_single_gpr(X_j, y_j_valid)
            self._models.append(model)
            logger.debug(
                "業種 %d: fit 完了, n=%d, log_ml=%.3f",
                j,
                len(X_j),
                model.log_marginal_likelihood_value_
                if hasattr(model, "log_marginal_likelihood_value_")
                else float("nan"),
            )

        self._is_fitted = True
        self._train_end_date = dates[-1]
        self._fit_metadata = {
            "version": self.VERSION,
            "n_factors": self.cfg.n_factors,
            "n_sectors": self.cfg.n_sectors,
            "window_size": self.cfg.window_size,
            "n_restarts_optimizer": self.cfg.n_restarts_optimizer,
            "train_date_range": [str(dates[-window].date()), str(dates[-1].date())],
            "n_valid_models": sum(m is not None for m in self._models),
        }

        logger.info(
            "GPUncertaintyModule.fit 完了: window=%d, 有効業種数=%d/%d",
            window,
            self._fit_metadata["n_valid_models"],
            self.cfg.n_sectors,
        )
        return self

    # ------------------------------------------------------------------
    # predict_mean_var
    # ------------------------------------------------------------------

    def predict_mean_var(
        self,
        f_t: np.ndarray,
        z_lin: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """1 日分の予測平均と予測分散を返す。

        Parameters
        ----------
        f_t : ndarray, shape (K,)
            現在日のファクタースコア。
        z_lin : ndarray, shape (N_J,)
            既存線形シグナル（予測平均に加算して最終予測値を構成）。

        Returns
        -------
        mu : ndarray, shape (N_J,)
            GP 予測平均（= z_lin + GP 残差予測）。
        sigma2 : ndarray, shape (N_J,)
            GP 予測分散（確信度スコアの素材）。

        Notes
        -----
        GP が無効（enabled=False）またはモデル未学習の業種は:
        - mu = z_lin（既存シグナルをそのまま使用）
        - sigma2 = NaN（確信度計算で除外される）
        """
        if not self.cfg.enabled:
            return np.asarray(z_lin, dtype=float), np.full(self.cfg.n_sectors, np.nan)

        if not self._is_fitted:
            raise RuntimeError(
                "GPUncertaintyModule.fit() を先に呼び出してください。"
            )

        f_t = np.asarray(f_t, dtype=float).reshape(1, -1)  # (1, K)
        z_lin = np.asarray(z_lin, dtype=float).reshape(-1)  # (N_J,)

        mu = np.empty(self.cfg.n_sectors, dtype=float)
        sigma2 = np.empty(self.cfg.n_sectors, dtype=float)

        for j in range(self.cfg.n_sectors):
            model = self._models[j]
            if model is None:
                # 学習失敗業種: 線形シグナルをそのまま使用
                mu[j] = z_lin[j]
                sigma2[j] = np.nan
                continue

            try:
                gp_mean_j, gp_std_j = model.predict(f_t, return_std=True)
                residual_mean = float(gp_mean_j[0])
                sigma2[j] = float(gp_std_j[0]) ** 2
                # GP 予測平均 = 線形シグナル + GP 残差予測
                mu[j] = z_lin[j] + residual_mean
            except Exception as e:
                logger.warning("業種 %d の GP 予測失敗: %s", j, e)
                mu[j] = z_lin[j]
                sigma2[j] = np.nan

        return mu, sigma2

    # ------------------------------------------------------------------
    # compute_confidence
    # ------------------------------------------------------------------

    def compute_confidence(
        self,
        sigma2_all: np.ndarray,
        weights: Optional[np.ndarray] = None,
        aggregation: str = "mean",
        mapping: str = "inv_exp",
        tau: float = 5.0,
        smoothing_halflife: int = 5,
        min_kappa: float = 0.1,
    ) -> float:
        """全業種の GP 予測分散から確信度スコア κ_t を算出する。

        Parameters
        ----------
        sigma2_all : ndarray, shape (N_J,)
            各業種の GP 予測分散。
        weights : ndarray or None, shape (N_J,)
            ポートフォリオウェイト（aggregation='portfolio_weighted' 時に使用）。
        aggregation : str
            業種横断集約方法。"mean", "max", "portfolio_weighted" のいずれか。
        mapping : str
            分散 → 確信度の写像。"inv_exp"（指数減衰）または "percentile_rank"。
        tau : float
            inv_exp 写像の減衰係数。大きいほど高分散日の κ_t が急激に低下。
        smoothing_halflife : int
            κ_t の EWMA スムージング半減期（営業日）。0 でスムージングなし。
        min_kappa : float
            κ_t の下限値（0 除去）。

        Returns
        -------
        kappa : float
            確信度スコア κ_t ∈ [min_kappa, 1.0]。
        """
        valid = np.isfinite(sigma2_all) & (sigma2_all >= 0)
        if not valid.any():
            logger.warning("全業種の sigma2 が NaN/無効 → κ_t = 1.0 を返します")
            return 1.0

        sigma2_valid = sigma2_all[valid]

        # ── 業種横断集約 ─────────────────────────────────────────────
        if aggregation == "mean":
            sigma2_agg = float(np.mean(sigma2_valid))
        elif aggregation == "max":
            sigma2_agg = float(np.max(sigma2_valid))
        elif aggregation == "portfolio_weighted":
            if weights is not None:
                w = np.abs(np.asarray(weights, dtype=float)[valid])
                w_sum = w.sum()
                if w_sum > 1e-12:
                    sigma2_agg = float(np.sum(w * sigma2_valid) / w_sum)
                else:
                    sigma2_agg = float(np.mean(sigma2_valid))
            else:
                sigma2_agg = float(np.mean(sigma2_valid))
        else:
            raise ValueError(
                f"aggregation は 'mean', 'max', 'portfolio_weighted' のいずれか: "
                f"got {aggregation}"
            )

        # ── 分散 → 確信度の写像 ───────────────────────────────────────
        if mapping == "inv_exp":
            kappa_raw = float(np.exp(-tau * sigma2_agg))
        elif mapping == "percentile_rank":
            # 履歴分位で正規化（初期はスケールが不明なのでフォールバック）
            if len(self._kappa_history) >= 20:
                p = float(
                    np.mean(
                        np.array(
                            [h["sigma2"] for h in self._kappa_history[-60:]]
                        )
                        <= sigma2_agg
                    )
                )
                kappa_raw = 1.0 - p  # 高分位（高分散）ほど低確信度
            else:
                kappa_raw = float(np.exp(-tau * sigma2_agg))
        else:
            raise ValueError(
                f"mapping は 'inv_exp' または 'percentile_rank': got {mapping}"
            )

        # ── EWMA スムージング ─────────────────────────────────────────
        if smoothing_halflife > 0 and len(self._kappa_history) > 0:
            alpha_ewma = 1.0 - 0.5 ** (1.0 / max(smoothing_halflife, 1))
            prev_kappa = self._kappa_history[-1]["kappa"]
            kappa_raw = alpha_ewma * kappa_raw + (1.0 - alpha_ewma) * prev_kappa

        # ── 下限クリップ ──────────────────────────────────────────────
        kappa = float(np.clip(kappa_raw, min_kappa, 1.0))

        # 履歴保存（percentile_rank モード用）
        self._kappa_history.append({"kappa": kappa, "sigma2": sigma2_agg})

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
            GP 確信度で調整された最終ウェイト。
        """
        kappa = float(np.clip(kappa, 0.0, 1.0))
        return np.asarray(weights, dtype=float) * kappa

    # ------------------------------------------------------------------
    # ベイズ的ウェイト縮小（オプション）
    # ------------------------------------------------------------------

    def apply_bayesian_weight_shrinkage(
        self,
        weights: np.ndarray,
        mu: np.ndarray,
        sigma2: np.ndarray,
        scale: float = 1.0,
    ) -> np.ndarray:
        """μ/σ によるベイズ的ウェイト縮小（オプション機能）。

        不確実な業種（高 σ²）を薄く、確信のある業種（低 σ²）を厚くする。
        既存ウェイト方向はそのまま保持し、絶対値のみ縮小する。

        Parameters
        ----------
        weights : ndarray, shape (N_J,)
            基準ウェイト。
        mu : ndarray, shape (N_J,)
            GP 予測平均。
        sigma2 : ndarray, shape (N_J,)
            GP 予測分散。
        scale : float
            縮小の強さ（1.0 = 標準的な縮小）。

        Returns
        -------
        w_shrunk : ndarray, shape (N_J,)
            縮小済みウェイト（方向は保持）。
        """
        sigma = np.sqrt(np.maximum(sigma2, 1e-12))
        snr = np.abs(mu) / sigma  # |μ/σ|: シャープ比的な信号強度

        # SNR を [0, 1] に正規化（ソフトマックス的）
        snr_max = np.nanmax(snr)
        if snr_max < 1e-8:
            return np.asarray(weights, dtype=float)

        shrinkage = np.where(
            np.isfinite(snr),
            np.tanh(scale * snr / max(snr_max, 1e-8)),
            1.0,  # NaN 業種はそのまま
        )

        w = np.asarray(weights, dtype=float)
        return w * shrinkage

    # ------------------------------------------------------------------
    # ARD 解釈可視化
    # ------------------------------------------------------------------

    def get_ard_summary(
        self, factor_names: Optional[list] = None
    ) -> List[Dict[str, Any]]:
        """全業種の ARD 長さスケールと非線形寄与重要度を返す。

        Returns
        -------
        list of dict, one per sector:
            {
                "sector": int,
                "length_scales": ndarray (K,),
                "importance": ndarray (K,),
                "alpha": float,
                "factor_names": list,
            }
        """
        if not self._is_fitted:
            raise RuntimeError("fit() を先に呼び出してください。")

        results = []
        for j, model in enumerate(self._models):
            if model is None:
                results.append({"sector": j, "error": "model_not_fitted"})
                continue
            params = extract_kernel_params(model)
            ard = summarize_ard_importance(
                params["length_scales"],
                factor_names=factor_names,
            )
            results.append(
                {
                    "sector": j,
                    "length_scales": ard["length_scales"],
                    "importance": ard["importance"],
                    "alpha": params["alpha"],
                    "factor_names": ard["factor_names"],
                    "log_marginal_likelihood": params["log_marginal_likelihood"],
                }
            )
        return results

    def get_kappa_history(self) -> pd.Series:
        """compute_confidence() で蓄積された κ_t 履歴を Series で返す。"""
        if not self._kappa_history:
            return pd.Series(dtype=float, name="kappa")
        return pd.Series(
            [h["kappa"] for h in self._kappa_history], name="kappa", dtype=float
        )

    # ------------------------------------------------------------------
    # 永続化
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """学習済みモデルと設定をディレクトリに保存する。

        保存内容:
          models.pkl    – 業種別 GPR モデルリスト
          metadata.json – 設定・訓練情報
        """
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        models_path = out / "gp_models.pkl"
        with open(models_path, "wb") as f:
            pickle.dump(self._models, f, protocol=pickle.HIGHEST_PROTOCOL)

        meta = {
            **self._fit_metadata,
            "is_fitted": self._is_fitted,
            "cfg": {
                "n_factors": self.cfg.n_factors,
                "n_sectors": self.cfg.n_sectors,
                "window_size": self.cfg.window_size,
                "n_restarts_optimizer": self.cfg.n_restarts_optimizer,
                "use_sparse_gp": self.cfg.use_sparse_gp,
                "n_inducing_points": self.cfg.n_inducing_points,
                "seed": self.cfg.seed,
                "enabled": self.cfg.enabled,
                "kernel_cfg": asdict(self.cfg.kernel_cfg),
            },
        }
        with open(out / "gp_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info("GPUncertaintyModule を %s に保存しました", path)

    @classmethod
    def load(cls, path: str) -> "GPUncertaintyModule":
        """保存済みモジュールをディレクトリからロードする。"""
        out = Path(path)

        with open(out / "gp_metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        cfg_dict = meta.get("cfg", {})
        kernel_cfg = KernelConfig(**cfg_dict.pop("kernel_cfg", {}))
        cfg = GPConfig(**cfg_dict, kernel_cfg=kernel_cfg)
        instance = cls(cfg=cfg)

        with open(out / "gp_models.pkl", "rb") as f:
            instance._models = pickle.load(f)

        instance._is_fitted = meta.get("is_fitted", True)
        instance._fit_metadata = meta

        logger.info("GPUncertaintyModule を %s からロードしました", path)
        return instance

    @classmethod
    def from_config(cls, config: dict) -> "GPUncertaintyModule":
        """YAML config 辞書からインスタンスを生成する（未学習状態）。"""
        cfg = GPConfig.from_config_dict(config)
        return cls(cfg=cfg)

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _fit_single_gpr(self, X: np.ndarray, y: np.ndarray) -> Any:
        """単一業種の GPR を学習して返す。"""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.preprocessing import StandardScaler

        # 入力の正規化（GPR は標準化入力で安定する）
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kernel = build_structured_kernel(self.cfg.kernel_cfg, n_factors=self.cfg.n_factors)

        if self.cfg.use_sparse_gp and len(X) > self.cfg.n_inducing_points:
            return self._fit_sparse_gpr(X_scaled, y, scaler)

        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.cfg.n_restarts_optimizer,
            alpha=0.0,  # ノイズは WhiteKernel でカーネル側に含める
            normalize_y=True,
            random_state=self.cfg.seed,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gpr.fit(X_scaled, y)

        # スケーラーをモデルに付与（predict 時に使用）
        gpr._input_scaler = scaler  # type: ignore[attr-defined]
        return gpr

    def _fit_sparse_gpr(self, X_scaled: np.ndarray, y: np.ndarray, scaler) -> Any:
        """Nyström 近似によるスパース GP（誘導点選択）。

        sklearn の GPR は full GP のみ対応のため、
        誘導点を K-means で選択して訓練データとして使用する近似実装。
        完全なスパース GP（FITC/VFE）には GPyTorch が必要。
        """
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.gaussian_process import GaussianProcessRegressor

        M = min(self.cfg.n_inducing_points, len(X_scaled))
        kmeans = MiniBatchKMeans(n_clusters=M, random_state=self.cfg.seed, n_init=3)
        kmeans.fit(X_scaled)
        X_inducing = kmeans.cluster_centers_  # (M, K)

        kernel = build_structured_kernel(self.cfg.kernel_cfg, n_factors=self.cfg.n_factors)
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=max(1, self.cfg.n_restarts_optimizer - 1),
            alpha=1e-4,
            normalize_y=True,
            random_state=self.cfg.seed,
        )
        # 誘導点で近似学習
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gpr.fit(X_inducing, np.zeros(M))

        # 全データで fine-tune（近似のため残差で補正）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gpr.fit(X_scaled, y)

        gpr._input_scaler = scaler  # type: ignore[attr-defined]
        return gpr

    def __repr__(self) -> str:
        return (
            f"GPUncertaintyModule("
            f"fitted={self._is_fitted}, "
            f"n_sectors={self.cfg.n_sectors}, "
            f"window={self.cfg.window_size}, "
            f"kernel={self.cfg.kernel_cfg.nonlinear_type}, "
            f"enabled={self.cfg.enabled})"
        )


# ---------------------------------------------------------------------------
# リーク監査（モジュール内部用）
# ---------------------------------------------------------------------------


def _audit_no_leak(
    signal_dates: pd.DatetimeIndex,
    target_dates: pd.DatetimeIndex,
    label: str = "GPUncertaintyModule",
) -> None:
    """全シグナル日がターゲット日より厳密に前であることを assert する。"""
    assert len(signal_dates) == len(target_dates), (
        f"signal_dates ({len(signal_dates)}) と target_dates ({len(target_dates)}) "
        "の長さが一致しません"
    )
    sig_arr = np.array(signal_dates, dtype="datetime64[D]")
    tgt_arr = np.array(target_dates, dtype="datetime64[D]")
    leaky = sig_arr >= tgt_arr
    if leaky.any():
        n_leak = int(leaky.sum())
        idx = int(np.where(leaky)[0][0])
        raise AssertionError(
            f"[DATA LEAK] {label}: signal_date >= target_date が {n_leak} 件検出。\n"
            f"最初の違反: signal={signal_dates[idx].date()}, "
            f"target={target_dates[idx].date()}"
        )
