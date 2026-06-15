"""domain.gp.kernels – 構造化カーネル設計モジュール

カーネル設計要件
-----------------
k(f, f') = k_linear(f, f') + α * k_nonlin(f, f')

- k_linear  : DotProductカーネル（線形）。既存線形シグナル z_lin と等価な骨格を担う。
- k_nonlin  : RBF または Matérn with ARD（自動適合度判定）。
              各ファクター次元に独立の長さスケールを持たせ、
              どのファクターが非線形性に寄与するかを学習・可視化可能にする。
- α         : 非線形項の寄与重み。小さい初期値から始め、上限を課す。

過学習防止制約
--------------
- α の上限 ≤ alpha_bounds[1]（デフォルト 0.5）
  → 非線形項が線形骨格を支配しないよう上界制約
- 長さスケールの下限 ≥ length_scale_bounds[0]（デフォルト 0.3）
  → 過小スケールによる短距離過適合を防止
- ノイズ分散の上下限も制約付き

使用例
------
>>> from domain.gp.kernels import build_structured_kernel, KernelConfig
>>> cfg = KernelConfig()
>>> kernel = build_structured_kernel(cfg, n_factors=6)
>>> # GPRに渡す
>>> from sklearn.gaussian_process import GaussianProcessRegressor
>>> gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# KernelConfig: カーネルのハイパーパラメータ設定
# ---------------------------------------------------------------------------


@dataclass
class KernelConfig:
    """GP構造化カーネルの設定パラメータ。

    Attributes
    ----------
    linear_sigma0 : float
        DotProduct カーネルの固定項 σ₀。0 のとき純粋な内積（線形）カーネル。
    alpha_init : float
        非線形項 k_nonlin の初期スケール係数 α。
    alpha_bounds : tuple[float, float]
        α の (下限, 上限)。上限 ≤ 0.5 で線形骨格の支配を防ぐ。
    nonlinear_type : str
        非線形カーネルの種類。``"rbf"`` または ``"matern"``。
    matern_nu : float
        Matérn カーネルの平滑度パラメータ（nonlinear_type="matern" 時のみ有効）。
        1.5 または 2.5 が推奨。
    length_scale_init : float
        ARD 長さスケールの初期値（全ファクター共通初期値）。
    length_scale_bounds : tuple[float, float]
        ARD 長さスケールの (下限, 上限)。下限 ≥ 0.3 で過学習防止。
    noise_level_init : float
        観測ノイズの初期分散。
    noise_bounds : tuple[float, float]
        ノイズ分散の (下限, 上限)。
    """

    linear_sigma0: float = 0.0
    alpha_init: float = 0.1
    alpha_bounds: Tuple[float, float] = (1e-5, 0.5)
    nonlinear_type: str = "rbf"  # "rbf" or "matern"
    matern_nu: float = 2.5
    length_scale_init: float = 1.0
    length_scale_bounds: Tuple[float, float] = (0.3, 10.0)
    noise_level_init: float = 0.01
    noise_bounds: Tuple[float, float] = (1e-5, 1.0)

    def __post_init__(self) -> None:
        assert self.alpha_bounds[1] <= 1.0, (
            f"alpha_bounds 上限は 1.0 以下にしてください。"
            f"（非線形項が線形骨格を支配しないよう）: got {self.alpha_bounds[1]}"
        )
        assert self.length_scale_bounds[0] >= 0.1, (
            f"length_scale_bounds 下限は 0.1 以上にしてください。"
            f"（過小スケールによる過学習防止）: got {self.length_scale_bounds[0]}"
        )
        assert self.nonlinear_type in {"rbf", "matern"}, (
            f"nonlinear_type は 'rbf' または 'matern' のみ対応: got {self.nonlinear_type}"
        )
        assert self.alpha_init <= self.alpha_bounds[1], (
            f"alpha_init ({self.alpha_init}) は alpha_bounds 上限 ({self.alpha_bounds[1]}) 以下にしてください"
        )

    @classmethod
    def from_config_dict(cls, d: dict) -> "KernelConfig":
        """YAML config dict から生成する。"""
        return cls(
            linear_sigma0=float(d.get("linear_sigma0", 0.0)),
            alpha_init=float(d.get("alpha_init", 0.1)),
            alpha_bounds=tuple(d.get("alpha_bounds", [1e-5, 0.5])),  # type: ignore[arg-type]
            nonlinear_type=str(d.get("nonlinear_type", "rbf")),
            matern_nu=float(d.get("matern_nu", 2.5)),
            length_scale_init=float(d.get("length_scale_init", 1.0)),
            length_scale_bounds=tuple(d.get("length_scale_bounds", [0.3, 10.0])),  # type: ignore[arg-type]
            noise_level_init=float(d.get("noise_level_init", 0.01)),
            noise_bounds=tuple(d.get("noise_bounds", [1e-5, 1.0])),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# カーネル構築関数
# ---------------------------------------------------------------------------


def build_structured_kernel(
    cfg: KernelConfig,
    n_factors: int = 6,
):
    """構造化カーネル k = k_linear + α * k_nonlin を構築して返す。

    scikit-learn の Kernel オブジェクトを返す。周辺尤度最大化時に
    sklearn.gaussian_process.GaussianProcessRegressor がハイパーパラメータを最適化する。

    Parameters
    ----------
    cfg : KernelConfig
        カーネル設定。
    n_factors : int
        ファクタースコアの次元数 K（ARD の長さスケール数に等しい）。

    Returns
    -------
    kernel : sklearn.gaussian_process.kernels.Kernel
        k_linear + alpha_scale * k_nonlin の構成済みカーネル。

    Notes
    -----
    sklearn の DotProduct カーネルは k(x, x') = σ₀² + x·x' の形。
    sigma_0=0 のとき純粋な内積カーネルになり、線形回帰と等価な骨格を担う。

    ConstantKernel で α をパラメータ化することで、
    周辺尤度最大化で α が自動推定される（bounds で上限制約）。
    """
    from sklearn.gaussian_process.kernels import (
        ConstantKernel,
        DotProduct,
        Matern,
        RBF,
        WhiteKernel,
    )

    # k_linear: 線形カーネル（既存線形モデルと等価な骨格）
    k_linear = DotProduct(sigma_0=cfg.linear_sigma0, sigma_0_bounds="fixed")

    # k_nonlin: 非線形項（ARD – 各次元に独立の長さスケール）
    ls_init = np.full(n_factors, cfg.length_scale_init)
    ls_bounds = (cfg.length_scale_bounds[0], cfg.length_scale_bounds[1])

    if cfg.nonlinear_type == "rbf":
        k_nonlin_base = RBF(
            length_scale=ls_init,
            length_scale_bounds=[ls_bounds] * n_factors,
        )
    else:  # matern
        k_nonlin_base = Matern(
            length_scale=ls_init,
            length_scale_bounds=[ls_bounds] * n_factors,
            nu=cfg.matern_nu,
        )

    # α スケーリング: ConstantKernel で α をパラメータ化
    alpha_scale = ConstantKernel(
        constant_value=cfg.alpha_init,
        constant_value_bounds=cfg.alpha_bounds,
    )
    k_nonlin = alpha_scale * k_nonlin_base

    # 観測ノイズ: WhiteKernel（GPRの noise 引数ではなくカーネル側で管理）
    k_noise = WhiteKernel(
        noise_level=cfg.noise_level_init,
        noise_level_bounds=cfg.noise_bounds,
    )

    # 構造化カーネル = 線形 + α * 非線形 + ノイズ
    kernel = k_linear + k_nonlin + k_noise

    return kernel


# ---------------------------------------------------------------------------
# ハイパーパラメータ抽出ユーティリティ
# ---------------------------------------------------------------------------


def extract_kernel_params(fitted_gpr) -> dict:
    """学習済み GPR から主要ハイパーパラメータを辞書で返す。

    Parameters
    ----------
    fitted_gpr : GaussianProcessRegressor
        fit() 済みの GPR インスタンス。

    Returns
    -------
    dict with keys:
        "alpha"          : float – 非線形項のスケール係数
        "length_scales"  : ndarray (K,) – ARD 長さスケール
        "noise_level"    : float – 観測ノイズ分散
        "linear_var"     : float – 線形項の σ₀²（fixed=0 のとき 0）
        "nonlinear_type" : str
        "log_marginal_likelihood" : float
    """
    kernel = fitted_gpr.kernel_

    params: dict = {
        "alpha": float("nan"),
        "length_scales": np.array([float("nan")]),
        "noise_level": float("nan"),
        "linear_var": float("nan"),
        "log_marginal_likelihood": float(fitted_gpr.log_marginal_likelihood_value_)
        if hasattr(fitted_gpr, "log_marginal_likelihood_value_")
        else float("nan"),
    }

    # カーネルパラメータをトラバース
    for name, val in kernel.get_params(deep=True).items():
        if "constant_value" in name and "bounds" not in name:
            params["alpha"] = float(val)
        if "length_scale" in name and "bounds" not in name:
            arr = np.atleast_1d(np.asarray(val, dtype=float))
            params["length_scales"] = arr
        if "noise_level" in name and "bounds" not in name:
            params["noise_level"] = float(val)
        if "sigma_0" in name and "bounds" not in name:
            params["linear_var"] = float(val) ** 2

    return params


def summarize_ard_importance(
    length_scales: np.ndarray,
    factor_names: Optional[list[str]] = None,
) -> dict:
    """ARD 長さスケールから各ファクターの非線形寄与重要度を計算する。

    長さスケールが小さいほど、そのファクターの非線形効果が大きい。
    重要度 = 1 / length_scale（正規化済み）

    Parameters
    ----------
    length_scales : ndarray (K,)
        各ファクターの ARD 長さスケール。
    factor_names : list of str or None
        ファクター名（省略時は "f_0", "f_1", ... を使用）。

    Returns
    -------
    dict : {"factor_names": list, "importance": ndarray, "length_scales": ndarray}
    """
    K = len(length_scales)
    names = factor_names if factor_names is not None else [f"f_{k}" for k in range(K)]
    importance = 1.0 / np.maximum(length_scales, 1e-8)
    importance = importance / importance.sum()  # 正規化
    return {
        "factor_names": names,
        "importance": importance,
        "length_scales": length_scales,
    }
