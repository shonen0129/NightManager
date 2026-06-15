"""domain.gp.calibration – GP 不確実性の較正テスト

不確実性較正（Calibration）とは
---------------------------------
GP が「80% 予測区間」と言うとき、実際に 80% の実現値が
その区間に収まれば「較正良好」と判断する。

較正が崩れている場合:
- 過信（overconfident）: 実際のカバレッジ < 期待値
  → GP は実際より予測分散を過小評価している
  → 確信度スコアの信頼性が低い（過度に高い κ_t）
- 過疎（underconfident）: 実際のカバレッジ > 期待値
  → GP は実際より予測分散を過大評価している
  → κ_t が必要以上に低下し、利益機会を失う

較正評価結果が崩れている場合は「WARN」または「REJECT」を明示する。

使用例
------
>>> result = coverage_test(mu, sigma2, y_realized, levels=[0.80, 0.90])
>>> if result["calibration_status"] == "WARN":
...     print("較正崩れ: GP 不確実性の信頼性に注意")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CalibrationResult
# ---------------------------------------------------------------------------


@dataclass
class CalibrationResult:
    """較正テストの結果コンテナ。

    Attributes
    ----------
    levels : list of float
        テストした予測区間の信頼水準。
    expected_coverages : list of float
        各水準の期待カバレッジ（= 水準値そのもの）。
    actual_coverages : list of float
        実際のカバレッジ（実現値が区間内に収まった割合）。
    coverage_errors : list of float
        actual - expected の差分。正 = 過疎、負 = 過信。
    calibration_status : str
        "GOOD", "WARN", "REJECT" のいずれか。
    warn_threshold : float
        |error| のこの閾値を超えたら WARN/REJECT。
    n_samples : int
        テストに使用したサンプル数（業種 × 日数）。
    summary : str
        人間が読みやすい要約文。
    extra : dict
        追加情報（業種別カバレッジなど）。
    """

    levels: List[float] = field(default_factory=lambda: [0.80])
    expected_coverages: List[float] = field(default_factory=list)
    actual_coverages: List[float] = field(default_factory=list)
    coverage_errors: List[float] = field(default_factory=list)
    calibration_status: str = "GOOD"
    warn_threshold: float = 0.05
    n_samples: int = 0
    summary: str = ""
    extra: dict = field(default_factory=dict)

    def is_well_calibrated(self, threshold: Optional[float] = None) -> bool:
        """較正良好かどうかを返す。"""
        thr = threshold if threshold is not None else self.warn_threshold
        return all(abs(e) <= thr for e in self.coverage_errors)

    def __str__(self) -> str:
        sep = "=" * 60
        lines = [
            sep,
            f"GP 不確実性較正テスト結果: {self.calibration_status}",
            sep,
            f"サンプル数: {self.n_samples}",
            f"{'信頼水準':<12} {'期待カバレッジ':>16} {'実績カバレッジ':>16} {'誤差':>10}",
            "-" * 58,
        ]
        for lvl, exp, act, err in zip(
            self.levels,
            self.expected_coverages,
            self.actual_coverages,
            self.coverage_errors,
        ):
            flag = " ⚠" if abs(err) > self.warn_threshold else ""
            lines.append(
                f"{lvl*100:.0f}%{'':<8} {exp*100:>14.1f}% {act*100:>15.1f}% "
                f"{err*100:>+9.1f}%{flag}"
            )
        lines.append(sep)
        lines.append(self.summary)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# coverage_test: メイン較正テスト
# ---------------------------------------------------------------------------


def coverage_test(
    mu: np.ndarray,
    sigma2: np.ndarray,
    y_realized: np.ndarray,
    levels: Optional[List[float]] = None,
    warn_threshold: float = 0.05,
) -> CalibrationResult:
    """GP 予測区間のカバレッジを検証する（較正テスト）。

    Parameters
    ----------
    mu : ndarray, shape (T, N_J) or (N_J,)
        GP 予測平均。時系列パネルまたは単一日。
    sigma2 : ndarray, shape (T, N_J) or (N_J,)
        GP 予測分散。
    y_realized : ndarray, shape (T, N_J) or (N_J,)
        実現リターン。
    levels : list of float or None
        テストする信頼水準。None の場合 [0.80, 0.90, 0.95]。
    warn_threshold : float
        |実績カバレッジ - 期待カバレッジ| の警告閾値。デフォルト 0.05（5pp）。

    Returns
    -------
    CalibrationResult

    Notes
    -----
    予測区間: μ ± z_α/2 × σ（正規分布を仮定）
    z_0.80 = 1.282, z_0.90 = 1.645, z_0.95 = 1.960

    GP が正しく較正されているとき、各水準のカバレッジは期待値と一致する。
    |実績 - 期待| > warn_threshold のとき較正崩れとして WARN/REJECT を返す。
    """
    from scipy.stats import norm

    if levels is None:
        levels = [0.80, 0.90, 0.95]

    # 配列化・フラット化
    mu_flat = np.asarray(mu, dtype=float).reshape(-1)
    sigma2_flat = np.asarray(sigma2, dtype=float).reshape(-1)
    y_flat = np.asarray(y_realized, dtype=float).reshape(-1)

    # 有効サンプルのみ使用
    sigma_flat = np.sqrt(np.maximum(sigma2_flat, 1e-12))
    valid = np.isfinite(mu_flat) & np.isfinite(sigma2_flat) & np.isfinite(y_flat) & (sigma2_flat > 0)
    n_valid = int(valid.sum())

    if n_valid < 10:
        result = CalibrationResult(
            levels=levels,
            expected_coverages=levels,
            actual_coverages=[float("nan")] * len(levels),
            coverage_errors=[float("nan")] * len(levels),
            calibration_status="WARN",
            warn_threshold=warn_threshold,
            n_samples=n_valid,
            summary="サンプル数不足（< 10）のため較正テスト不可。",
        )
        return result

    mu_v = mu_flat[valid]
    sigma_v = sigma_flat[valid]
    y_v = y_flat[valid]

    actual_coverages = []
    coverage_errors = []

    for lvl in levels:
        z = float(norm.ppf(0.5 + lvl / 2.0))  # 両側区間: z_{(1+lvl)/2}
        lower = mu_v - z * sigma_v
        upper = mu_v + z * sigma_v
        in_interval = (y_v >= lower) & (y_v <= upper)
        actual_cov = float(in_interval.mean())
        actual_coverages.append(actual_cov)
        coverage_errors.append(actual_cov - lvl)

    # ── 判定 ─────────────────────────────────────────────────────────────
    max_abs_error = max(abs(e) for e in coverage_errors)
    if max_abs_error > 2 * warn_threshold:
        status = "REJECT"
    elif max_abs_error > warn_threshold:
        status = "WARN"
    else:
        status = "GOOD"

    # ── 要約文 ───────────────────────────────────────────────────────────
    summary_parts = []
    for lvl, act, err in zip(levels, actual_coverages, coverage_errors):
        direction = "過信（予測分散過小）" if err < 0 else "過疎（予測分散過大）"
        if abs(err) > warn_threshold:
            summary_parts.append(
                f"[{lvl*100:.0f}%区間] 実績={act*100:.1f}% ({err*100:+.1f}pp): {direction}"
            )

    if status == "GOOD":
        summary = f"較正良好: 全水準で |誤差| ≤ {warn_threshold*100:.0f}pp (n={n_valid})"
    elif status == "WARN":
        summary = (
            f"[WARN] 較正に軽微な崩れがあります (n={n_valid}):\n"
            + "\n".join(f"  • {s}" for s in summary_parts)
            + "\n→ 確信度スコアの信頼性に注意が必要です。"
        )
    else:
        summary = (
            f"[REJECT] 較正が大きく崩れています (n={n_valid}):\n"
            + "\n".join(f"  • {s}" for s in summary_parts)
            + "\n→ GP の予測分散を確信度として使用するには較正不良。採用非推奨。"
        )

    return CalibrationResult(
        levels=levels,
        expected_coverages=list(levels),
        actual_coverages=actual_coverages,
        coverage_errors=coverage_errors,
        calibration_status=status,
        warn_threshold=warn_threshold,
        n_samples=n_valid,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# coverage_test_by_sector: 業種別の較正テスト
# ---------------------------------------------------------------------------


def coverage_test_by_sector(
    mu: np.ndarray,
    sigma2: np.ndarray,
    y_realized: np.ndarray,
    level: float = 0.80,
    warn_threshold: float = 0.05,
) -> pd.DataFrame:
    """業種別の予測区間カバレッジを計算する。

    Parameters
    ----------
    mu : ndarray, shape (T, N_J)
    sigma2 : ndarray, shape (T, N_J)
    y_realized : ndarray, shape (T, N_J)
    level : float
        信頼水準（例: 0.80 → 80% 区間）。
    warn_threshold : float

    Returns
    -------
    pd.DataFrame, shape (N_J,)
        各業種の実績カバレッジ、誤差、判定。
    """
    from scipy.stats import norm

    mu = np.asarray(mu, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    y = np.asarray(y_realized, dtype=float)

    assert mu.ndim == 2, "mu は shape (T, N_J) の 2 次元配列"
    T, N_J = mu.shape
    sigma = np.sqrt(np.maximum(sigma2, 1e-12))
    z = float(norm.ppf(0.5 + level / 2.0))

    rows = []
    for j in range(N_J):
        mu_j = mu[:, j]
        sigma_j = sigma[:, j]
        y_j = y[:, j]
        valid = np.isfinite(mu_j) & np.isfinite(sigma_j) & np.isfinite(y_j) & (sigma2[:, j] > 0)
        n_j = int(valid.sum())
        if n_j < 5:
            rows.append({
                "sector": j, "n_valid": n_j,
                "coverage": float("nan"), "error": float("nan"), "status": "INSUFFICIENT"
            })
            continue
        lower = mu_j[valid] - z * sigma_j[valid]
        upper = mu_j[valid] + z * sigma_j[valid]
        cov = float(((y_j[valid] >= lower) & (y_j[valid] <= upper)).mean())
        err = cov - level
        status = "GOOD" if abs(err) <= warn_threshold else ("WARN" if abs(err) <= 2 * warn_threshold else "REJECT")
        rows.append({"sector": j, "n_valid": n_j, "coverage": cov, "error": err, "status": status})

    return pd.DataFrame(rows).set_index("sector")


# ---------------------------------------------------------------------------
# reliability_diagram: 較正チャート（可視化）
# ---------------------------------------------------------------------------


def reliability_diagram(
    mu: np.ndarray,
    sigma2: np.ndarray,
    y_realized: np.ndarray,
    n_bins: int = 10,
    title: str = "GP 不確実性較正チャート",
):
    """較正チャート（Reliability Diagram）を生成する。

    横軸: 期待カバレッジ（0〜1）
    縦軸: 実績カバレッジ（0〜1）
    対角線: 完全較正ライン

    Parameters
    ----------
    mu, sigma2, y_realized : ndarray
        GP 予測値と実現値（flat でも (T, N_J) でも可）。
    n_bins : int
        信頼水準のビン数。
    title : str
        チャートタイトル。

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    mu_f = np.asarray(mu, dtype=float).reshape(-1)
    sigma2_f = np.asarray(sigma2, dtype=float).reshape(-1)
    y_f = np.asarray(y_realized, dtype=float).reshape(-1)
    sigma_f = np.sqrt(np.maximum(sigma2_f, 1e-12))

    valid = np.isfinite(mu_f) & np.isfinite(sigma2_f) & np.isfinite(y_f) & (sigma2_f > 0)
    mu_v, sigma_v, y_v = mu_f[valid], sigma_f[valid], y_f[valid]

    if len(mu_v) < 10:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "サンプル不足", ha="center", va="center", transform=ax.transAxes)
        return fig

    levels = np.linspace(0.05, 0.95, n_bins)
    actual_covs = []
    for lvl in levels:
        z = float(norm.ppf(0.5 + lvl / 2.0))
        in_iv = (y_v >= mu_v - z * sigma_v) & (y_v <= mu_v + z * sigma_v)
        actual_covs.append(float(in_iv.mean()))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="完全較正")
    ax.plot(levels, actual_covs, "o-", color="steelblue", linewidth=2.0, label="GP 実績")
    ax.fill_between(levels, levels - 0.05, levels + 0.05, alpha=0.1, color="gray", label="±5pp 許容域")
    ax.set_xlabel("期待カバレッジ")
    ax.set_ylabel("実績カバレッジ")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
