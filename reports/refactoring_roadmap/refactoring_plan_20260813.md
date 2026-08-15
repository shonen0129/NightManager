# V2 本番リファクタリング 実装仕様書

> 作成日: 2026-08-13
> 最終更新: 2026-08-13
> 目的: 一から作り直すかリファクタリングするかの議論を踏まえ、段階的リファクタリングの**推奨実装仕様**を定義する

---

## 1. 結論：変更内容は一意に定まるか？

**結論：全てが一意に定まるわけではない。推奨デフォルトを定め、実装者の判断余地を残す。**

以下の2種類に分ける。

### 1.1 ほぼ確定的な変更（"must"）

これらは、現行アーキテクチャの問題を解消するための**制約条件**であり、逸脱するとテスト・監査・運用で破綻する。

| 変更 | 理由 | 実装先 | 制約 |
|---|---|---|---|
| `research` パッケージへの本番依存を断つ | 本番パスが研究パッケージを import している構造的欠陥 | `src/leadlag/models/blpx.py` 新設 | 本番/Step2 パスが `research` を import しない |
| gap 行列計算を on-demand 化 | ファイル駆動による「当日 gap 行列不在 → flat」という運用事故リスク | `src/leadlag/core/gap_adjustment.py` 新設 | Step 1 BLPX シグナルブロックを `compute_blp_signal(return_matrices=True)` から再構成 |
| `production.yaml` のフラット化 | `_flatten_nested_yaml` による設定解決の複雑さ | `configs/production/production.yaml` 修正 | `ProductionV2RunConfig._flatten_nested_yaml` を廃止 |
| `ProductionV2Model` をクラス中心に整理 | 現行は関数ベースで薄いラッパー | `src/leadlag/models/production_v2.py` 修正 | `generate_v2_production_portfolio` の入出力を `decide` へ集約 |

### 1.2 設計選択が必要な変更（"decision"）

これらは複数の正解があり、ここでは**推奨値をデフォルトとして提示**する。異なる選択をする場合は、その理由を文書化し、テストで担保する。

| 選択肢 | 推奨デフォルト | 代替案 |
|---|---|---|
| gap 補正計算の配置 | `src/leadlag/core/gap_adjustment.py` | `src/leadlag/models/gap_adjustment.py` |
| `.npy` ファイルの扱い | 監査証跡・キャッシュとして残す | 完全に削除し SQLite のみにする |
| `SectorRelativeEnsembleBLPEnhancedModel` の改名 | `ProductionBLPXModel` | クラス名は維持 `SectorRelativeEnsembleBLPEnhancedModel` |
| `blp_base.py` の置き場 | `src/leadlag/models/blp_base.py` | `src/leadlag/core/blp_base.py` |
| 設定モデルの分割 | `ProductionConfig` + `BacktestConfig` | `AppConfig` 一本化（v2 フィールド名変更） |
| データレイヤー抽象化の実装方式 | `DataProvider` ABC | `MarketData` ファサード |
| `blpx_model` 初期化場所 | `ProductionRunner.__init__` | `ProductionV2Model.__init__` の optional factory |
| `BLPXConfig` 値の使い分け | `gap_open_coef` / `gap_open_coef_neg` の US-direction 選択を `compute_distribution` で実施 | `ProductionBLPXModel` 内部で隠蔽 |
| Step 1 `Omega_struct` 計算方式 | `compute_blp_signal(return_matrices=True)` から `_omega_from_blp_res` で計算 | Step 1 ファイルを on-demand でも読む |

本書では **推奨デフォルト** に基づいた実装仕様を記述する。

---

## 2. 不変条件（変更禁止）

リファクタリング中も以下を絶対に維持する。これらは `AGENTS.md` の「不変条件」に基づく。

| # | 不変条件 | 確認方法 | 責任ファイル |
|---|---|---|---|
| 1 | ルックアヘッド禁止 | `test_leakage_audit.py` | `src/leadlag/compliance/v2_auditor.py` |
| 2 | ベースライン期間 2010-2014 固定 | テスト・コード検索 | `src/leadlag/core/pipeline.py`, `src/leadlag/core/correlation.py` |
| 3 | バックテスト開始日 2015-01-05 以降 | `config/schemas.py`, `BacktestEngine` | `src/leadlag/execution/backtester.py` |
| 4 | 市場中立 net ±0.05、gross ≤ 2.0 | `test_portfolio.py`, `test_risk.py` | `src/leadlag/core/portfolio.py` |
| 5 | ティッカー定義の一元化 | `test_ticker_registry.py` | `src/leadlag/data/tickers.py` |
| 6 | gap 行列欠損時はフラットポジション | `test_production_v2.py` | `src/leadlag/models/production_v2.py` |
| 7 | 前日 gap 行列の使用禁止 | コードレビュー | `src/leadlag/utils/gap_matrix_io.py` |
| 8 | 全テスト pass | `bash scripts/run_tests_parallel.sh` | 全テスト |

---

## 3. Phase 0: 事前準備（計測と基準化）

### 3.1 目的

リファクタリング前後で振る舞いが一致することを **数値として** 検証できる状態を作る。

### 3.2 作成するファイルとその内容

#### 3.2.1 `tests/regression/__init__.py`

空ファイル。

#### 3.2.2 `tests/regression/conftest.py`

```python
"""Regression test fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

@pytest.fixture(scope="session")
def regression_baseline_dir() -> Path:
    path = Path(__file__).parent / "baselines"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def regression_df_exec(sample_df_exec) -> pd.DataFrame:
    """Use the existing sample_df_exec fixture.

    ``sample_df_exec`` returns ``(df_exec, raw_data)``; we only need ``df_exec``.
    """
    df_exec, _ = sample_df_exec
    return df_exec
```

#### 3.2.3 `tests/regression/test_v2_baseline.py`

```python
"""Capture and verify baseline V2 behavior."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.config import load_config_from_yaml
from leadlag.models.production_v2 import ProductionV2Model
from leadlag.utils.gap_matrix_io import load_gap_matrices

from .conftest import regression_baseline_dir


BASELINE_VERSION = "v20260813"


def _build_current_prices_from_df_exec(
    df_exec: pd.DataFrame,
    trade_date: str,
) -> dict[str, float] | None:
    """Build 09:10 current prices dict from ``jp_open_trade_*`` columns.

    ``preprocessor.py`` writes ``jp_open_trade_{ticker}`` (09:10 midpoint or
    open) for each JP ticker. If the column is missing, fall back to
    ``jp_close_{ticker} * (1 + jp_gap_{ticker})``. Return None when neither
    source is available so that the caller can fall back to the file cache.
    """
    if trade_date not in df_exec.index:
        return None
    row = df_exec.loc[trade_date]
    prices = {}
    for t in JP_TICKERS:
        open_col = f"jp_open_trade_{t}"
        gap_col = f"jp_gap_{t}"
        close_col = f"jp_close_{t}"
        if open_col in row.index and not pd.isna(row[open_col]):
            prices[t] = float(row[open_col])
        elif gap_col in row.index and not pd.isna(row[gap_col]) \
                and close_col in row.index and not pd.isna(row[close_col]):
            prices[t] = float(row[close_col]) * (1.0 + float(row[gap_col]))
    if len(prices) == len(JP_TICKERS):
        return prices
    return None


def _capture_v2_snapshot(
    df_exec: pd.DataFrame,
    trade_date: str,
    gap_input_dir: Path,
    current_prices: dict[str, float] | None = None,
    config_path: str = "configs/production/production.yaml",
) -> dict:
    app_config = load_config_from_yaml(config_path)
    model = ProductionV2Model(app_config.v2)
    result = model.decide(
        trade_date=trade_date,
        gap_input_dir=gap_input_dir,
        df_exec=df_exec,
        current_prices=current_prices,
    )
    return {
        "w_final": result["w_final"].tolist(),
        "scores": result["scores"].tolist(),
        "pit_binning": result["pit_binning"],
        "summary": {k: v for k, v in result["summary"].items() if k not in (
            "trade_date", "version", "candidate"
        )},
    }


def test_v2_snapshot_matches_baseline(
    regression_baseline_dir: Path,
    regression_df_exec: pd.DataFrame,
) -> None:
    """Compare current model output against the captured baseline."""
    baseline_file = regression_baseline_dir / f"v2_snapshot_{BASELINE_VERSION}.json"

    if not baseline_file.exists():
        pytest.skip(f"Baseline file not found: {baseline_file}")

    with open(baseline_file) as f:
        baseline = json.load(f)

    # Use the last available trade date from the fixture for the test.
    trade_date = str(regression_df_exec.index[-1].date())
    current_prices = _build_current_prices_from_df_exec(
        regression_df_exec, trade_date
    )
    snapshot = _capture_v2_snapshot(
        regression_df_exec,
        trade_date,
        regression_baseline_dir,
        current_prices=current_prices,
    )

    np.testing.assert_allclose(
        snapshot["w_final"], baseline["w_final"], atol=1e-12,
        err_msg="w_final mismatch against baseline",
    )
    np.testing.assert_allclose(
        snapshot["scores"], baseline["scores"], atol=1e-12,
        err_msg="scores mismatch against baseline",
    )
    assert snapshot["pit_binning"]["assigned_bin"] == baseline["pit_binning"]["assigned_bin"]
    assert snapshot["pit_binning"]["multiplier"] == pytest.approx(
        baseline["pit_binning"]["multiplier"], abs=1e-12
    )
```

#### 3.2.4 `scripts/capture_v2_baseline.py`

```python
#!/usr/bin/env python
"""One-off script to capture the current V2 baseline for regression tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.config import load_config_from_yaml
from leadlag.models.production_v2 import ProductionV2Model


def _build_current_prices_from_df_exec(
    df_exec: pd.DataFrame,
    trade_date: str,
) -> dict[str, float] | None:
    """Build 09:10 current prices dict from ``jp_open_trade_*`` columns.

    Falls back to ``jp_close_{ticker} * (1 + jp_gap_{ticker})`` when the
    open column is missing.
    """
    if trade_date not in df_exec.index:
        return None
    row = df_exec.loc[trade_date]
    prices = {}
    for t in JP_TICKERS:
        open_col = f"jp_open_trade_{t}"
        gap_col = f"jp_gap_{t}"
        close_col = f"jp_close_{t}"
        if open_col in row.index and not pd.isna(row[open_col]):
            prices[t] = float(row[open_col])
        elif gap_col in row.index and not pd.isna(row[gap_col]) \
                and close_col in row.index and not pd.isna(row[close_col]):
            prices[t] = float(row[close_col]) * (1.0 + float(row[gap_col]))
    if len(prices) == len(JP_TICKERS):
        return prices
    return None


def main() -> int:
    app_config = load_config_from_yaml("configs/production/production.yaml")
    model = ProductionV2Model(app_config.v2)

    # Load a production-like df_exec fixture or the local cache.
    from leadlag.data.cache import load_df_exec_from_local_cache
    from leadlag.data.fetcher import download_data
    from leadlag.data.preprocessor import preprocess_data

    df_exec = load_df_exec_from_local_cache()
    if df_exec is None or df_exec.empty:
        raw_data = download_data(beta_window=60)
        df_exec = preprocess_data(raw_data, beta_window=60)
    if df_exec is None or df_exec.empty:
        raise RuntimeError("No df_exec available for baseline capture")

    trade_date = df_exec.index[-1].strftime("%Y-%m-%d")
    gap_input_dir = Path("var/live/pipeline_data/gap_adjusted_distribution/latest")

    current_prices = _build_current_prices_from_df_exec(df_exec, trade_date)
    result = model.decide(
        trade_date=trade_date,
        gap_input_dir=gap_input_dir,
        df_exec=df_exec,
        current_prices=current_prices,
    )

    baseline = {
        "w_final": result["w_final"].tolist(),
        "scores": result["scores"].tolist(),
        "pit_binning": result["pit_binning"],
        "summary": {k: v for k, v in result["summary"].items() if k not in (
            "trade_date", "version", "candidate"
        )},
    }

    out_dir = Path("tests/regression/baselines")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "v2_snapshot_v20260813.json", "w") as f:
        json.dump(baseline, f, indent=2, default=str)

    # Copy PIT history so regression tests can reproduce pit_binning exactly.
    pit_src = gap_input_dir / "full_history_diagnostics.csv"
    if pit_src.exists():
        import shutil
        shutil.copy(pit_src, out_dir / "full_history_diagnostics.csv")

    print(f"Baseline captured for {trade_date}: {out_dir / 'v2_snapshot_v20260813.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### 3.3 取得する基準値

以下を `reports/refactoring_roadmap/baseline_metrics_20260813.json` として保存する。

```bash
python3 -m leadlag.cli backtest \
    --config configs/production/production.yaml \
    --start-date 2015-01-05 \
    --gap-dir var/live/pipeline_data/gap_adjusted_distribution/latest
```

`src/research/scripts/backtest/run_production_backtest.py` は deprecated なので使用しない。

```bash
python3 -m leadlag.cli backtest --start-date 2015-01-05
```

保存項目:

- `net_sharpe`
- `total_return`
- `max_drawdown`
- `turnover_mean`
- `fallback_rate`
- `weights` (日次全件 CSV)
- `daily_returns_net` (日次系列)
- `equity_curve` (日次系列)

### 3.4 完了基準

- `tests/regression/` ディレクトリが存在
- `scripts/capture_v2_baseline.py` が実行可能
- `reports/refactoring_roadmap/baseline_metrics_20260813.json` が作成されている
- `ProductionV2Model.compute_distribution(..., use_file_cache=True)` と `(..., use_file_cache=False)` の `w_final` 差分が < 1e-12（`tests/regression/test_v2_baseline.py` 内で確認）
- 現行 `bash scripts/run_tests_parallel.sh` が pass

### 3.5 フェーズ別テスト実行サブセット

全テストは最終ゲートで1回だけ実行する。開発中は各フェーズで対象サブセットを絞る。

| フェーズ | 対象テスト | 目的 |
|---|---|---|
| Phase 0 | `tests/regression/` | ベースライン取得 |
| Phase 1 | `tests/research/`, `tests/integration/test_production_residual_blpx.py` | 研究コード移設確認 |
| Phase 2 | `tests/unit/test_gap_*.py`, `tests/integration/test_production_v2.py` | gap on-demand 化確認 |
| Phase 3 | `tests/unit/test_config_*.py`, `tests/unit/test_config_frozen.py` | 設定変更確認 |
| Phase 4 | `tests/unit/test_close_positions.py`, `tests/unit/test_execution_submodules.py` | runner/CLI 確認 |
| Phase 5 | `tests/unit/test_preprocessor.py`, `tests/unit/test_pipeline.py` | データレイヤー確認 |

---

## 4. Phase 1: 研究コードの本番切り離し

### 4.1 背景

`tools/research/compute_gap_adjusted_distribution.py` は以下を import している。

```python
from research.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
)
```

これを解消する。

### 4.2 変更前後のファイル構成

#### 変更前

```
src/leadlag/models/
├── __init__.py
├── production_v2.py
├── signal_enhancement.py
└── ml_order_overlay.py

src/research/models/
├── __init__.py
├── base.py
├── blp_base.py
└── sector_relative_ensemble_blp_enhanced.py
```

#### 変更後

```
src/leadlag/models/
├── __init__.py
├── blp_base.py          # 新設（_BLPBase 移設 + BaseModel 互換メソッド）
├── blpx.py              # 新設（旧 SectorRelativeEnsembleBLPEnhancedModel）
├── production_v2.py
├── signal_enhancement.py
└── ml_order_overlay.py

src/research/models/
├── __init__.py
├── base.py              # 互換のため当面残す（最終的に削除）
├── blp_base.py          # 削除または archive へ
└── sector_relative_ensemble_blp_enhanced.py  # 削除または archive へ
```

### 4.3 具体的なファイル操作

#### 4.3.1 新規作成: `src/leadlag/models/blp_base.py`

`src/research/models/blp_base.py` と `src/research/models/base.py` の `BaseModel` を統合し、`src/leadlag/models/blp_base.py` として新設する。`_BLPBase` は `BLPModelBase` を継承し、`_resolve_val`, `_resolve_nested`, `_resolve_slippage_bps`, `normalize_signals`, `build_weights`, `get_audit_context` 等を提供する。

```python
"""Shared base class for BLPX production models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from leadlag.compliance.auditor import AuditContext
from leadlag.core import signal as signals


class BLPModelBase:
    """Minimal base for config-driven production models."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def _resolve_val(self, key: str, default: Any) -> Any:
        ...

    def _resolve_nested(self, key: str, default: Any) -> Any:
        ...


class _BLPBase(BLPModelBase):
    """Internal base for BLPX models."""

    _config_sections: list[str] = ["model", "ensemble", "portfolio", "costs", "residualization", "blpx"]
    _config_aliases: dict[str, list[str]] = {}

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # Resolve core parameters used by build_common_inputs and compute_blp_signal.
        self.ewma_halflife = int(self._resolve_val("ewma_halflife", 120))
        self.beta_window = int(self._resolve_val("beta_window", 60))
        self.include_v4_prior = bool(self._resolve_val("include_v4_prior", False))
        self.us_res_enabled = bool(self._resolve_val("us_res_enabled", False))
        self.us_res_gamma = float(self._resolve_val("us_res_gamma", 0.5))
        self.us_res_beta_window = int(self._resolve_val("us_res_beta_window", 252))
        self.frac_diff_enabled = bool(self._resolve_val("frac_diff_enabled", False))
        self.frac_diff_d = float(self._resolve_val("frac_diff_d", 0.1))
        self.frac_diff_threshold = float(self._resolve_val("frac_diff_threshold", 1e-5))
        self.frac_diff_window = int(self._resolve_val("frac_diff_window", 100))
        self.frac_diff_normalize = self._resolve_val("frac_diff_normalize", None)

    def _resolve_slippage_bps(self) -> float:
        ...

    def normalize_signals(self, sig: np.ndarray, method: str = "zscore") -> np.ndarray:
        ...

    def build_weights(
        self, signal: np.ndarray, q: float | None = None, Sigma_YY: np.ndarray | None = None,
    ) -> np.ndarray:
        ...

    def get_audit_context(self) -> AuditContext:
        ...

    def clear_caches(self) -> None:
        """Clear per-instance signal and correlation caches before pickling."""
        for attr in ("_production_signal_cache", "_residual_signal_cache",
                     "_raw_pca_cache", "_residual_pca_cache",
                     "_blp_corr_cache", "_common_inputs_cache"):
            if hasattr(self, attr):
                getattr(self, attr).clear()

    def _prepare_common_inputs(
        self,
        df_exec: pd.DataFrame,
        horizon: int = 1,
    ) -> CommonInputs:
        """Build and cache CommonInputs for the given df_exec and horizon."""
        from leadlag.data.preprocessor import compute_jp_target_returns
        from leadlag.core.pipeline import build_common_inputs
        from leadlag.data.tickers import JP_TICKERS, US_TICKERS

        # Cache key: df_exec id (e.g. its first/last index) + horizon.
        cache_key = (id(df_exec), horizon)
        if not hasattr(self, "_common_inputs_cache"):
            self._common_inputs_cache: dict = {}
        if cache_key in self._common_inputs_cache:
            return self._common_inputs_cache[cache_key]

        n_u = len(US_TICKERS)
        n_j = len(JP_TICKERS)
        y_jp_target = compute_jp_target_returns(df_exec, JP_TICKERS, horizon=horizon)

        inputs = build_common_inputs(
            df_exec,
            y_jp_target,
            n_u=n_u,
            n_j=n_j,
            ewma_half_life=self.ewma_halflife,
            beta_window=self.beta_window,
            include_v4_prior=self.include_v4_prior,
            us_res_enabled=getattr(self, "us_res_enabled", False),
            us_res_gamma=getattr(self, "us_res_gamma", 0.5),
            us_res_beta_window=getattr(self, "us_res_beta_window", 252),
            frac_diff_enabled=self.frac_diff_enabled,
            frac_diff_d=self.frac_diff_d,
            frac_diff_threshold=self.frac_diff_threshold,
            frac_diff_window=self.frac_diff_window,
            frac_diff_normalize=self.frac_diff_normalize,
        )
        self._common_inputs_cache[cache_key] = inputs
        return inputs

    # _build_blp_diagnostics 内で ``return_matrices=True`` の場合は
    # ``z_U_t``（``z_U`` ではない）を含む ``Sigma_XX`` / ``Sigma_YX`` /
    # ``Sigma_YY`` / ``B_struct`` / ``z_U_t`` / ``mu_Y`` / ``sigma_Y`` /
    # ``sigma_Y_denorm`` 等を返す。
    # compute_production_signal 等も research から移設
```

注：`ProductionV2Model` はこのクラスを継承しない。`ProductionV2Model` は `generate_v2_production_portfolio` をラップした独立したファサードである。

実装時の注意：`src/research/models/base.py` から `_resolve_val` / `_resolve_nested` を移設する際、`self.config` を `self.cfg`（`BLPModelBase.__init__` で設定）に読み替える。`ProductionBLPXModel` には `ProductionV2RunConfig.model_dump()`（`blpx` / `costs` / `portfolio` 等のセクションを含む dict）を渡す。

#### 4.3.2 新規作成: `src/leadlag/models/blpx.py`

`src/research/models/sector_relative_ensemble_blp_enhanced.py` をコピーし、以下を修正。

- ファイル名: `blpx.py`
- クラス名: `ProductionBLPXModel`
- import パス:

```python
from leadlag.models.blp_base import BLPModelBase, _BLPBase
```

- `_BLPBase` は `BLPModelBase` を継承するように `src/leadlag/models/blp_base.py` 内で統合
- 他の `research` import を `leadlag` 内に置き換え

```python
class ProductionBLPXModel(_BLPBase):
    """Production BLPX model (migrated from research package)."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # Resolve BLPX-specific parameters used by compute_blp_signal.
        self.rho = float(self._resolve_val("rho", 0.01))
        self.alpha_xx = float(self._resolve_val("alpha_xx", 0.20))
        self.alpha_yx = float(self._resolve_val("alpha_yx", 0.15))
        self.alpha_yy = float(self._resolve_val("alpha_yy", 0.50))
        self.lambda_pca = float(self._resolve_val("lambda_pca", 0.10))
        self.lambda_sector = float(self._resolve_val("lambda_sector", 0.60))
        self.beta_conf = float(self._resolve_val("beta_conf", 0.25))
        self.winsor_sigma = float(self._resolve_val("winsor_sigma", 3.0))
        self.blp_window = int(self._resolve_val("blp_window", 504))
        self.sector_eta = float(self._resolve_val("sector_eta", 0.5))
        self.sector_gamma = float(self._resolve_val("sector_gamma", 4.0))
        self.target = str(self._resolve_val("target", "topix_residual"))
        self.use_raw_target = bool(self._resolve_val("use_raw_target", False))
        self.asymmetry_mode = str(self._resolve_val("asymmetry_mode", "scalar"))
        self.asymmetry_delta = float(self._resolve_val("asymmetry_delta", 0.0))
        self.gap_open_coef = float(self._resolve_val("gap_open_coef", 0.70))
        self.topix_beta_coef = float(self._resolve_val("topix_beta_coef", 0.60))
        neg = self._resolve_val("gap_open_coef_neg", None)
        self.gap_open_coef_neg = float(neg) if neg is not None and str(neg).lower() != "none" else None
        neg = self._resolve_val("topix_beta_coef_neg", None)
        self.topix_beta_coef_neg = float(neg) if neg is not None and str(neg).lower() != "none" else None
        self.vol_adjusted_target = bool(self._resolve_val("vol_adjusted_target", False))
        self.macro_confidence_enabled = bool(self._resolve_val("macro_confidence_enabled", True))
        self.macro_kappa_enabled = bool(self._resolve_val("macro_kappa_enabled", True))
        self.macro_direction_enabled = bool(self._resolve_val("macro_direction_enabled", True))
        self.macro_kappas = tuple(self._resolve_val("macro_kappas", (3.0, 0.5, 0.5)))
        self.macro_surprise_halflife_mean = float(self._resolve_val("macro_surprise_halflife_mean", 20.0))
        self.macro_surprise_halflife_vol = float(self._resolve_val("macro_surprise_halflife_vol", 60.0))
        self.copula_enabled = bool(self._resolve_val("copula_enabled", True))
        self.copula_blend_weight = float(self._resolve_val("copula_blend_weight", 1.0))
        self.copula_dynamic_blend = bool(self._resolve_val("copula_dynamic_blend", True))
        self.copula_stress_threshold = float(self._resolve_val("copula_stress_threshold", 1.5))
        self.copula_nu_init = float(self._resolve_val("copula_nu_init", 5.0))
        self.asymmetry_post_gap_delta = float(self._resolve_val("asymmetry_post_gap_delta", 0.0))
        self.asymmetry_post_gap_mode = str(self._resolve_val("asymmetry_post_gap_mode", "signal_split"))
        self.frobenius_scale_priors = bool(self._resolve_val("frobenius_scale_priors", False))

    def compute_blp_signal(
        self,
        all_returns: np.ndarray,
        current_index: int,
        gap_override: np.ndarray | None = None,
        betas_t: np.ndarray | None = None,
        topix_night_t: float | None = None,
        v0_static: np.ndarray | None = None,
        c_full: np.ndarray | None = None,
        is_residual: bool = True,
        return_matrices: bool = False,
    ) -> dict:
        """Compute BLPX signal and optional diagnostic matrices.

        Returns at minimum ``signal`` (the gap-adjusted JP forecast). When
        ``return_matrices=True``, also returns the blocks needed to
        reconstruct ``Omega_struct``: ``z_hat_j_t1``, ``sigma_Y``,
        ``sigma_Y_denorm``, ``mu_Y``, ``Sigma_XX``, ``Sigma_YX``,
        ``Sigma_YY``, ``B_struct`` and ``z_U_t``.
        """
        ...

    # compute_production_signal, predict_signals, _compute_pca_prior 等を research から移設
```

#### 4.3.3 更新: `src/leadlag/models/__init__.py`

```python
from leadlag.models.blpx import ProductionBLPXModel
from leadlag.models.ml_order_overlay import (
    MLOrderOverlayModel,
    generate_v2_production_portfolio_with_overlay,
)
from leadlag.models.production_v2 import (
    ProductionV2Model,
    generate_v2_production_portfolio,
    generate_v2_production_portfolio_from_distribution,
)

__all__ = [
    "ProductionBLPXModel",
    "MLOrderOverlayModel",
    "ProductionV2Model",
    "generate_v2_production_portfolio",
    "generate_v2_production_portfolio_from_distribution",
    "generate_v2_production_portfolio_with_overlay",
]
```

#### 4.3.4 更新: `src/research/models/__init__.py`（移行期間限定）

以下の shims を追加して研究スクリプトの後方互換を維持（当面のみ）。

注意：`research` から `leadlag` への import は移行期間中の一時的措置。`import-linter` の契約に `research → leadlag` 方向の例外を明示し、移行完了後は削除する。循環 import（`tools/research` → `leadlag` → `research` → `leadlag`）を避けるため、shim クラスは `ProductionBLPXModel` を直接ラップするのみとする。

```python
"""Research package compatibility re-exports.

These re-exports are transitional and will be removed once all research
scripts have migrated to leadlag.models.blpx.
"""

from leadlag.models.blpx import ProductionBLPXModel as SectorRelativeEnsembleBLPEnhancedModel
from leadlag.models.blp_base import _BLPBase

__all__ = [
    "SectorRelativeEnsembleBLPEnhancedModel",
    "_BLPBase",
]
```

#### 4.3.5 更新: `tools/research/compute_gap_adjusted_distribution.py`

```python
# 変更前
from research.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
)

# 変更後
from leadlag.models.blpx import ProductionBLPXModel
```

使用箇所も修正:

```python
# 変更前
model = SectorRelativeEnsembleBLPEnhancedModel(cfg)

# 変更後
model = ProductionBLPXModel(cfg)
```

#### 4.3.6 更新: `src/research/scripts/backtest/run_production_backtest.py`

同様に import パスを `leadlag.models.blpx` に変更。

### 4.4 変更後の import 禁止ルール

`import-linter` に以下を追加する。

```toml
[[tool.importlinter.contracts]]
name = "tools and production must not depend on research"
type = "forbidden"
source_modules = [
    "leadlag.cli",
    "leadlag.execution",
    "leadlag.models",
    "leadlag.pipeline",
    "tools",
]
forbidden_modules = ["research"]

# During the transition, `research` re-exports `leadlag.models.blpx`.
# This exception must be removed once all research scripts are migrated.
[[tool.importlinter.contracts]]
name = "research may re-export leadlag during transition"
type = "forbidden"
source_modules = ["research"]
forbidden_modules = ["leadlag"]
ignore_imports = [
    "research.models.__init__ -> leadlag.models.blpx",
    "research.models.__init__ -> leadlag.models.blp_base",
]
```

### 4.5 完了基準

- `python3 -m compileall src/leadlag tools/research src/research` が成功
- `python3 scripts/experiments/verify_blpx_import.py` が成功（import 確認専用スクリプト新設）
- `bash scripts/run_tests_parallel.sh` が pass
- `python -m importlinter --verbose` が `research` 依存違反を検出しない

### 4.6 リスク

中。`blp_base.py` 内で `leadlag.data` 以外の `leadlag` 下位レイヤーを参照していないか確認。

---

## 5. Phase 2: gap 行列 on-demand 化

### 5.1 目的

`mu_gap` / `omega_gap` をファイル読み込みではなく、**`df_exec` + 9:10 gap 価格から純粋関数として計算**する。

### 5.2 前提：Step 1 と Step 2 の分離を維持

- **Step 1** (`tools/research/compute_structured_prediction_covariance.py`): 米国終値後に計算可能。出力は `omega_struct_YYYYMMDD.npy`。
- **Step 2** (gap 補正): 日本 9:10 価格取得後に計算可能。

本計画では **Step 1 は依然としてファイルで受け渡し**、Step 2（gap 補正）を on-demand 化する。

- `ProductionV2Model.compute_distribution` は、まず Step 2 ファイル `mu_gap_YYYYMMDD.npy` / `omega_gap_YYYYMMDD.npy` を `gap_input_dir` から読む。
- Step 2 ファイルが存在しない場合のみ、BLPX シグナル計算を on-demand に実行し、そこから `Omega_struct` を再構成して gap 補正を行う。
- Step 1 ファイル `omega_struct_YYYYMMDD.npy` は on-demand 経路では使用しない。`compute_blp_signal(return_matrices=True)` の返り値から `Omega_struct` を計算する。

#### 用語定義

- `Omega_struct`: 標準化 JP 予測リターンの相関構造行列（分散 1）。`compute_blp_signal` から直接返らず、共分散ブロックから計算する。
- `Omega_raw`: デノーマライズされた予測共分散行列 `D @ Omega_struct @ D`。
- `mu_raw`: 9:10 gap 適用前の予測平均リターン。`vol_adjusted_target` によって `z * sigma` または `mu_Y + sigma_Y * z` の形を取る。
- `mu_gap` / `Omega_gap`: 9:10 gap 情報を `mu_raw` / `Omega_raw` に適用した後の予測分布。
- `gap_open_coef` (c) / `topix_beta_coef` (b): gap 補正の係数。US マーケットが負の場合、`gap_open_coef_neg` / `topix_beta_coef_neg` に切り替える。

### 5.3 新規作成: `src/leadlag/core/gap_adjustment.py`

以下の純粋関数を提供する。

```python
"""Pure gap adjustment functions for V2 production."""

from __future__ import annotations

import numpy as np


def compute_filtered_gap(
    gap_override: np.ndarray,
    betas_t: np.ndarray,
    topix_night_t: float,
    gap_open_coef: float,
    topix_beta_coef: float,
    denominator_floor: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute filtered gap and denominator.

    Args:
        gap_override: raw JP opening gap returns (n_j,)
        betas_t: per-ticker TOPIX betas (n_j,)
        topix_night_t: TOPIX overnight return (scalar)
        gap_open_coef: idiosyncratic gap coefficient (c). Callers must select
            ``gap_open_coef_neg`` when the US market is negative if asymmetric
            gap correction is configured.
        topix_beta_coef: TOPIX systematic coefficient (b). Same neg override
            applies.
        denominator_floor: floor applied to 1 + gap_filt

    Returns:
        (gap_filt, denominator, denominator_floored)
    """
    gap_syst = betas_t * topix_night_t
    gap_idio = gap_override - gap_syst
    gap_filt = gap_open_coef * gap_idio + (gap_open_coef - topix_beta_coef) * gap_syst
    denominator = 1.0 + gap_filt
    denominator_floored = np.maximum(denominator, denominator_floor)
    return gap_filt, denominator, denominator_floored


def compute_gap_adjusted_distribution(
    mu_raw: np.ndarray,
    omega_raw: np.ndarray,
    gap_override: np.ndarray,
    betas_t: np.ndarray,
    topix_night_t: float,
    gap_open_coef: float = 0.70,
    topix_beta_coef: float = 0.60,
    denominator_floor: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return gap-adjusted predictive distribution (mu_gap, Omega_gap).

    Pure function: the inputs ``mu_raw`` and ``omega_raw`` must already be
    reconstructed from the BLPX signal at the signal date. The 9:10 inputs
    ``gap_override``, ``betas_t``, ``topix_night_t`` become available on
    trade_date.

    ``gap_open_coef`` and ``topix_beta_coef`` must reflect the US-direction
    sensitive selection used inside the BLPX model (``gap_open_coef_neg``
    and ``topix_beta_coef_neg`` when the US market is negative and the
    asymmetric configuration is set).
    """
    _, _, denom_floored = compute_filtered_gap(
        gap_override, betas_t, topix_night_t,
        gap_open_coef, topix_beta_coef, denominator_floor,
    )
    d = 1.0 / denom_floored
    D = np.diag(d)
    mu_gap = (1.0 + mu_raw) * d - 1.0
    omega_gap = D @ omega_raw @ D
    omega_gap = 0.5 * (omega_gap + omega_gap.T)
    return mu_gap, omega_gap
```

### 5.3.1 `build_raw_distribution()` も同じモジュールに配置

`ProductionBLPXModel.compute_blp_signal(..., return_matrices=True)` は標準化 BLPX 出力に加え、以下の診断行列を返す。

- `z_hat_j_t1`: クロスセクショナル z-score
- `sigma_Y`: 標準化 JP リターン列の標準偏差ベクトル
- `sigma_Y_denorm`: 予測標準偏差（デノーマライズ）ベクトル
- `mu_Y`: 予測平均ベクトル
- `Sigma_XX`, `Sigma_YX`, `Sigma_YY`: BLP 係数推定に使用された共分散ブロック
- `B_struct`: 構造化 BLP 係数行列
- `z_U_t`: 当日 US リターン（US 方向判定用）

`Omega_struct` は `compute_blp_signal` から直接返らない。返却された共分散ブロックから計算する。

```python
def _omega_from_blp_res(blpx_result: dict) -> np.ndarray:
    """Compute standardized ``Omega_struct`` from BLPX matrix outputs."""
    Sigma_XX = blpx_result["Sigma_XX"]
    Sigma_YX = blpx_result["Sigma_YX"]
    Sigma_YY = blpx_result["Sigma_YY"]
    B_struct = blpx_result["B_struct"]
    Sigma_XY = Sigma_YX.T

    Omega_struct = (
        Sigma_YY
        - B_struct @ Sigma_XY
        - Sigma_YX @ B_struct.T
        + B_struct @ Sigma_XX @ B_struct.T
    )
    Omega_struct = 0.5 * (Omega_struct + Omega_struct.T)
    return Omega_struct


def build_raw_distribution(
    blpx_result: dict,
    vol_adjusted_target: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct (mu_raw, Omega_raw) from BLPX signal output.

    ``vol_adjusted_target`` changes the de-normalization of ``z_hat_j_t1``:
    - True: ``mu_raw = z_hat_j_t1 * sigma_Y_denorm``
    - False: ``mu_raw = mu_Y + sigma_Y * z_hat_j_t1``
    """
    z = blpx_result["z_hat_j_t1"]
    sigma = blpx_result["sigma_Y_denorm"]
    mu_y = blpx_result["mu_Y"]
    sigma_Y = blpx_result.get("sigma_Y", sigma)

    if vol_adjusted_target:
        mu_raw = z * sigma
    else:
        mu_raw = mu_y + sigma_Y * z

    corr = _omega_from_blp_res(blpx_result)
    D = np.diag(sigma)
    omega_raw = D @ corr @ D
    omega_raw = 0.5 * (omega_raw + omega_raw.T)
    return mu_raw, omega_raw
```

注：`_omega_from_blp_res` も `src/leadlag/core/gap_adjustment.py` に配置する。上記は `tools/research/compute_gap_adjusted_distribution.py::_omega_from_blp_res` と同一式。

### 5.4 変更: `src/leadlag/models/production_v2.py`

#### 5.4.1 `ProductionV2Model.__init__` の変更

```python
class ProductionV2Model:
    def __init__(
        self,
        config: ProductionV2RunConfig,
        *,
        blpx_model: ProductionBLPXModel,
        overlay_model: MLOrderOverlayModel | None = None,
    ) -> None:
        """Initialize the V2 production model.

        ``blpx_model`` is a required dependency. All call sites
        (``BacktestEngine._generate_v2_weights``, ``v2_bridge.py``,
        ``cli.py`` and tests) must be updated to construct it before
        passing it in.
        """
        self.run_config = config
        self._blpx_model = blpx_model
        self._overlay_model = overlay_model
```

#### 5.4.2 新メソッド `compute_distribution()`

```python
from leadlag.core.gap_adjustment import (
    build_raw_distribution,
    compute_gap_adjusted_distribution,
)


class ProductionV2Model:
    ...

    def compute_distribution(
        self,
        trade_date: str,
        df_exec: pd.DataFrame,
        current_prices: dict[str, float],
        *,
        horizon: int = 1,
        mu_pattern: str | None = None,
        omega_pattern: str | None = None,
        use_file_cache: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute (mu_gap, Omega_gap) for trade_date and horizon.

        1. The validated Step 2 file cache (``mu_gap_YYYYMMDD.npy`` and
           ``omega_gap_YYYYMMDD.npy``) is the primary, trusted path.
        2. If the cache is missing, fall back to on-demand computation
           from ``df_exec`` and ``current_prices``.
        3. If both are available, compare them in shadow mode and keep
           the file cache. An audit alert is emitted if they differ.
        """
        # 1. Try file cache first.
        file_mu = file_omega = None
        if use_file_cache and self.run_config.gap_input_dir is not None:
            if horizon == 1:
                _mu_pattern = mu_pattern or "matrices/mu_gap_{date}.npy"
                _omega_pattern = omega_pattern or "matrices/omega_gap_{date}.npy"
                _pattern_kwargs = None
            else:
                _mu_pattern = mu_pattern or self.run_config.mh_mu_file_pattern_h
                _omega_pattern = omega_pattern or self.run_config.mh_omega_file_pattern_h
                _pattern_kwargs = {"h": horizon}
            file_mu, file_omega, _ = load_gap_matrices(
                self.run_config.gap_input_dir,
                trade_date,
                mu_pattern=_mu_pattern,
                omega_pattern=_omega_pattern,
                pattern_kwargs=_pattern_kwargs,
                n_j=len(JP_TICKERS),
                strict=False,
            )

        # 2. Always compute on-demand so we can validate against cache.
        mu_ondemand, omega_ondemand = self._compute_ondemand(
            trade_date, df_exec, current_prices, horizon=horizon
        )

        # 3. Prefer file cache; emit audit alert if on-demand differs.
        if file_mu is not None and file_omega is not None:
            if not np.allclose(file_mu, mu_ondemand, atol=1e-12) or \
               not np.allclose(file_omega, omega_ondemand, atol=1e-12):
                logger.warning(
                    "On-demand result differs from file cache for %s; "
                    "using file cache and recording audit alert.", trade_date
                )
            return file_mu, file_omega

        return mu_ondemand, omega_ondemand

    def _compute_ondemand(
        self,
        trade_date: str,
        df_exec: pd.DataFrame,
        current_prices: dict[str, float],
        *,
        horizon: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute (mu_gap, Omega_gap) on-demand from df_exec and prices."""
        current_index = self._resolve_current_index(df_exec, trade_date)

        gap_override, betas_t, topix_night_t = self._extract_gap_inputs(
            df_exec, current_index, current_prices
        )

        # Build all inputs required by the BLPX model.
        inputs = self._extract_all_returns(df_exec, current_index, horizon=horizon)
        all_returns = inputs["all_returns"]
        v0_static = inputs["v0_static"]
        c_full = inputs["c_full"]

        blpx_result = self._blpx_model.compute_blp_signal(
            all_returns=all_returns,
            current_index=current_index,
            gap_override=gap_override,
            betas_t=betas_t,
            topix_night_t=topix_night_t,
            v0_static=v0_static,
            c_full=c_full,
            is_residual=True,
            return_matrices=True,
        )

        mu_raw, omega_raw = build_raw_distribution(
            blpx_result,
            vol_adjusted_target=self._blpx_model.vol_adjusted_target,
        )

        # Replicate the US-direction-sensitive coefficient selection used
        # inside ``_apply_gap_adjustment``.
        us_negative = float(np.nanmean(blpx_result["z_U_t"])) < 0.0
        if us_negative and self._blpx_model.gap_open_coef_neg is not None:
            gap_open_coef = self._blpx_model.gap_open_coef_neg
            topix_beta_coef = self._blpx_model.topix_beta_coef_neg
        else:
            gap_open_coef = self._blpx_model.gap_open_coef
            topix_beta_coef = self._blpx_model.topix_beta_coef

        return compute_gap_adjusted_distribution(
            mu_raw=mu_raw,
            omega_raw=omega_raw,
            gap_override=gap_override,
            betas_t=betas_t,
            topix_night_t=topix_night_t,
            gap_open_coef=gap_open_coef,
            topix_beta_coef=topix_beta_coef,
        )
```

#### 5.4.2a 補助メソッド `_extract_all_returns()`

`compute_distribution` 内で呼ばれる補助メソッド。`df_exec` から BLPX 計算に必要な入力を構築する。

```python
def _extract_all_returns(
    self,
    df_exec: pd.DataFrame,
    current_index: int,
    horizon: int = 1,
) -> dict[str, Any]:
    """Build BLPX inputs for the given trade date and horizon.

    Returns a dict with at least:
      - ``all_returns``: residualized returns matrix (``jp_res_returns_p3``)
      - ``v0_static``: static prior vectors
      - ``c_full``: baseline correlation matrix (``c_full_p3``)
    """
    inputs = self._blpx_model._prepare_common_inputs(df_exec, horizon=horizon)
    return {
        "all_returns": inputs["jp_res_returns_p3"],
        "v0_static": inputs["v0_static"],
        "c_full": inputs["c_full_p3"],
    }
```

注意：`_prepare_common_inputs` は `df_exec` 全体に対して `build_common_inputs` を呼ぶ。9:10 計算時間削減のため、`CommonInputs` をインスタンスレベルでキャッシュすることを検討する。

#### 5.4.2b 補助メソッド `_resolve_*` / `_extract_gap_inputs`

```python
def _resolve_sig_date(
    self,
    df_exec: pd.DataFrame,
    trade_date: str,
) -> str:
    """Return the US signal date corresponding to trade_date."""
    return str(df_exec.loc[trade_date, "sig_date"].date())


def _resolve_current_index(
    self,
    df_exec: pd.DataFrame,
    trade_date: str,
) -> int:
    """Return the integer position of trade_date in df_exec."""
    return int(df_exec.index.get_loc(trade_date))


def _extract_gap_inputs(
    self,
    df_exec: pd.DataFrame,
    current_index: int,
    current_prices: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (gap_override, betas_t, topix_night_t) for current_index."""
    trade_date = df_exec.index[current_index]
    row = df_exec.loc[trade_date]
    prev_close_cols = [f"jp_close_sig_{t}" for t in JP_TICKERS]
    prev_close = row[prev_close_cols].values.astype(float)

    gap_override = np.zeros(len(JP_TICKERS))
    for i, t in enumerate(JP_TICKERS):
        if t in current_prices and np.isfinite(prev_close[i]) and prev_close[i] > 0:
            gap_override[i] = current_prices[t] / prev_close[i] - 1.0

    beta_cols = [f"jp_beta_{t}" for t in JP_TICKERS]
    betas_t = row[beta_cols].values.astype(float)
    topix_night_t = float(row["topix_night_return"])
    return gap_override, betas_t, topix_night_t
```

#### 5.4.3 `decide()` のシグナチャ拡張

```python
def decide(
    self,
    trade_date: str,
    gap_input_dir: str | Path | None = None,
    df_exec: pd.DataFrame | None = None,
    current_prices: dict[str, float] | None = None,
    overlay_enabled: bool = True,
    use_file_cache: bool = True,
) -> dict:
    """Generate the final V2 decision.

    ``gap_input_dir`` provides the validated Step 2 file cache and is
    preferred. If the cache is missing, on-demand computation is attempted
    from ``df_exec`` and ``current_prices``.
    """
    if gap_input_dir is not None:
        self.run_config = self.run_config.model_copy(
            update={"gap_input_dir": Path(gap_input_dir)}
        )

    if df_exec is None or current_prices is None:
        raise ValueError("df_exec and current_prices are required for V2 decision")

    if self.run_config.mh_blend_enabled and len(self.run_config.mh_horizons) > 1:
        mu_gap, omega_gap, scores = self._multi_horizon_scores(
            trade_date=trade_date,
            df_exec=df_exec,
            current_prices=current_prices,
            use_file_cache=use_file_cache,
        )
    else:
        mu_gap, omega_gap = self.compute_distribution(
            trade_date=trade_date,
            df_exec=df_exec,
            current_prices=current_prices,
            horizon=1,
            use_file_cache=use_file_cache,
        )
        scores = None

    result = generate_v2_production_portfolio_from_distribution(
        mu_gap=mu_gap,
        omega_gap=omega_gap,
        trade_date=trade_date,
        run_config=self.run_config,
        df_exec=df_exec,
        scores=scores,
    )

    if overlay_enabled:
        result = self._apply_overlay(result, trade_date, df_exec)

    return result


def _multi_horizon_scores(
    self,
    trade_date: str,
    df_exec: pd.DataFrame,
    current_prices: dict[str, float],
    use_file_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-horizon (mu_gap, Omega_gap) and blend mu_over_sigma scores.

    Mirrors ``leadlag.models.signal_enhancement.apply_multi_horizon_blend``
    but works on-demand instead of loading pre-computed files.
    """
    from leadlag.data.tickers import JP_TICKERS
    from leadlag.models.signal_enhancement import cross_sectional_zscore

    n_j = len(JP_TICKERS)
    blended = np.zeros(n_j)
    total_weight = 0.0
    scores_h1 = None
    mu_h1 = None
    omega_h1 = None

    for h, w in zip(self.run_config.mh_horizons, self.run_config.mh_weights):
        mu_h, omega_h = self.compute_distribution(
            trade_date=trade_date,
            df_exec=df_exec,
            current_prices=current_prices,
            horizon=h,
            use_file_cache=use_file_cache,
        )
        if h == 1:
            mu_h1, omega_h1 = mu_h, omega_h
        sigma_h = np.sqrt(np.maximum(np.diag(omega_h), 1e-6))
        scores_h = mu_h / sigma_h
        z_h = cross_sectional_zscore(scores_h)
        blended += w * z_h
        total_weight += w

    if total_weight < 1e-8:
        blended = z_h if scores_h1 is None else cross_sectional_zscore(scores_h1)
    else:
        blended = blended / total_weight

    # Rescale to h=1 score magnitude and shift to h=1 median.
    if mu_h1 is not None:
        sigma_1 = np.sqrt(np.maximum(np.diag(omega_h1), 1e-6))
        scores_h1 = mu_h1 / sigma_1
        h1_std = np.std(scores_h1)
        blended_std = np.std(blended)
        if blended_std > 1e-8:
            blended = blended * (h1_std / blended_std)
        blended = blended + np.median(scores_h1)

    return mu_h1, omega_h1, blended
```

注意：`generate_v2_production_portfolio_from_distribution` は新設または `generate_v2_production_portfolio` から gap 行列 I/O 部分を切り出した関数。`w_final`, `scores`, `pit_binning`, `summary` を生成する。

#### 5.4.5 既存 `generate_v2_production_portfolio` の切り分け

`generate_v2_production_portfolio` から、gap 行列 I/O に依存しない部分を `generate_v2_production_portfolio_from_distribution` として切り出す。

```python
def generate_v2_production_portfolio_from_distribution(
    mu_gap: np.ndarray,
    omega_gap: np.ndarray,
    trade_date: str,
    run_config: ProductionV2RunConfig,
    df_exec: pd.DataFrame,
    scores: np.ndarray | None = None,
) -> dict:
    """Build V2 portfolio from a pre-computed gap-adjusted distribution.

    This function contains all ranking, RuleD, risk, and weight-building
    logic, but no gap matrix file I/O. ``scores`` may be pre-computed
    (e.g. multi-horizon blended); otherwise ``mu_gap / sigma_gap`` is used.
    """
    n_j = len(JP_TICKERS)
    if scores is None:
        sigma_gap = np.sqrt(np.maximum(np.diag(omega_gap), run_config.sigma_floor))
        scores = mu_gap / sigma_gap

    # Apply cross-sectional rank-reversal overlay if configured.
    # This still loads a pre-computed file and is out of scope for on-demand.
    if run_config.cs_overlay_enabled and run_config.gap_input_dir is not None:
        from leadlag.models.signal_enhancement import apply_rank_reversal_overlay
        scores, _ = apply_rank_reversal_overlay(
            scores=scores,
            gap_input_dir=run_config.gap_input_dir,
            date_str=trade_date,
            weight=run_config.cs_overlay_weight,
            file_pattern=run_config.cs_rank_reversal_file_pattern,
        )

    # Ranking, RuleD, risk, weight building, PIT binning
    ...
```

```python
def parse_run_config(cfg: dict) -> ProductionV2RunConfig:
    """Convert a raw (possibly flat) config dict into ProductionV2RunConfig."""
    from leadlag.config.schemas import _map_flat_to_nested
    mapped = _map_flat_to_nested(safe_config_copy(cfg))
    return ProductionV2RunConfig(**mapped)


def generate_v2_production_portfolio(
    trade_date: str,
    gap_input_dir: Path | None,
    cfg: ProductionV2RunConfig | dict,
    df_exec: pd.DataFrame | None,
    blpx_model: ProductionBLPXModel,
) -> dict:
    """Backward-compatible wrapper: file-driven V2 decision."""
    cfg = safe_config_copy(cfg)
    run_cfg = cfg if isinstance(cfg, ProductionV2RunConfig) else parse_run_config(cfg)
    v2_model = ProductionV2Model(run_cfg, blpx_model=blpx_model)
    return v2_model.decide(
        trade_date=trade_date,
        gap_input_dir=gap_input_dir,
        df_exec=df_exec,
        current_prices=_build_current_prices_from_df_exec(df_exec, trade_date) if df_exec is not None else None,
    )
```

`generate_v2_production_portfolio` は当面、ファイルから gap 行列を読み込んで `ProductionV2Model.decide` を呼ぶ thin wrapper として残す。移行期間後に削除。

#### 5.4.6 `Overlay` 対応

```python
class ProductionV2Model:
    ...

    def _apply_overlay(
        self,
        result: dict,
        trade_date: str,
        df_exec: pd.DataFrame,
    ) -> dict:
        """Apply the ML order overlay if enabled.

        ``apply_overlay`` operates on the full V2 result dict, not just
        ``w_final``. It returns the (possibly unchanged) result dict.
        """
        if self._overlay_model is None or not self.run_config.ml_overlay_enabled:
            return result
        from leadlag.models.ml_order_overlay import apply_overlay
        return apply_overlay(result, df_exec, self._overlay_model, trade_date)
```

`src/leadlag/models/ml_order_overlay.py` 内の `generate_v2_production_portfolio_with_overlay` も、移行期間中は `ProductionV2Model` を使う thin wrapper にする。

```python
def generate_v2_production_portfolio_with_overlay(
    trade_date: str,
    gap_input_dir: Path | None,
    cfg: ProductionV2RunConfig | dict,
    df_exec: pd.DataFrame | None,
    overlay_model: MLOrderOverlayModel | None,
    blpx_model: ProductionBLPXModel | None = None,
) -> dict:
    """Backward-compatible wrapper: build V2 decision and optionally apply overlay."""
    from leadlag.models.production_v2 import (
        ProductionV2Model, safe_config_copy,
    )
    cfg = safe_config_copy(cfg)
    run_cfg = cfg if isinstance(cfg, ProductionV2RunConfig) else parse_run_config(cfg)
    if blpx_model is None:
        blpx_model = ProductionBLPXModel(run_cfg.model_dump())
    v2_model = ProductionV2Model(
        run_cfg, blpx_model=blpx_model, overlay_model=overlay_model
    )
    result = v2_model.decide(
        trade_date=trade_date,
        gap_input_dir=gap_input_dir,
        df_exec=df_exec,
        current_prices=_build_current_prices_from_df_exec(df_exec, trade_date)
            if df_exec is not None else None,
    )
    return result
```

### 5.5 変更: `src/leadlag/execution/v2_bridge.py`

`run_v2_decision` 内で `ProductionRunner`（または `ProductionV2Model`）を使用する。`generate_v2_production_portfolio` 関数は `ProductionV2Model.decide` への移行期間中に thin wrapper 化する。

```python
from leadlag.runner.production import ProductionRunner, RunnerInputs

runner = ProductionRunner(app_config)
result = runner.run(
    RunnerInputs(
        trade_date=trade_date,
        df_exec=df_exec,
        gap_input_dir=gap_dir,
        current_prices=manual_opens,
        use_file_cache=True,
        previous_positions=previous_positions,
    )
)
```

### 5.6 変更: `src/leadlag/execution/backtester.py`

`run_v2_backtest` 内で `ProductionV2Model`（および `ProductionBLPXModel`）を構築し、`_generate_v2_weights` では `decide` を呼び出す。

```python
# Near the start of run_v2_backtest
blpx_model = ProductionBLPXModel(app_config.v2.model_dump())
overlay_model = None
if app_config.v2.ml_overlay_enabled:
    from leadlag.models.ml_order_overlay import load_overlay_model
    overlay_model = load_overlay_model(Path(app_config.v2.ml_overlay_model_dir))
v2_model = ProductionV2Model(
    app_config.v2,
    blpx_model=blpx_model,
    overlay_model=overlay_model,
)

# Clear any per-instance caches before pickling workers (only matters for n_jobs > 1).
blpx_model.clear_caches()

# Remove the old procedural branch that called
# generate_v2_production_portfolio_with_overlay directly; always use v2_model.decide.
```

`_generate_v2_weights` に `v2_model` を引数で渡し、内部で `df_exec` と 9:10 価格を渡す。

```python
def _generate_v2_weights(
    ...,
    v2_model: ProductionV2Model,
    app_config: AppConfig,
) -> dict:
    ...
    result = v2_model.decide(
        trade_date=date_str,
        gap_input_dir=effective_gap_dir,
        df_exec=df_exec,
        current_prices=_build_current_prices_from_df_exec(df_exec, date_str),
        overlay_enabled=app_config.v2.ml_overlay_enabled,
        use_file_cache=True,
    )
    ...
```

`_build_current_prices_from_df_exec` を `backtester.py` 内で import または定義する。価格は `jp_open_trade_*` 列から取得する。`BacktestEngine.run_v2_backtest` 内で `v2_model` を 1 回構築して `_generate_v2_weights` 経由で再利用する。

注意：`n_jobs > 1` で `Parallel` を使う場合、`v2_model`（特に `blpx_model` 内部のキャッシュ）を pickle でワーカーに渡す必要がある。並列化前に `blpx_model.clear_caches()` を呼び出し、`ProductionBLPXModel` に大きなキャッシュが残らないようにする。キャッシュクリア後も pickle サイズが大きい場合は `n_jobs=1` を推奨する。

### 5.7 GapStore / .npy の役割変更

`src/leadlag/utils/gap_matrix_io.py` は維持するが、役割を以下に明確化。

- **primary**: on-demand 計算
- **file cache**: 高速化・再現性確認
- **GapStore**: 監査証跡

`ProductionV2Model.compute_distribution` の動作:

1. ファイルキャッシュを読み込む（`use_file_cache=True` 時）。
2. 常に on-demand 計算も実行し、ファイルキャッシュと比較する（shadow / audit）。
3. 両方存在して一致しない場合、**ファイルキャッシュを採用**し、監査警告を発行。

Step 2 ファイルは事前計算・監査済みであるため、本番ではこれを信頼する。on-demand 結果との差分は shadow 運用で記録する。

### 5.8 変更内容が一意に定まるか

**定まらない点**: 

- `ProductionBLPXModel` の初期化をどこで行うか（`ProductionV2Model.__init__` か、runner 内か）

**推奨デフォルト**:

- `ProductionV2Model` は `ProductionBLPXModel` を **構築時に受け取る**（DI）。
- runner / backtester は `ProductionBLPXModel` を構築して `ProductionV2Model` に注入する。

### 5.9 完了基準

- `tests/unit/test_gap_adjustment.py` 新設、純粋関数のテスト pass
- `ProductionV2Model.decide(trade_date, df_exec=..., current_prices=...)` が `gap_input_dir=None` でも動作
- ファイル読み込み時と on-demand 計算時の `w_final` 差分 < 1e-12
- `compute_distribution` の on-demand 実行時間は実測により判断。目標は 1 日あたり 5 秒以下だが、`tools/validation/monitor_residual_blpx_shadow_performance.py` で 3 営業日以上計測し、9:10 カットオフ前に完了することを確認
- ファイルキャッシュ優先動作の確認
- 全テスト pass

### 5.10 リスク

高。Step 2 計算が9:10までに終わらない場合、本番 flat position 増加。Phase 2 完了前に `tools/validation/monitor_residual_blpx_shadow_performance.py` で実測し、計算時間が 30 秒を超える場合は on-demand 化を見送り、ファイル駆動のまま運用を継続する。

---

## 6. Phase 3: 設定スキーマの整理

### 6.1 現状の問題

`src/leadlag/config/schemas.py` 内の `StrategyConfig` は V1 用パラメータを多数含む。
`ProductionV2RunConfig` は `_flatten_nested_yaml` で入れ子 YAML を flat にしている。

### 6.2 変更後の `configs/production/production.yaml`

フラット化。入れ子は廃止。

```yaml
# 簡易版例（完全版は別途検討）
model_name: production_residual_blpx
version: production_residual_blpx_v2

ranking_mode: mu_over_sigma
sigma_floor: 1.0e-6

# RuleD
pit_rolling_window: 252
tertile_low_pct: 33.3333
tertile_high_pct: 66.6667
mult_low: 0.75
mult_mid: 1.00
mult_high: 1.00
baseline_gross: 2.0
fallback_multiplier: 1.00

# portfolio
long_count: 5
short_count: 5
minvar_enabled: true
minvar_alpha: 0.8

# costs
slippage_bps: 5.0
cost_bps_per_gross: 10.0
overnight_alpha_long: 0.75
overnight_alpha_short: 0.5
buy_interest_annual: 0.025
borrow_fee_annual: 0.0115
reverse_fee_bps: 2.0

# BLPX
blpx_rho: 0.01
blpx_alpha_xx: 0.20
blpx_alpha_yx: 0.15
blpx_alpha_yy: 0.50
blpx_lambda_pca: 0.10
blpx_lambda_sector: 0.60
blpx_beta_conf: 0.25
blpx_winsor_sigma: 3.0
blpx_blp_window: 504
blpx_ewma_halflife: 120
blpx_sector_eta: 0.5
blpx_sector_gamma: 4.0
blpx_target: topix_residual
blpx_use_raw_target: false
blpx_asymmetry_mode: scalar
blpx_asymmetry_delta: 0.30
blpx_gap_open_coef: 0.70
blpx_topix_beta_coef: 0.60
blpx_gap_open_coef_neg: 0.60
blpx_topix_beta_coef_neg: 0.60
blpx_vol_adjusted_target: false
blpx_execution_target_cost_adjustment: none
blpx_asymmetry_post_gap_delta: 0.0
blpx_asymmetry_post_gap_mode: signal_split
blpx_frobenius_scale_priors: false

# residualization
residualization_enabled_for_p3: true
residualization_beta_window: 60
residualization_beta_winsor_sigma: 3.0
residualization_beta_shrinkage: 0.05

# macro
macro_kappa_enabled: true
macro_kappas: [3.0, 0.5, 0.5]
macro_surprise_halflife_mean: 20.0
macro_surprise_halflife_vol: 60.0

# multi-horizon
mh_blend_enabled: true
mh_horizons: [1, 3, 5]
mh_weights: [0.8, 0.1, 0.1]

# cs overlay
cs_overlay_enabled: true
cs_overlay_weight: 0.05

# ml order overlay
ml_overlay_enabled: true
ml_overlay_model_dir: models/ml_order_overlay/phase2_8
ml_overlay_fallback_to_baseline: true

# execution
execution_side_leverage: 1.5

# output
output_base_dir: results/production_residual_blpx
output_live_dir: live/production_residual_blpx
```

### 6.3 新規作成: 設定移行スクリプト

`scripts/migrate_production_config.py`:

```python
#!/usr/bin/env python
"""Migrate nested production.yaml to flat schema."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def flatten(src: dict) -> dict:
    """Flatten nested YAML into a dict suitable for ProductionV2RunConfig."""
    flat = {}
    blpx = {}
    costs = {}

    # Direct top-level keys that are already flat (and not in a sub-model)
    COST_FIELDS = {
        "slippage_bps", "cost_bps_per_gross", "overnight_alpha_long",
        "overnight_alpha_short", "buy_interest_annual", "borrow_fee_annual",
        "reverse_fee_bps", "side_leverage",
    }
    flat.update({k: v for k, v in src.items() if not isinstance(v, dict) and k not in COST_FIELDS})

    # blpx section -> blpx_* prefixed flat keys
    for k, v in src.get("blpx", {}).items():
        flat[f"blpx_{k}"] = v

    # costs section -> flat, will be re-nested by load_config_from_yaml
    for k, v in src.get("costs", {}).items():
        costs[k] = v

    # Other nested sections map directly without prefix, matching
    # ProductionV2RunConfig field names.
    for section in [
        "portfolio", "gross_scaling", "fallback",
        "residualization", "features",
    ]:
        if isinstance(src.get(section), dict):
            for k, v in src[section].items():
                if isinstance(v, dict):
                    # e.g. gross_scaling.multipliers -> mult_low etc.
                    for kk, vv in v.items():
                        flat[kk] = vv
                else:
                    flat[k] = v

    # Explicit residualization mapping.
    res = src.get("residualization", {})
    if isinstance(res, dict):
        for k, v in res.items():
            if k == "enabled_for_p3":
                flat["residualization_enabled_for_p3"] = v
            elif k == "beta_window":
                flat["residualization_beta_window"] = v
            elif k == "winsor_sigma":
                flat["residualization_beta_winsor_sigma"] = v
            elif k == "shrinkage":
                flat["residualization_beta_shrinkage"] = v

    # Explicit multi-horizon blend mapping.
    mh = src.get("multi_horizon_blend", {})
    if isinstance(mh, dict):
        for k, v in mh.items():
            if k == "enabled":
                flat["mh_blend_enabled"] = v
            elif k == "horizons":
                flat["mh_horizons"] = v
            elif k == "weights":
                flat["mh_weights"] = v
            elif k == "mu_file_pattern_h":
                flat["mh_mu_file_pattern_h"] = v
            elif k == "omega_file_pattern_h":
                flat["mh_omega_file_pattern_h"] = v

    # Explicit cross-sectional overlay mapping (keys differ from section name).
    cs = src.get("cs_feature_overlay", {})
    if isinstance(cs, dict):
        for k, v in cs.items():
            if k == "enabled":
                flat["cs_overlay_enabled"] = v
            elif k == "weight":
                flat["cs_overlay_weight"] = v
            elif k == "rank_reversal_file_pattern":
                flat["cs_rank_reversal_file_pattern"] = v

    flat["costs"] = costs
    return flat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    with open(args.input) as f:
        src = yaml.safe_load(f)

    flat = flatten(src)

    with open(args.output, "w") as f:
        yaml.safe_dump(flat, f, sort_keys=False)

    print(f"Migrated {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
```

### 6.4 `ProductionV2RunConfig` の変更

`_flatten_nested_yaml` を廃止。直接 flat YAML を読み込む。

```python
class CostConfig(BaseModel):
    """Cost and financing parameters shared by production and backtest."""

    slippage_bps: float = Field(default=5.0, ge=0.0)
    cost_bps_per_gross: float = Field(default=10.0, ge=0.0)
    overnight_alpha_long: float = Field(default=0.75, ge=0.0, le=1.0)
    overnight_alpha_short: float = Field(default=0.5, ge=0.0, le=1.0)
    buy_interest_annual: float = Field(default=0.025, ge=0.0)
    borrow_fee_annual: float = Field(default=0.0115, ge=0.0)
    reverse_fee_bps: float = Field(default=2.0, ge=0.0)
    side_leverage: float = Field(default=1.5, ge=1.0)


class BLPXConfig(BaseModel):
    """BLPX model parameters."""

    rho: float = Field(default=0.01, ge=0.0)
    alpha_xx: float = Field(default=0.20, ge=0.0, le=1.0)
    alpha_yx: float = Field(default=0.15, ge=0.0, le=1.0)
    alpha_yy: float = Field(default=0.50, ge=0.0, le=1.0)
    lambda_pca: float = Field(default=0.10, ge=0.0)
    lambda_sector: float = Field(default=0.60, ge=0.0)
    beta_conf: float = Field(default=0.25, ge=0.0)
    winsor_sigma: float = Field(default=3.0, ge=0.0)
    blp_window: int = Field(default=504, ge=1)
    ewma_halflife: int = Field(default=120, ge=1)
    sector_eta: float = Field(default=0.5, ge=0.0)
    sector_gamma: float = Field(default=4.0, ge=0.0)
    target: str = Field(default="topix_residual")
    use_raw_target: bool = Field(default=False)
    asymmetry_mode: str = Field(default="scalar")
    asymmetry_delta: float = Field(default=0.30, ge=0.0)
    gap_open_coef: float = Field(default=0.70, ge=0.0, le=1.0)
    topix_beta_coef: float = Field(default=0.60, ge=0.0, le=1.0)
    gap_open_coef_neg: float = Field(default=0.60, ge=0.0, le=1.0)
    topix_beta_coef_neg: float = Field(default=0.60, ge=0.0, le=1.0)
    vol_adjusted_target: bool = Field(default=False)
    execution_target_cost_adjustment: str = Field(default="none")
    asymmetry_post_gap_delta: float = Field(default=0.0, ge=0.0)
    asymmetry_post_gap_mode: str = Field(default="signal_split")
    frobenius_scale_priors: bool = Field(default=False)
    macro_confidence_enabled: bool = Field(default=True)
    macro_kappa_enabled: bool = Field(default=True)
    macro_direction_enabled: bool = Field(default=True)
    macro_kappas: tuple[float, float, float] = Field(default=(3.0, 0.5, 0.5))
    macro_surprise_halflife_mean: float = Field(default=20.0, ge=1.0)
    macro_surprise_halflife_vol: float = Field(default=60.0, ge=1.0)
    copula_enabled: bool = Field(default=True)
    copula_blend_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    copula_dynamic_blend: bool = Field(default=True)
    copula_stress_threshold: float = Field(default=1.5, ge=0.0)
    copula_nu_init: float = Field(default=5.0, ge=2.0)


class ProductionV2RunConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    model_name: str = Field(default="production_residual_blpx")
    version: str = Field(default="production_residual_blpx_v2")

    # Sub-models
    blpx: BLPXConfig = Field(default_factory=BLPXConfig)
    costs: CostConfig = Field(default_factory=CostConfig)

    # RuleD
    pit_rolling_window: int = Field(default=252, ge=1)
    tertile_low_pct: float = Field(default=33.3333, ge=0.0, le=100.0)
    tertile_high_pct: float = Field(default=66.6667, ge=0.0, le=100.0)
    mult_low: float = Field(default=0.75, ge=0.0, le=1.0)
    mult_mid: float = Field(default=1.00, ge=0.0, le=1.0)
    mult_high: float = Field(default=1.00, ge=0.0, le=1.0)
    baseline_gross: float = Field(default=2.0, ge=0.0)
    fallback_multiplier: float = Field(default=1.00, ge=0.0, le=1.0)

    # portfolio
    long_count: int = Field(default=5, ge=1)
    short_count: int = Field(default=5, ge=1)
    minvar_enabled: bool = Field(default=False)
    minvar_alpha: float = Field(default=0.5, ge=0.0, le=1.0)

    # ranking
    ranking_mode: str = Field(default="mu_over_sigma")
    sigma_floor: float = Field(default=1.0e-6, gt=0.0)

    # multi-horizon
    mh_blend_enabled: bool = Field(default=False)
    mh_horizons: tuple[int, ...] = Field(default=(1, 3, 5))
    mh_weights: tuple[float, ...] = Field(default=(0.8, 0.1, 0.1))
    mh_mu_file_pattern_h: str = Field(default="matrices/mu_gap_h{h}_{date}.npy")
    mh_omega_file_pattern_h: str = Field(default="matrices/omega_gap_h{h}_{date}.npy")

    # cs overlay
    cs_overlay_enabled: bool = Field(default=False)
    cs_overlay_weight: float = Field(default=0.05, ge=0.0)
    cs_rank_reversal_file_pattern: str = Field(default="matrices/rank_reversal_{date}.npy")

    # fractional diff
    frac_diff_enabled: bool = Field(default=True)
    frac_diff_d: float = Field(default=0.1, ge=0.0)
    frac_diff_threshold: float = Field(default=1.0e-5, gt=0.0)
    frac_diff_window: int = Field(default=100, ge=1)

    # ml overlay
    ml_overlay_enabled: bool = Field(default=True)
    ml_overlay_model_dir: Path = Field(default=Path("models/ml_order_overlay/phase2_8"))
    ml_overlay_use_ticker: bool = Field(default=True)
    ml_overlay_use_classification: bool = Field(default=False)
    ml_overlay_per_ticker_interactions: bool = Field(default=True)
    ml_overlay_p_trade_ema_span: float = Field(default=0.0, ge=0.0)
    ml_overlay_fallback_to_baseline: bool = Field(default=True)

    # residualization
    residualization_enabled_for_p3: bool = Field(default=True)
    residualization_beta_window: int = Field(default=60, ge=1)
    residualization_beta_winsor_sigma: float = Field(default=3.0, ge=0.0)
    residualization_beta_shrinkage: float = Field(default=0.05, ge=0.0)

    # fallback
    fallback_on_gap_data_missing: bool = Field(default=True, description="gap data 欠損時に flat position (w_final=0) を返す")
    fallback_on_audit_failure: bool = Field(default=True, description="数値監査失敗時に flat position (w_final=0) を返す")

    # `load_config_from_yaml` performs a pre-validation mapping:
    #   - keys with `blpx_` prefix are moved under the `blpx` sub-dict
    #   - keys with `costs_` prefix or the known cost fields are moved
    #     under the `costs` sub-dict
    #   - remaining flat keys map directly to ProductionV2RunConfig fields
    # Example: `blpx_rho` -> `blpx.rho`, `costs_slippage_bps` -> `costs.slippage_bps`

    # file paths
    gap_input_dir: Path | None = Field(default=None)
```

### 6.4.1 `load_config_from_yaml()` 修正

```python
# Mapping from flat YAML to nested Pydantic sub-models
_BLPX_PREFIX = "blpx_"
_COSTS_FIELDS = {
    "slippage_bps", "cost_bps_per_gross", "overnight_alpha_long",
    "overnight_alpha_short", "buy_interest_annual", "borrow_fee_annual",
    "reverse_fee_bps", "side_leverage",
}


def _map_flat_to_nested(raw: dict) -> dict:
    """Move flat keys prefixed with ``blpx_`` (or in the cost field set)
    under their respective sub-dicts before Pydantic validation."""
    out = {}
    blpx = {}
    costs = {}
    for k, v in raw.items():
        if k.startswith(_BLPX_PREFIX):
            blpx[k[len(_BLPX_PREFIX):]] = v
        elif k in _COSTS_FIELDS:
            costs[k] = v
        else:
            out[k] = v
    if blpx:
        out["blpx"] = blpx
    if costs:
        out["costs"] = costs
    return out


def load_config_from_yaml(path: str | Path, *, strict: bool = False) -> AppConfig:
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)

    # Split top-level AppConfig fields from V2 production fields.
    app_config_fields = {
        "strategy", "risk", "kabu", "tachibana", "broker_provider",
        "output_base_dir", "output_live_dir", "run_audit", "gap_distribution_dir",
    }
    app_kwargs = {k: v for k, v in raw.items() if k in app_config_fields}
    v2_kwargs = {k: v for k, v in raw.items() if k not in app_config_fields}

    mapped = _map_flat_to_nested(v2_kwargs)
    v2_cfg = ProductionV2RunConfig(**mapped)
    return AppConfig(v2=v2_cfg, **app_kwargs)
```

### 6.5 追加：コスト設定の移動と `BacktestEngine` 対応

`src/leadlag/execution/backtester.py` は現在 `app_config.strategy` からコストパラメータ（`slippage_bps`, `overnight_alpha_*`, 金利等）を取得している。`StrategyConfig` を廃止するときは、以下を同時に実施する。

1. `ProductionV2RunConfig` 内に `CostConfig` サブモデルを追加。
2. `BacktestEngine` を `app_config.v2.costs` 参照に更新。
3. 旧 `app_config.strategy.slippage_bps` 等は deprecated 警告を出しつつ、`app_config.v2.costs` からの読み出しに移行。
4. 全コスト関連設定の単一正本は `ProductionV2RunConfig.costs` とする。

### 6.6 変更内容が一意に定まるか

**定まらない点**:

- `StrategyConfig` を完全に削除するか、`LegacyConfig` として残すか
- `AppConfig` を `ProductionConfig` + `BacktestConfig` に分割するか、一本化するか

**推奨デフォルト**:

- `StrategyConfig` は `archive/legacy_src/config/schemas.py` へ移設。
- `ProductionV2RunConfig` を `AppConfig.v2` として維持する（`production` フィールド名変更は行わない）。
- `AppConfig` の `strategy` フィールドは deprecated として警告を出す。
- 設定 YAML は人間が読みやすいフラット形式を維持し、`load_config_from_yaml` 内で `blpx` / `costs` サブモデルにマッピング。具体的には：
  - `blpx_` プレフィックスを持つ flat キー → `blpx` dict
  - `costs_` プレフィックスまたは既知のコストフィールド名 → `costs` dict
  - それ以外 → `ProductionV2RunConfig` トップレベル

### 6.7 完了基準

- flat `production.yaml`（`blpx_*` プレフィックス）で `load_config_from_yaml` が成功
- nested 旧 YAML でも `load_config_from_yaml` が成功（互換変換後）
- 新・旧両方の YAML で同一 `w_final` を出力
- `BacktestEngine` が `app_config.v2.costs` を参照して動作
- `ProductionBLPXModel` が `app_config.v2.blpx` パラメータを正しく読む
- 全テスト pass

---

## 7. Phase 4: 本番パイプラインワンステップ化

### 7.1 目的

`run_gap_distribution.sh` (6:30) と `run_decision_v2.sh` (9:05) を一本化。

### 7.2 前提

Step 1 (`omega_struct`) は 6:30 バッチで事前計算。Step 2 (gap 補正) を 9:05-9:10 に on-demand 実行。on-demand 経路では `omega_struct` ファイルを直接読まず、`compute_blp_signal(return_matrices=True)` から `Omega_struct` を再構成する。

### 7.3 新規作成: `src/leadlag/runner/__init__.py`

```python
"""Production runner package."""

from leadlag.runner.production import ProductionRunner

__all__ = ["ProductionRunner"]
```

### 7.4 新規作成: `src/leadlag/runner/production.py`

```python
"""One-step production runner for V2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RunnerInputs:
    """Inputs required for the production decision."""
    trade_date: str
    df_exec: pd.DataFrame
    gap_input_dir: Path | None
    current_prices: dict[str, float]
    use_file_cache: bool = True
    previous_positions: dict[str, int] | None = None


class ProductionRunner:
    """High-level runner that orchestrates the full V2 decision flow."""

    def __init__(self, app_config: Any) -> None:
        self.app_config = app_config
        # ProductionBLPXModel expects the v2-level config dict with blpx/costs sections.
        blpx_model = ProductionBLPXModel(app_config.v2.model_dump())
        overlay_model = None
        if app_config.v2.ml_overlay_enabled:
            from leadlag.models.ml_order_overlay import load_overlay_model
            overlay_model = load_overlay_model(
                Path(app_config.v2.ml_overlay_model_dir)
            )
        self.model = ProductionV2Model(
            app_config.v2,
            blpx_model=blpx_model,
            overlay_model=overlay_model,
        )

    def run(self, inputs: RunnerInputs) -> dict:
        # 1. 市場開閉判定
        # 2. df_exec 検証
        # 3. ポートフォリオ構築とシグナル計算
        return self.model.decide(
            trade_date=inputs.trade_date,
            gap_input_dir=inputs.gap_input_dir,
            df_exec=inputs.df_exec,
            current_prices=inputs.current_prices,
            overlay_enabled=self.app_config.v2.ml_overlay_enabled,
            use_file_cache=inputs.use_file_cache,
        )
```

### 7.5 変更: `src/leadlag/cli.py`

`daily` サブコマンドを `ProductionRunner` に委譲。`--use-file-cache` オプションを追加（default: True）。shadow/audit 目的で False を選べる。

```python
def _handle_daily(args: argparse.Namespace) -> int:
    from leadlag.runner.production import ProductionRunner, RunnerInputs

    # build inputs
    df_exec = _load_or_build_df_exec(app_config)
    manual_opens = _build_current_prices_from_df_exec(df_exec, args.trade_date)
    gap_dir = _resolve_gap_dir(args)

    inputs = RunnerInputs(
        trade_date=args.trade_date,
        df_exec=df_exec,
        gap_input_dir=gap_dir,
        current_prices=manual_opens,
        use_file_cache=getattr(args, "use_file_cache", True),
        previous_positions=_load_previous_positions(args),
    )

    runner = ProductionRunner(app_config)
    result = runner.run(inputs)
    return 0
```

### 7.6 新規作成：Step 1 用スクリプト `src/leadlag/pipeline/compute_omega_struct.py`

`tools/research/compute_structured_prediction_covariance.py` から Step 1 ロジックを昇格。Step 1 は 6:30 バッチで `Omega_struct`（標準化相関構造）を前日確定情報までで計算する。

- 前日 18:00 JST（米国終値後）〜 06:30 JST までの間に実行
- `ProductionBLPXModel(app_config.v2.model_dump())` を使用
- 当日 gap はまだ不明なため、`compute_blp_signal` には gap_override=None / betas_t=None / topix_night_t=None で `return_matrices=True` を呼び、`Sigma_XX` / `Sigma_YX` / `Sigma_YY` / `B_struct` から `_omega_from_blp_res` で `Omega_struct` を計算
- 出力: `var/live/pipeline_data/omega_struct/omega_struct_YYYYMMDD.npy`
- 将来的には `tools/research/compute_structured_prediction_covariance.py` を削除

### 7.7 バッチスクリプト更新

`scripts/batch/run_gap_distribution.sh`:

```bash
#!/bin/bash
# Step 1 専用：omega_struct 事前計算
python3 src/leadlag/pipeline/compute_omega_struct.py --config configs/production/production.yaml
```

`scripts/batch/run_decision_v2.sh`:

```bash
#!/bin/bash
# Step 2 専用：decision on-demand
python3 -m leadlag.cli daily --config configs/production/production.yaml
```

### 7.8 変更内容が一意に定まるか

**定まらない点**:

- runner の責務範囲（発注まで含めるか、decision 生成までか）
- `execute_post_decision_flow` を runner 内に含めるか分離するか

**推奨デフォルト**:

- `ProductionRunner.run()` は decision 生成までを責務とする。
- 発注は既存 `execute_post_decision_flow` を runner から呼び出す（移設は Phase 5 以降）。

### 7.9 完了基準

- `leadlag.cli daily` が 9:05-9:10 で一括実行可能
- 実行時間 < 15 分
- シャドー運用 3 営業日以上で指標差分なし

---

## 8. Phase 5: データレイヤー抽象化（optional / 別計画書推奨）

注意：本 Phase は主要目的（research 依存断ち・gap on-demand 化・config 整理）と独立した大規模変更。Phase 1-4 完了後、または別計画書として実施することを推奨。最小限の修正では `fetcher.py`, `cache.py`, `preprocessor.py` の責務整理に留め、新規 ABC/SessionState/Patch 移設は次段階とする。

### 8.1 新規作成: `src/leadlag/data/providers/__init__.py`

```python
"""Data provider abstraction layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class OHLC:
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


class DataProvider(ABC):
    """Abstract data provider for OHLC and quote data."""

    @abstractmethod
    def fetch_daily_ohlc(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> dict[str, pd.DataFrame]:
        """Return dict of {ticker: DataFrame indexed by date}."""
        ...

    @abstractmethod
    def fetch_intraday_quote(
        self,
        tickers: list[str],
        at: Any,
    ) -> dict[str, float]:
        """Return best available price for each ticker at a given time."""
        ...

    @abstractmethod
    def source_name(self) -> str:
        ...
```

### 8.2 新規作成: `src/leadlag/data/providers/yfinance_provider.py`

`fetcher.py` から yfinance ロジックを `YFinanceProvider` として分離。ただし初回は `fetcher.py` の内部呼び出しを `YFinanceProvider.fetch_daily_ohlc` に置き換えるラッパーとして導入し、完全な置き換えは段階的に行う。

### 8.3 新規作成: `src/leadlag/data/providers/tachibana_provider.py`

Tachibana API から 9:10 pDPP を取得。

### 8.4 新規作成: `src/leadlag/data/patches/__init__.py`

```python
"""Data quality patches with provenance tracking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PatchResult:
    patched: bool
    patch_name: str
    affected_dates: list[str]
    original_values: dict[str, float]
    patched_values: dict[str, float]


class DataPatch(ABC):
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> PatchResult:
        ...
```

### 8.5 新規作成: `src/leadlag/data/patches/etf_1629_nav.py`

`fetcher.py` の 1629.T NAV パッチをここに移動。

### 8.6 新規作成: `src/leadlag/data/session.py`

```python
"""Session state model for execution rows."""

from __future__ import annotations

from enum import Enum


class SessionState(Enum):
    PRE_OPEN = "pre_open"          # before JP market open
    POST_OPEN = "post_open"        # after 09:10 prices available
    POST_CLOSE = "post_close"      # after JP market close
    SETTLED = "settled"            # all prices confirmed
```

### 8.7 変更: `src/leadlag/data/preprocessor.py`

- `is_provisional` 列は当面維持。
- 新たに `session_state` 列を追加（`pre_open` / `post_open` / `settled`）。
- 1629.T NAV パッチ等は `data/patches/` へ移動。
- `fetcher.py` 内の yfinance ロジックは `data/providers/yfinance_provider.py` を呼ぶ形に変更。

### 8.8 変更内容が一意に定まるか

**定まらない点**:

- `OHLC` 型の詳細（volume 含むか、調整後か等）
- `DataProvider` のキャッシュ戦略
- `SessionState` をどこまで導入するか

**推奨デフォルト**:

- 当面は `preprocessor.py` に `session_state` 列を追加し、`is_provisional` は deprecated マーカーとして残す。
- キャッシュは `market_data_cache.py` をそのまま利用。

### 8.9 完了基準

- `DataProvider` ABC が定義されている
- `YFinanceProvider` がテスト可能（network なし）
- `preprocessor.py` から 1629.T パッチロジックが分離
- 全テスト pass

---

## 9. Phase 6: 本番エントリの CLI 一本化

### 9.1 目的

`tools/production/run_daily_production_v2.py` と `python3 -m leadlag.cli decision` の二本立てを解消し、本番日次実行の唯一のエントリポイントを CLI `decision` に一本化する。

これにより、

- 本番パスが `ProductionRunner` / `v2_bridge` のみになる
- gap 行列不在時の on-demand 計算が CLI からも利用可能になる
- 設定の検証・価格取得・リスクチェック・注文送信が一貫したコードパスで行われる
- `run_daily_production_v2.py` の保守・二重実装が不要になる

### 9.2 実装内容

#### 9.2.1 `src/leadlag/cli.py` / `src/leadlag/execution/v2_bridge.py` の拡張

CLI `decision` サブコマンドに以下を追加する。

- `--trade-date latest`
  - `latest_weights.csv` の `trade_date` セルを読み、前日または当日の取引日を解決
  - `latest_weights.csv` 不在時は、本日が取引日なら本日、非取引日なら前取引日（`previous_trading_day`）へ fallback
  - どちらの経路でも `logger.warning` で記録
- `--dry-run`
  - ポジション計算のみ行い、発注処理（`execute_post_decision_flow`）をスキップ
  - `write_production_files(..., dry_run=True)` によりファイル出力も抑制
- `self-test` サブコマンド、または `--self-test` フラグ
  - 現在 `run_daily_production_v2.py` の `run_self_tests()` で行っている以下をカバー
    - `solve_baseline_style` の net/gross チェック
    - `get_rolling_pit_bin` Low/Medium/High および履歴不足 fallback
    - `run_leakage_audit` の valid / same-date
    - `run_numerical_audit` の valid weights
    - コスト式の一致
  - `tests/unit/test_cli_self_test.py` 等を新規作成し、スクリプト側の `assert` を `pytest` に移行

#### 9.2.2 `tools/production/run_daily_production_v2.py` の削除

- スクリプトを削除
- `AGENTS.md` の「よく使うコマンド」例を CLI ベースに更新
- 運用スケジューラ（cron 等）の呼び出し先を

  ```bash
  python3 -m leadlag.cli decision --trade-date latest --api-enable
  ```

  に変更

#### 9.2.3 設定・ドキュメントの更新

- `configs/production/production.yaml` には影響なし
- `docs/ARCHITECTURE.md` において、本番エントリが CLI `decision` であることを明記
- `run_daily_production_v2.py` が deprecated となった旨を `AGENTS.md` に追記

### 9.3 変更内容が一意に定まるか

**定まる点**:

- 本番エントリは CLI `decision` のみとする
- `run_daily_production_v2.py` は削除する
- `latest` 日付解決は `v2_bridge` 内に helper 関数として実装する

**定まらない点**:

- `dry-run` を `--dry-run` フラグとするか、別サブコマンドとするか
- `self-test` を `cli.py` のフラグとするか、独立サブコマンドとするか

**推奨デフォルト**:

- `dry-run` は `--dry-run` フラグ
- `self-test` は独立サブコマンド `python3 -m leadlag.cli self-test` とする

### 9.4 完了基準

- `python3 -m leadlag.cli decision --trade-date latest` が `run_daily_production_v2.py --trade-date latest` と同等のファイルを出力する
- `python3 -m leadlag.cli decision --dry-run` でファイル出力・発注が行われない
- `python3 -m leadlag.cli self-test` が全ての診断をパスする
- `tools/production/run_daily_production_v2.py` が削除されている
- 全テスト pass
- シャドー運用で 3 営業日以上、旧スクリプトと CLI の出力が一致

---

## 10. 検証手順

### 10.1 数値一致検証

各フェーズ後に以下を実行。

```bash
python3 scripts/capture_v2_baseline.py
python3 -m pytest tests/regression/test_v2_baseline.py -v
```

`w_final` 最大差分 < 1e-12、`scores` 最大差分 < 1e-12。

### 10.2 テスト実行

```bash
bash scripts/run_tests_parallel.sh
```

### 10.3 バックテスト再実行

```bash
python3 -m leadlag.cli backtest \
    --config configs/production/production.yaml \
    --start-date 2015-01-05 \
    --gap-dir var/live/pipeline_data/gap_adjusted_distribution/latest
```

`src/research/scripts/backtest/run_production_backtest.py` は deprecated のため使用しない。

基準値との差分:

- w_final / scores 最大差分: < 1e-12（同一モデル・同一 gap 行列なら exact を目指す）
- net Sharpe: Δ < 0.05
- total return: Δ < 0.5%
- max DD: Δ < 0.5%
- turnover: Δ < 0.05

### 10.4 シャドー運用

```bash
python3 tools/validation/monitor_residual_blpx_shadow_performance.py \
    --config configs/production/production.yaml \
    --days 10
```

---

## 11. リスク管理

### 11.1 ロールバック方針

1. 各フェーズは独立ブランチ `refactor/phase-N` で実施。
2. 本番影響フェーズ（2, 4）では `git tag` を打つ。
3. `production.yaml` の変更は旧ファイル `production.yaml.legacy` を並列保存。
4. 不具合発生時は `git checkout -- configs/production/production.yaml` で即時復旧可能。

### 11.2 不採用条件

以下に該当する場合は打ち切り。

1. net Sharpe Δ < -0.2
2. fallback 率が 5% 以上上昇
3. シャドー運用で 3 営業日以上連続して本番 V2 と不一致
4. 全テスト通過に失敗し、3 営業日以内に解消しない
5. on-demand 計算時間が shadow 運用で 9:10 カットオフ前に完了しない、または 30 秒を超過

---

## 12. まとめ：推奨実装仕様

### 12.1 ほぼ確定的な変更（must）

これらは逸脱するとテスト・監査・運用で破綻する。

- 本番パスから `research` パッケージへの import を断つ
- Step 2 gap 調整済み分布（`mu_gap` / `Omega_gap`）の計算を on-demand 化
- `_flatten_nested_yaml` 等、設定解決の ad-hoc ロジックを廃止
- 全テスト pass と数値一致を維持

### 12.2 推奨デフォルトを定めた変更（decision）

| 選択肢 | 推奨値 |
|---|---|
| BLPX モデル名 | `ProductionBLPXModel` |
| `blp_base.py` 置き場 | `src/leadlag/models/blp_base.py` |
| gap 補正配置 | `src/leadlag/core/gap_adjustment.py` |
| `.npy` ファイル | キャッシュ・監査証跡として残す（Step 2 ファイル優先、on-demand はフォールバック） |
| 設定モデル | `ProductionConfig` 新設、`StrategyConfig` 廃止 |
| runner 責務 | decision 生成まで |
| データレイヤー | `DataProvider` ABC（段階的導入） |

### 12.3 結論

**完全な一意性は達成できないが、上記の推奨デフォルトを採用すれば、実装者間で解釈の食い違いは最小化できる。** 異なる選択をする場合は、その理由をテストとドキュメントで担保すること。

---

## 13. 関連ファイル

- `src/leadlag/models/production_v2.py`
- `src/leadlag/models/blpx.py`
- `src/leadlag/models/blp_base.py`
- `src/leadlag/runner/production.py`
- `src/leadlag/execution/backtester.py`
- `src/leadlag/execution/v2_bridge.py`
- `src/leadlag/config/schemas.py`
- `src/leadlag/data/fetcher.py`
- `src/leadlag/data/preprocessor.py`
- `src/leadlag/data/market_data.py`
- `tools/research/compute_gap_adjusted_distribution.py`
- `tools/research/compute_structured_prediction_covariance.py`
- `tools/production/run_daily_production_v2.py`（Phase 6 で削除）
- `src/research/models/sector_relative_ensemble_blp_enhanced.py`
- `src/research/models/blp_base.py`
- `src/research/models/base.py`
- `configs/production/production.yaml`
- `AGENTS.md`
- `docs/ARCHITECTURE.md`

---

## 14. 実行進捗

### 2026-08-13 時点

- [x] Phase 0: 回帰ベースライン整備
  - `tests/regression/__init__.py`, `conftest.py`, `test_v2_baseline.py` 新設
  - `scripts/capture_v2_baseline.py` 新設
  - ベースライン `tests/regression/baselines/v2_snapshot_v20260813.json` および gap 行列・PIT history をコミット
- [x] Phase 1: `research` パッケージ import 断ち切り（前セッションで完了済）
- [x] Phase 2: Step 2 gap 調整済み分布 on-demand 化（前セッションで完了済）
- [x] Phase 10: ベースライン V2 バックテスト実行
  - 期間: 2026-06-15 〜 2026-08-13（40 日）
  - 指標: net Sharpe 10.49, AR 141.79%, MDD -6.96%, ターンバー平均 1.46, fallback 0%
  - `reports/refactoring_roadmap/baseline_metrics_20260813.json` 保存
- [x] 全テスト通過: 507 tests ALL PASSED（新規 `tests/regression/test_v2_baseline.py` 含む）

### 2026-08-13 時点（本セッション完了）

- [x] Phase 3/6: `production.yaml` フラット化 + `_flatten_nested_yaml` 廃止 + `build_app_config_from_dict` 対応
  - `configs/production/production.yaml` をフラット化。旧ネスト版は `configs/production/production.yaml.legacy` に保存。
  - `src/leadlag/config/schemas.py` から `_flatten_nested_yaml` を廃止。設定正規化は `_map_flat_to_nested` のみで実施。
  - `parse_run_config` / `build_app_config_from_dict` / `load_config_from_yaml` が新旧両方の YAML レイアウトを受け入れるように更新。
- [x] Phase 5: `ProductionV2Model` クラス中心整理（`generate_v2_production_portfolio` 解体）
  - `ProductionV2Model._file_cache_or_flat` を新設し、file cache 読み込み / flat fallback をクラス内で完結。
  - `decide()` が on-demand 計算、file cache、on-demand 失敗時の file cache フォールバックを統一的に制御。
  - `generate_v2_production_portfolio()` は `ProductionV2Model.decide()` への薄いラッパーに変更。
- [x] Phase 8: `DataProvider` ABC（decision 項目・最小実装）
  - `src/leadlag/data/providers/__init__.py` に `DataProvider` ABC、`YFinanceProvider`、`TachibanaProvider` を新設。
  - `src/leadlag/data/session.py` に `Session` / `SessionState` / `ExchangeCalendarPatch` の雛形を新設。
  - `tests/unit/test_data_provider.py` を新設し、両プロバイダの基本動作を検証。
- [x] 全テスト通過: `bash scripts/run_tests_parallel.sh` で 507 tests ALL PASSED（ruff F821 も pass）
- [x] 本来の full 期間バックテスト: 2015-01-05 〜 2026-08-13
  - `var/live/pipeline_data/gap_adjusted_distribution/20260731_024303`（2020-01-06 〜 2026-07-29 の gap 行列）を使用。2020-01-05 以前と 2026-07-30 以降は on-demand BLPX 計算で gap 行列を補完。
  - 指標: net Sharpe 4.1214, AR 199.04%, MDD -8.26%, ターンバー平均 1.43, fallback 0%
  - 最終 wealth: 875,838,871x, 総コスト 5.28（スリッページ 81.1%, 逆日歩 11.3%, ロング金利 5.8%, 貸株 1.8%）
  - 結果保存先: `var/results/20260815_111443_full_2015_20260813`

### 未完了・次回対象

- [ ] Phase 10.4: シャドー運用 3 営業日（連続したライブ市場の観察が必要なため、セッション外で実施）

### 作業ブランチ

- `refactor/complete-plan`

