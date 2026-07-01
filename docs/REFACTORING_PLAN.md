# リファクタリング計画 (2026-07-01)

> **ステータス**: ✅ 全フェーズ完了 (2026-07-01)
> **前提**: ファイル移動 + importパス修正 + テスト修正 + ドキュメント更新

---

## 現状の問題点

### 問題1: `src/` 配下に `leadlag/` 外の散在パッケージ
- `src/features/` (4ファイル) — ヒンジ特徴量実験コード
- `src/models/` (7ファイル) — Hinge オーバーレイ実験モデル
- `src/reports/` (2ファイル) — sprint3 レポート生成
- **参照元**: `scripts/run_sprint3a_hinge_features.py`, `scripts/run_sprint3b_hinge_interactions.py`, `tests/unit/test_sprint3b.py` のみ
- **問題**: 本番パッケージ `leadlag/` と同階層に実験コードが混在、パッケージ境界が不明確

### 問題2: `leadlag/diagnostics/` にsprint実験コードが混入
- `sprint0.py`, `sprint0_qa.py`, `sprint1_experiments.py` (3ファイル)
- **参照元**: `scripts/run_sprint0_*.py`, `scripts/run_sprint1_*.py`, `scripts/run_sprint2_*.py`, `scripts/run_sprint3_*.py`, `tests/unit/test_sprint0_*.py`, `tests/unit/test_sprint1*.py`
- **問題**: 本番パッケージ内に実験モジュールが存在し、本番コードと誤認されるリスク

### 問題3: `leadlag/cost/` の単一ファイルモジュール
- `cost_calculator.py` (1ファイルのみ)
- **参照元**: `tests/unit/test_cost_calculator.py` のみ
- **依存**: `leadlag/execution/slippage_model.py`, `leadlag/execution/order_book_schema.py` へ依存
- **問題**: モジュールとしての独立性が低く、`execution/` の一部として統合すべき

### 問題4: `leadlag/execution/` の肥大化
- LOB/スリッページ関連5ファイルがランナー層に混在:
  - `order_book_schema.py`, `order_book_cost.py`, `slippage_model.py`, `execution_constraints.py`, `live_quote_logger.py`
- **参照元**: `scripts/run_sprint2c_lob_slippage.py`, `leadlag/models/net_score_ranking_lob.py`, `leadlag/cost/cost_calculator.py`, `leadlag/execution/execution_constraints.py` (内部相互参照)
- **問題**: 実行ランナー（decision, fast, close, backtest）と執行ミクロ構造（LOB, スリッページ）が未分離

### 問題5: `scripts/` の肥大化（54ファイル）
- 実験スクリプト（`experiment_*.py`）: 24ファイル
- sprint実行スクリプト（`run_sprint*.py`）: 8ファイル
- バックテスト実行スクリプト（`run_*backtest*.py`）: 4ファイル
- バッチ/シェルスクリプト（`run_*.sh`, `run_*.bat`, `*.plist`, `*.ps1`）: 8ファイル
- テスト・接続確認（`test_*.py`）: 2ファイル
- その他: 8ファイル
- **問題**: 実験コードと運用バッチが混在、ファイル数が多く目的別の分離が必要

### 問題6: `tools/` の役割不明確（11ファイル）
- 本番ツール: `run_daily_production_v2.py`, `compute_gap_adjusted_distribution.py`
- 検証ツール: `validate_production_residual_blpx.py`, `monitor_residual_blpx_shadow_performance.py`
- 旧本番ツール: `run_daily_residual_blpx_shadow.py`, `run_daily_sector_relative_ensemble.py`
- 大規模分析スクリプト: `analyze_blp_enhanced_refined.py` (95KB), `final_model_selection.py` (71KB), `compute_gap_adjusted_distribution.py` (76KB)
- **問題**: 本番ツールと分析ツールが混在

### 問題7: `configs/` の研究設定混在
- 本番設定: `production.yaml`, `production_v2_primary_ruleD.yaml`, `production_v3_phase2.yaml`, `production_residual_blpx.yaml`
- 研究設定: `research_*.yaml` (6ファイル), `final_model_selection.yaml`
- 業務設定: `sector_exposure_map.yaml`
- アーカイブ: `configs/archive/` (18ファイル)
- **問題**: 本番設定と研究設定が同階層に混在

---

## リファクタリング計画

### Phase R1: 実験パッケージの `experiments/` への統合

**目標**: `src/features/`, `src/models/`, `src/reports/`, `leadlag/diagnostics/` を `src/experiments/` パッケージに統合

**移動先構造**:
```
src/experiments/
├── __init__.py
├── features/
│   ├── __init__.py          # src/features/__init__.py から移動
│   ├── asset_exposures.py
│   ├── feature_selection_fdr.py
│   ├── hinge_features.py
│   └── hinge_interactions.py
├── models/
│   ├── __init__.py          # src/models/__init__.py から移動
│   ├── hinge_elasticnet_overlay.py
│   ├── hinge_interaction_elasticnet.py
│   ├── hinge_interaction_gbdt.py
│   ├── hinge_interaction_overlay.py
│   ├── hinge_interaction_ridge.py
│   ├── hinge_overlay.py
│   └── hinge_ridge_overlay.py
├── reports/
│   ├── __init__.py          # src/reports/__init__.py から移動
│   ├── sprint3a_hinge_report.py
│   └── sprint3b_hinge_interaction_report.py
└── diagnostics/
    ├── __init__.py          # 新規作成
    ├── sprint0.py           # leadlag/diagnostics/sprint0.py から移動
    ├── sprint0_qa.py        # leadlag/diagnostics/sprint0_qa.py から移動
    └── sprint1_experiments.py
```

**import修正対象** (14ファイル):
- `scripts/run_sprint3a_hinge_features.py`: `from features...` → `from experiments.features...`
- `scripts/run_sprint3b_hinge_interactions.py`: `from features...` → `from experiments.features...`
- `scripts/run_sprint0_diagnostics.py`: `from leadlag.diagnostics...` → `from experiments.diagnostics...`
- `scripts/run_sprint0_qa.py`: 同上
- `scripts/run_sprint1_experiments.py`: 同上
- `scripts/run_sprint1_aum1m_tachibana.py`: 同上
- `scripts/run_sprint2_cost_aware_aum1m.py`: 同上
- `scripts/run_sprint2b_qa.py`: 同上
- `scripts/run_sprint3a_hinge_features.py`: `from models...` → `from experiments.models...`, `from reports...` → `from experiments.reports...`
- `scripts/run_sprint3b_hinge_interactions.py`: 同上
- `tests/unit/test_sprint3b.py`: `from features...` → `from experiments.features...`
- `tests/unit/test_sprint0_diagnostics.py`: `from leadlag.diagnostics...` → `from experiments.diagnostics...`
- `tests/unit/test_sprint0_qa.py`: 同上
- `tests/unit/test_sprint1.py`: 同上
- `tests/unit/test_sprint1_aum1m.py`: 同上

**リスク**: 低 — 参照元が限定され、全て `sys.path.insert(0, "src")` 経由で参照しているため

---

### Phase R2: `cost/` の `execution/` への統合

**目標**: `leadlag/cost/cost_calculator.py` を `leadlag/execution/cost_calculator.py` へ移動

**移動**:
- `src/leadlag/cost/cost_calculator.py` → `src/leadlag/execution/cost_calculator.py`
- `src/leadlag/cost/__init__.py` → 削除

**import修正対象** (1ファイル):
- `tests/unit/test_cost_calculator.py`: `from leadlag.cost.cost_calculator import...` → `from leadlag.execution.cost_calculator import...`

**リスク**: 極低 — 参照元がテスト1件のみ

---

### Phase R3: `execution/` のLOB/スリッページサブパッケージ化

**目標**: LOB/スリッページ関連5ファイルを `leadlag/execution/microstructure/` サブパッケージに分離

**移動先構造**:
```
src/leadlag/execution/
├── microstructure/
│   ├── __init__.py
│   ├── order_book_schema.py
│   ├── order_book_cost.py
│   ├── slippage_model.py
│   ├── execution_constraints.py
│   └── live_quote_logger.py
├── cost_calculator.py       # Phase R2 で移動
├── config.py
├── helpers.py
├── decision.py
├── fast.py
├── close.py
├── backtest.py
└── backtester.py
```

**import修正対象** (6ファイル):
- `scripts/run_sprint2c_lob_slippage.py`: `from leadlag.execution.order_book_schema...` → `from leadlag.execution.microstructure.order_book_schema...`
- `src/leadlag/models/net_score_ranking_lob.py`: 同上 (3箇所)
- `src/leadlag/execution/cost_calculator.py` (Phase R2後): `from leadlag.execution.slippage_model...` → `from leadlag.execution.microstructure.slippage_model...`
- `src/leadlag/execution/microstructure/execution_constraints.py`: 内部参照 `from leadlag.execution.order_book_...` → 相対参照 `from .order_book_...`
- `src/leadlag/execution/microstructure/slippage_model.py`: 同上
- `src/leadlag/execution/microstructure/live_quote_logger.py`: 同上

**リスク**: 中 — 内部相互参照が多く、相対importへの変更が必要

---

### Phase R4: `scripts/` の目的別ディレクトリ分割

**目標**: 54ファイルを目的別に整理

**移動先構造**:
```
scripts/
├── experiments/          # 実験スクリプト (experiment_*.py 24ファイル)
│   ├── experiment_blpx_parameter_optimization.py
│   ├── experiment_correlation_estimation.py
│   ├── experiment_cs_features.py
│   ├── ... (全 experiment_*.py)
│   └── experiment_models.py
├── sprint/               # sprint実行スクリプト (run_sprint*.py 8ファイル)
│   ├── run_sprint0_diagnostics.py
│   ├── run_sprint0_qa.py
│   ├── ... (全 run_sprint*.py)
│   └── finalize_sprint2_report.py
├── backtest/             # バックテスト実行スクリプト (4ファイル)
│   ├── run_production_backtest.py
│   ├── run_overnight_holding_backtest.py
│   ├── run_overnight_robustness_analysis.py
│   └── run_selective_overnight_backtest.py
├── batch/                # バッチ/シェルスクリプト (8ファイル)
│   ├── run_decision.sh / .bat
│   ├── run_close_positions.sh / .bat
│   ├── run_auto_login.bat
│   ├── setup_scheduler.ps1
│   ├── setup_scheduler_macos.sh
│   ├── com.leadlag.close.plist
│   └── com.leadlag.decision.plist
├── test/                 # 接続テスト (2ファイル)
│   ├── test_tachibana_connection.py
│   └── test_tachibana_demo_order.py
└── print_top5_ref_results.py  # 単独ユーティリティ
```

**修正内容**:
- 各スクリプト内の `ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))` の相対パス調整（`".."` → `"../.."`）
- バッチスクリプト内のパス参照調整

**リスク**: 中 — スクリプト数が多く、パス修正の漏れが出やすい。バッチスクリプトのcron/launchd登録パスも要更新

---

### Phase R5: `tools/` の本番ツールと分析ツール分離

**目標**: 本番ツールと研究・分析ツールを分離

**移動先構造**:
```
tools/
├── production/            # 本番日次実行ツール
│   ├── run_daily_production_v2.py
│   └── compute_gap_adjusted_distribution.py
├── validation/            # 検証・シャドウツール
│   ├── validate_production_residual_blpx.py
│   ├── monitor_residual_blpx_shadow_performance.py
│   ├── run_daily_residual_blpx_shadow.py
│   ├── build_residual_blpx_shadow_run_package.py
│   └── apply_production_residual_blpx.py
├── research/              # 大規模分析スクリプト
│   ├── analyze_blp_enhanced_refined.py
│   ├── final_model_selection.py
│   └── run_daily_sector_relative_ensemble.py
└── backtest_sector_relative_ensemble.py  # 単独バックテストツール
```

**修正内容**:
- 各スクリプト内の `sys.path` / import パス調整
- ドキュメント（日次運用手順書）のパス参照更新

**リスク**: 中 — 日次運用手順書に記載のコマンドパスが変わるため、運用手順の更新が必要

---

### Phase R6: `configs/` の本番・研究分離

**目標**: 本番設定と研究設定を明確分離

**移動先構造**:
```
configs/
├── production/            # 本番設定
│   ├── production.yaml
│   ├── production_v2_primary_ruleD.yaml
│   ├── production_v3_phase2.yaml
│   └── production_residual_blpx.yaml
├── research/              # 研究用設定
│   ├── research_blp_enhanced.yaml
│   ├── research_blp_enhanced_refined.yaml
│   ├── research_blp_projection.yaml
│   ├── research_p0_p4_ensemble.yaml
│   ├── research_rrr_projection.yaml
│   ├── research_us_residual_prior.yaml
│   ├── research_us_residualization.yaml
│   └── final_model_selection.yaml
├── sector_exposure_map.yaml  # 業務設定（共通参照）
└── archive/               # 廃止設定
```

**修正対象**:
- 本番設定を参照する全スクリプト・ツールのパス修正
- `leadlag/execution/config.py` のデフォルトパス修正（該当する場合）

**リスク**: 高 — 本番設定ファイルのパス変更は日次運用に直接影響。**要入念な検証**

---

### Phase R7: `scratch/` の整理

**目標**: 一時スクリプトの整理

**対応**:
- `scratch/` 内の7ファイルを評価:
  - `vix_ic_analysis.py`, `vix_return_dispersion.py` — 実験コード → `scripts/experiments/` または `archive/`
  - `compare_configs.py`, `diagnose_backtest.py`, `check_inf_values.py`, `timing_test.py` — デバッグツール → `archive/` または削除
  - `test_tachibana_login.py` → `scripts/test/` に統合
- `.gitignore` に `scratch/` を追加（今後の一時ファイル蓄積防止）

**リスク**: 低

---

## 実行順序と依存関係

```
Phase R1 (実験パッケージ統合)     ──┐
Phase R2 (cost/ 統合)             ──┤── 独立実行可能
Phase R7 (scratch/ 整理)          ──┘
         │
Phase R3 (execution/ サブパッケージ)  ── R2 完了後に実行
         │
Phase R4 (scripts/ 分割)          ──┐
Phase R5 (tools/ 分離)            ──┤── R1完了後に実行推奨（import修正済みの状態で）
Phase R6 (configs/ 分離)          ──┘
```

**推奨実行単位**: Phase R1 → R2 → R3 を1セッションで実行し、テスト通過を確認。R4〜R6は別セッションで個別実行。

---

## 検証手順

各Phase完了後に以下を実行:

```bash
# 1. テストスイート全体実行
python3 -m pytest tests/ -v

# 2. 本番バックテスト実行（本番設定パス変更時）
python3 -m leadlag.cli backtest --config configs/production/production.yaml

# 3. import確認（移動したモジュール）
python3 -c "from experiments.features.hinge_features import build_full_feature_panel"
python3 -c "from experiments.diagnostics.sprint0 import run_sprint0_calculations"
python3 -c "from leadlag.execution.cost_calculator import CostCalculator"
python3 -c "from leadlag.execution.microstructure.slippage_model import compute_entry_cost_bps"
```

---

## 影響範囲サマリー

| Phase | 移動ファイル数 | import修正ファイル数 | リスク | 本番への影響 |
|---|---|---|---|---|
| R1 | 16 | 14 | 低 | なし（実験コードのみ） |
| R2 | 1 | 1 | 極低 | なし |
| R3 | 5 | 6 | 中 | なし（本番実行パス未使用） |
| R4 | 54 | ~20 | 中 | バッチスクリプトパス要更新 |
| R5 | 11 | ~11 | 中 | 日次運用手順書要更新 |
| R6 | 12 | ~10 | 高 | **本番設定パス変更** |
| R7 | 7 | 0 | 低 | なし |

---

## 今後の構造改善提案（別途検討）

- **`pyproject.toml` の `packages` 設定更新**: `experiments` パッケージを追加
- **`__init__.py` による公開API制限**: `leadlag/__init__.py` で本番モジュールのみエクスポート
- **`ruff` / `mypy` 設定の分割**: `experiments/` には緩いlint設定を適用
- **CI/CD 導入時**: `experiments/` のテストをoptional扱いに
