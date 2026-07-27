# Fractional Differentiation (d=0.1) 厳密監査レポート

**Date**: 2026-07-26
**Scope**: 本番適用中の Fractional Differentiation (`features.fractional_diff`, `d=0.1`) の実装に修正が必要かを体系的に調査。

---

## 1. 調査方法

- `python3 _check_syntax.py` — 構文チェック
- `python3 -m pytest tests/features/test_fractional_diff.py -v` — ユニットテスト
- `python3 -m pytest tests/integration/test_production_v2.py -v`
- `python3 -m pytest tests/integration/test_sector_relative_ensemble_blp_enhanced.py -v`
- `python3 -m pytest tests/integration/test_production_residual_blpx.py -v`
- カスタムエッジケーススクリプト: `scripts/experiments/audit_fractional_diff_edge_cases.py`

---

## 2. 主要な発見

### 2.1 本番経路の整合性

| 観点 | 結果 |
|---|---|
| `production.yaml` config | `features.fractional_diff.enabled: true`, `d: 0.1` で本番有効化済 |
| 旧config (production_v2_primary_ruleD.yaml, production_residual_blpx.yaml) | Fractional Diff セクション追記済、本番経路と整合 |
| `sre.py` (旧PCA-Ensemble) | `frac_diff_*` パラメータを `build_common_inputs` に渡すよう修正済 |
| `blp_base.py` | Fractional Diff config を `build_common_inputs` に正しく伝播 |
| `pipeline.py` `build_common_inputs` | US return 列に対して fractional diff を適用し、`all_returns_raw` を構築 |
| `compute_gap_adjusted_distribution.py` | `SectorRelativeEnsembleBLPEnhancedModel` → `build_common_inputs` 経路で fractional diff が反映 |
| `production_v2.py` | gap 行列をロードするだけ、fractional diff は gap 行列生成時点で反映済 |

**結論**: 本番パイプライン全体で fractional diff `d=0.1` は正しく有効化されている。ルックアヘッドリークはなし。

### 2.2 数値・理論上の問題

#### P2-002 / 改善済み: NaN の 0 埋めが無警告だった

- `fractional_diff` は入力に NaN が含まれると、lookback window 分（最大 `window` 日）NaN を伝播させる。
- `apply_fractional_diff_to_df_exec` および `build_common_inputs` で `fillna(0.0)` しており、データ品質問題を静かに隠蔽していた。
- 本番では `preprocess_data` が NaN 行をスキップするため、通常は発生しないが、Tachibana 価格注入やデータパイプライン障害時に NaN が混入するリスクあり。

**修正**: NaN が存在する場合に `logger.warning` を発し、0 埋めを行う前に警告を記録。

- `src/leadlag/features/fractional_diff.py` (`apply_fractional_diff_to_df_exec`)
- `src/leadlag/core/pipeline.py` (`build_common_inputs`)

#### P2-003: docstring 修正済み

- `fractional_diff()` の docstring は「NaN warmup period」と記載されていたが、実装は partial weights を使用。
- 現在のコードでは docstring が partial-window 実装に合わせて修正済み。
- `apply_fractional_diff_to_df_exec` のコメント「Forward-fill the warmup NaNs」が誤解を招くため、正確なコメントに修正済。

#### P3-001: 弱いテストを強化

- `test_no_nans_after_fill` は「NaN がないこと」だけを確認していた。
- 追加テスト:
  - `test_nan_input_is_filled_with_zero_and_warns`: NaN 入力が window 分伝播し、0 埋め + 警告されることを検証。
  - `test_window_truncate_bias_documented`: `d=0.1` + `window=100` での重み打ち切りバイアスを文書化・数値化。

### 2.3 重み打ち切りによる理論的バイアス

これは **実装バグではなく、メソッドの限界**。

| d | `compute_weights` での重み数 | 重み合計 | `window=100` で定数系列 (c=5) に対する最終出力 | 実効重み合計 |
|---|---|---|---|---|
| 0.1 | 4,076 | 0.408 | 2.95 | ~0.59 |
| 0.5 | 927 | 0.019 | 0.28 | ~0.056 |
| 1.0 | 2 | 0.000 | 0.00 (最初の1点 NaN) | 0.00 |

**解釈**:

- `d=0.1` は重みの減衰が極めて遅い (`|w_k| ~ k^{-1.1}`)。
- `threshold=1e-5` で 4,076 個の重みを計算するが、`window=100` で打ち切るため、実際に使われるのは先頭 100 個。
- 先頭 100 個の重み合計は ~0.59 のまま残り、定数系列に対しても ~59% のレベルが残る。
- これは真の分数階差分ではないが、**ウォークフォワード検証で 12/12 ウィンドウ改善**を示しているため、現状のパラメータセットは経験的に有効。
- 重みを正規化して合計=0（定常化）または合計=1（レベル保持）にすると、シグナル特性が大きく変わる可能性がある。変更には追加のウォークフォワード検証が必要。

**推奨**: 現行の `d=0.1` / `window=100` パラメータは変更せず、バイアスを文書化して監視する。将来的に `window` を増やすか、正規化を検討する場合は独立した実験を行う。

### 2.4 その他関数（本番未使用）

| 関数 | 状態 | 備考 |
|---|---|---|
| `adf_test` | 数値安定性改善済 | 入力標準化、非 finite チェック、擬似逆行列使用 |
| `hurst_exponent` | 定数・短系列で NaN を返す | 本番未使用、許容範囲 |
| `find_optimal_d` | 本番未使用 | オフライン実験用 |

---

## 3. 実施した修正

### 3.1 ソースコード

1. **`src/leadlag/features/fractional_diff.py`**
   - `apply_fractional_diff_to_df_exec`: NaN 0 埋め前に警告ログを追加。
   - 誤解を招くコメントを修正。

2. **`src/leadlag/core/pipeline.py`**
   - `build_common_inputs`: NaN 0 埋め前に警告ログを追加。
   - コメントを正確に修正。

### 3.2 テスト

1. **`tests/features/test_fractional_diff.py`**
   - `test_nan_input_is_filled_with_zero_and_warns` 追加
   - `test_window_truncate_bias_documented` 追加

### 3.3 エッジケース調査スクリプト

- `scripts/experiments/audit_fractional_diff_edge_cases.py` 新規作成

---

## 4. 検証結果

| テスト | 結果 |
|---|---|
| `_check_syntax.py` | 14/14 OK |
| `tests/features/test_fractional_diff.py` | 25 passed |
| `tests/integration/test_production_v2.py` | 43 passed |
| `tests/integration/test_sector_relative_ensemble_blp_enhanced.py` | 17 passed |
| `tests/integration/test_production_residual_blpx.py` | 15 passed |

---

## 5. 結論

### 修正が必要だった点

1. **NaN 0 埋めの無警告化** — データ品質問題を隠していた。警告ログを追加して修正済。
2. **誤解を招くコメント** — 修正済。
3. **弱いテスト** — NaN 伝播と重み打ち切りバイアスを検証するテストを追加済。

### 修正しないこととした点

1. **重み打ち切りバイアス (`d=0.1` + `window=100`)** — 理論的には完全な分数階差分ではないが、ウォークフォワードで 12/12 改善を示しており、本番パフォーマンスに好影響を与えている。変更は追加検証を要する。
2. **重み正規化** — シグナル特性を大きく変えるため、本番昇格済の現状を維持。
3. **`hurst_exponent` / `find_optimal_d`** — 本番未使用、現状で許容。

### 推奨アクション

1. 本番ログで `build_common_inputs` / `apply_fractional_diff_to_df_exec` の NaN 警告が出ていないか定期的に確認。
2. 将来の改善実験では、重み正規化（合計=0 または 合計=1）と異なる `window` 値を比較検討。
3. `d=0.1` の本番効果をシャドー／ライブで継続監視。

---

## 6. 参考ファイル

- `src/leadlag/features/fractional_diff.py`
- `src/leadlag/core/pipeline.py`
- `tests/features/test_fractional_diff.py`
- `scripts/experiments/audit_fractional_diff_edge_cases.py`
- `reports/fractional_diff_walkforward_audit_report.md`
- `reports/code_review_fractional_diff_local_changes.md`
