# Phase 19: BLPX `mu_gap` 改善レポート — `vol_adjusted_target: false`

## 目的

BLPX の `mu_gap` 生成精度を改善する。

## 変更内容

1. `src/leadlag/models/sector_relative_ensemble_blp_enhanced.py`  
   `_build_blp_diagnostics` の `mu`/`sigma` X/Y 分割を、誤っていた `len(...)//2` から `len(US_TICKERS)`/`len(JP_TICKERS)` に修正。`vol_adjusted_target=false` パスを正しく機能させるため。

2. `configs/production/production.yaml`  
   `blpx.vol_adjusted_target: false` を追加。

3. `docs/ARCHITECTURE.md`  
   Phase 19 として変更履歴を追記。

## 変更の意味

- `vol_adjusted_target=true`（旧デフォルト）: `mu_raw = z_hat_j * sigma_j_t`（0 平均、直近 20 日実現ボラスケーリング）
- `vol_adjusted_target=false`（新設定）: `mu_raw = mu_Y + sigma_Y * z_hat_j`（インサンプル平均 + インサンプル標準偏差で標準化予測を復元）

## バックテスト結果（2020-01-01 〜 latest, 1544 日）

| metric | baseline (`vol_adjusted_target=true`) | `vol_adjusted_target=false` | 差分 |
|---|---|---|---|
| net total | 768.90% | **1006.06%** | +237.16% |
| net Sharpe | 6.0571 | **7.3882** | +1.3311 |
| max DD | -8.39% | **-6.99%** | +1.40% |
| turnover | 1.3725 | **1.3082** | -0.0643 |
| fallback rate | 11.33% | **3.63%** | -7.70% |

## 2024 年サブ検証

| metric | baseline | `vol_adjusted_target=false` |
|---|---|---|
| net total | 130.82% | **148.51%** |
| net Sharpe | 6.5033 | **7.2723** |
| max DD | -8.39% | **-7.06%** |
| turnover | 1.5283 | 1.5261 |

## テスト

- `python3 _check_syntax.py`: 43/43 OK
- `python3 -m pytest tests/unit/test_vol_adjustment.py tests/unit/test_gap_distribution.py`: 20 passed
- `python3 -m pytest tests/integration/test_leakage_audit.py tests/integration/test_production_residual_blpx.py`: 39 passed
- `scripts/run_tests_parallel.sh` + p7 再実行: 全 403 テスト PASS（1 件は `test_ml_order_overlay.py` のヘルパーで `w_final=0` だったものを修正）

## 残タスク

- 本番 `live/pipeline_data/gap_adjusted_distribution/latest` の gap 行列は `vol_adjusted_target=true` の古い行列のため、次回 `compute_gap_adjusted_distribution.py` または `run_gap_distribution.sh` 実行時に再計算が必要。
- 既存の ML Order Overlay 改修（未コミット、`src/leadlag/execution/backtester.py` 他）は本 `mu_gap` 改善とは別件。テストは通る状態で残している。

## 実行したコマンド

```bash
python3 tools/production/compute_gap_adjusted_distribution.py \
  --config configs/experiments/vol_adjusted_false.yaml \
  --output-dir outputs/gap_adjusted_vol_false \
  --start 2020-01-01 --end latest --n-jobs 1

python3 scripts/experiments/compare_vol_adjusted_false_full.py
```
