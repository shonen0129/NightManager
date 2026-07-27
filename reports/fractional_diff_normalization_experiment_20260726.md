# Fractional Diff 重み打ち切りバイアス修正の性能影響検証レポート

**Date**: 2026-07-26
**Objective**: 重み打ち切りバイアスを修正（重み正規化または窓拡大）した場合の性能変化を検証する。

---

## 1. 方法

### 1.1 修正パターン

| Variant | 内容 |
|---|---|
| `baseline` | 現行: `d=0.1`, `window=100`, 正規化なし |
| `zero` | `normalize='zero'`（重み合計が0になるようシフト） |
| `unit` | `normalize='unit'`（重み合計が1になるようスケール） |
| `window252` | `window=252`（長い履歴を使用） |
| `window504` | `window=504`（さらに長い履歴を使用） |

### 1.2 検証指標

- **Signal Quality**: Rank IC, ICIR, Hit Rate, Long-Short Spread (bps)
- **Simple LS Backtest**: 毎日 top/bottom 銘柄を等权長短、5 bps 片道コスト、日次リバランス
  - 年率リターン (AR %)、Sharpe、最大DD (MDD %)、ターンオーバー

### 1.3 期間

- `df_exec`: 全期間（決定cache 4157 日、2010-2026 想定）
- バックテスト対象期間: 2015-01-05 〜 2024-12-31
- ベースライン期間 (2010-2014) は `build_common_inputs` で必要なため、信号生成は全期間で実施後に評価期間でスライス

### 1.4 使用スクリプト

- `scripts/experiments/experiment_fractional_diff_normalization.py`
- `--n-jobs 4` で並列化

---

## 2. 結果

| Variant | RankIC | ICIR | HitRate | LSSpread_bps | Sharpe | AR (%) | MDD (%) | Turnover |
|---|---|---|---|---|---|---|---|---|
| `baseline` | 0.2244 | 0.7827 | 0.4593 | 59.53 | **7.63** | **108.36** | **-6.76** | 0.826 |
| `zero` | 0.2232 | 0.7798 | 0.4584 | 59.39 | 7.61 | 108.06 | -7.63 | 0.825 |
| `unit` | 0.2244 | 0.7827 | 0.4593 | 59.53 | 7.63 | 108.36 | -6.76 | 0.826 |
| `window252` | 0.2242 | 0.7815 | 0.4594 | 59.44 | 7.61 | 108.13 | -6.76 | 0.827 |
| `window504` | 0.2242 | 0.7817 | 0.4595 | 59.47 | 7.62 | 108.20 | -6.76 | 0.827 |

---

## 3. 解釈

1. **`unit` 正規化は無効**
   - `baseline` と完全に一致。最終的なアンサンブル信号が z-score / cross-sectional 正規化を受けるため、US return 列を同じ係数でスケールしても信号に影響がない。

2. **`zero` 正規化は微減**
   - RankIC / ICIR / HitRate / LSSpread がわずかに低下。
   - Sharpe も微減 (7.63 → 7.61)、MDD が悪化 (-6.76% → -7.63%)。
   - 定数成分を除去すると、微小なレベル情報が失われ、性能がわずかに悪化する。

3. **窓拡大（252, 504）はほぼ無効**
   - ウィンドウを 100 から 252/504 に広げても、信号品質・LS 性能ともにほぼ変わらない。
   - 重み打ち切りバイアスは理論的には存在するが、実際の予測性能に与える影響は限定的。

---

## 4. 結論と推奨

- **重み打ち切りバイアスを修正しても性能は改善しない**。`zero` 正規化ではわずかな劣化が見られる。
- 現行の `d=0.1`, `window=100`, `normalize=None` 設定を維持すべき。
- 将来的に `window` を増やす or 正規化を試す必要がある場合は、本レポートの結果を前提に ROI を見極めること。
- `normalize` パラメータは実験用に `fractional_diff` / パイプラインに追加済み（デフォルト `None`、本番未使用）。

---

## 5. 検証履歴

- 実験スクリプト: `scripts/experiments/experiment_fractional_diff_normalization.py`
- 結果 CSV: `artifacts/fractional_diff_normalization/fractional_diff_normalization_results.csv`
- 変更ファイル:
  - `src/leadlag/features/fractional_diff.py` — `normalize` 引数追加
  - `src/leadlag/core/pipeline.py` — `build_common_inputs` へ伝播
  - `src/leadlag/models/blp_base.py`, `src/leadlag/models/sre.py` — config 読み込み
  - `tests/features/test_fractional_diff.py` — `normalize` 用 unit test 追加
