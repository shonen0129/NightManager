# US セクター特徴量追加実験 — 採用判断（2026-08-01）

## 目的
Residual-BLPX-RA v2 の ML order overlay に、米国セクター close-to-close リターン由来の特徴量を追加して性能を改善できるか検証する。

## 検証した特徴量セット

### 1. Basic US 特徴量
- `us_mapped`: 各 JP 銘柄にベースライン相関で最も絶対値の高い US セクター cc return
- `us_avg` / `us_dispersion`: 全 US セクターの平均・標準偏差
- `topix_night_minus_us_avg`
- `score_x_us_mapped` / `gap_x_us_mapped` 等の交互作用

### 2. Refined US 特徴量
- `us_mapped_zscore` / `us_mapped_rank`: 当日 US 横断面内での z-score / ランク
- `us_idio` = `us_mapped - us_avg`
- `us_top_minus_bottom`: US セクター幅
- `us_jp_corr_60d`: 過去 60 日の mapped US ↔ JP cc ラグ付き相関
- 上記と score / gap / gap_idio の交互作用

## ウォークフォワード結果（2022-2024、3-fold OOS）

| モデル | Pooled OOS Sharpe | Δ vs Base | 勝率 | DSR |
|---|---|---|---|---|
| Baseline | 6.6387 | — | — | — |
| per-ticker only | 6.6992 | +0.0605 | 2/3 | 1.0000 |
| per-ticker + basic US | 6.7043 | +0.0656 | 2/3 | 1.0000 |
| per-ticker + refined US | 6.6904 | +0.0517 | 2/3 | 1.0000 |

## 判断

- **US セクター特徴量の純粋な追加効果は限定的**（basic +0.005、refined は per-ticker only より低下）。
- すべての改善は `per-ticker` 交互作用（ticker × score / gap / score×gap）が主体。
- refined US は特徴量数を 92 → 115 に増やすも性能低下。過学習 / ノイズ増加を示唆。
- ルックアヘッドはなし（相関はラグ、US リターンは JP 寄付前確定）。

## 採用判断

**US セクター close-to-close 特徴量（basic / refined いずれも）本番採用見送り。**

既存の `ml_order_overlay`（`models/ml_order_overlay/phase2_8`、per-ticker 交互作用のみ）を本番継続。

## 本番状態確認

- `configs/production/production.yaml`:
  ```yaml
  ml_order_overlay:
    enabled: true
    model_dir: models/ml_order_overlay/phase2_8
    per_ticker_interactions: true
  ```
- 2024-12-30 dry-run: 正常終了、overlay 適用確認（p_trade mean=0.4913, std=0.0074）
- Self-test: PASSED

## 次のステップ

1. シャドー運用（`tools/validation/monitor_residual_blpx_shadow_performance.py`）で per-ticker overlay の live 整合を継続監視。
2. US 連動度を別のデータソース（ADR、為替、金利、VIX）で再検討する場合は、改めて実験設計・WF 検証を実施。
