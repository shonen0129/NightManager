# Phase 4: ML Order Decision Overlay 本番統合レポート

## 1. 目的

Phase 2.8 で採用を承認した `ticker × score` / `ticker × gap` / `ticker × score × gap`
LightGBM オーバーレイを、V2 日次本番パイプラインに安全に統合する。

## 2. 変更内容

### 新規モジュール

- `src/leadlag/models/ml_order_overlay.py`
  - 本番用オーバーレイロジック
  - PIT 特徴量構築（per-ticker 交互作用）
  - LightGBM モデル学習・保存・読み込み
  - V2 結果への適用（スコア再調整 → ウェイト再計算 → 監査再実行）

- `tools/production/train_ml_order_overlay.py`
  - 本番オーバーレイモデルの学習・保存
  - `load_df_exec_from_local_cache()` から履歴を読み込み
  - `models/ml_order_overlay/phase2_8/` に `model.pkl` + `metadata.json` を出力

### 改修

- `tools/production/run_daily_production_v2.py`
  - `ml_order_overlay` 設定が有効な場合、モデルと `df_exec` をロード
  - `generate_v2_production_portfolio_with_overlay()` を呼び出し
  - オーバーレイ無効・モデル欠損・df_exec 欠損時は V2 ベースラインに自動 fallback

- `configs/production/production.yaml`
  - `ml_order_overlay` セクション追加（デフォルト `enabled: false`）

## 3. 学習済みモデル

| 項目 | 値 |
|---|---|
| モデル種別 | LightGBM Regressor |
| 学習期間 | 2020-01-06 → 2022-12-31 |
| 特徴量 | `ticker` categorical + `ticker × score` / `ticker × gap` / `ticker × score × gap` |
| 保存先 | `models/ml_order_overlay/phase2_8/` |
| PIT 情報のみ使用 | Yes（`jp_gap_*`, `topix_night_return`, `jp_beta_*`, 20日 `market_vol_20d` は全て 9:10 以前に確定） |

## 4. 動作確認

### 4.1 dry-run（2023-03-27）

```bash
python3 tools/production/run_daily_production_v2.py \
    --trade-date 2023-03-27 \
    --gap-input-dir results/gap_adjusted_distribution/20260615_004113 \
    --dry-run true
```

結果:

- モデル正常ロード
- `p_trade mean=0.4972`, `std=0.0228`
- PIT bin=Medium, gross=2.0000
- Leakage / Numerical audit: PASSED
- Target Gross / Net: 2.0000 / -0.0000

### 4.2 単体 / 統合テスト

| テスト | 結果 |
|---|---|
| `run_daily_production_v2.py --self-test true` | PASS |
| `pytest tests/integration/test_production_v2.py -q` | 45 passed |
| `pytest tests/integration/test_production_residual_blpx.py -q` | 15 passed |

## 5. 本番運用手順

### 有効化

`configs/production/production.yaml` を編集:

```yaml
ml_order_overlay:
  enabled: true
  model_dir: models/ml_order_overlay/phase2_8
```

### モデル再学習

新しい歴史データが溜まったら:

```bash
python3 tools/production/train_ml_order_overlay.py \
    --train-start 2020-01-06 \
    --train-end <最新日> \
    --gap-input-dir results/gap_adjusted_distribution/latest \
    --output-dir models/ml_order_overlay/phase2_8
```

### 日次実行

既存の `run_daily_production_v2.py` と同じ。オーバーレイは config 設定に応じて自動適用:

```bash
python3 tools/production/run_daily_production_v2.py \
    --trade-date latest \
    --gap-input-dir results/gap_adjusted_distribution/latest
```

## 6. フォールバック動作

| 状況 | 動作 |
|---|---|
| `ml_order_overlay.enabled: false` | V2 ベースラインのまま |
| モデルファイル不在 | 警告ログ + V2 ベースライン |
| `df_exec` 不在 | 警告ログ + V2 ベースライン |
| gap 行列欠損（V2 fallback） | V2 フラットポジション (`w_final=0`) |
| オーバーレイ後 numerical audit FAILED | オーバーレイ適用前の V2 結果に戻す |

## 7. 既知の制限

- 本番モデルは 2020-2022 年で学習済み。2025 年以降の市場環境変化に伴う再学習は監視が必要。
- 日次 `df_exec` は `market_data/decision_cache.npz` または `market_data/etf_data.pkl` から読み込む。9:10 執行前に最新 `jp_gap_*` / `topix_night_return` がキャッシュされていることを確認。
- シャドー運用期間中は、オーバーレイ適用前後の IR / ウェイト差分を日次ログ (`p_trade_mean`, `p_trade_std`) で監視すること。

## 8. 結論

Phase 2.8 の ML 発注判定オーバーレイを `production_v2.py` 本番パイプラインに統合した。
デフォルトでは無効であり、本番管理者が `production.yaml` の `enabled: true` にすることで、
学習済みモデルを用いたスコア調整が有効になる。
全フォールバック条件が整備され、既存の V2 監査も通過する。

次のステップは **シャドー運用**（`tools/validation/monitor_residual_blpx_shadow_performance.py`）に進む。
