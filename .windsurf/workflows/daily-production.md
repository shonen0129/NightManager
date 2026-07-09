---
description: 日次の本番実行（v2）— gap調整分布の事前計算とデイリー意思決定パイプラインを実行する
---

# 日次本番実行（v2）

ProductionV2Model (Residual-BLPX-RA v2) の日次実行パイプライン。

## 手順

1. gap調整分布の事前計算（v2 の入力）:

```
python3 tools/production/compute_gap_adjusted_distribution.py
```

2. 日次本番実行（v2）:

```
python3 tools/production/run_daily_production_v2.py
```

3. CLI経由のデシジョン（代替）:

```
python3 -m leadlag.cli decision --start-date 2015-01-05
```

4. クローズ処理（必要時）:

```
python3 -m leadlag.cli close
```

## 注意事項

- **ハング既知パターン**: yfinance ダウンロード、`cache.py` の fcntl ファイルロック、`close.py` の auto-close 無限待機、API再試行バックオフに注意。長時間実行はタイムアウト付きで
- `--fast-mode` で事前計算済みキャッシュを使用可能（重い分解をスキップ）
- `--api-enable` で実際の発注が可能（`--api-dry-run` でシミュレーション）
- 詳細は `docs/スタック再発防止策.md` を参照
