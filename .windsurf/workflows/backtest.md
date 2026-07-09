---
description: 本番configでバックテストを実行し、net Sharpe・最大DD・ターンオーバー等の指標を確認する
---

# バックテスト実行

本番モデル（ProductionV2Model / Residual-BLPX）のバックテストを実行する。

## 手順

1. 本番バックテストを実行:

```
python3 src/research/scripts/backtest/run_production_backtest.py --config configs/production/production.yaml --start-date 2015-01-05
```

2. 結果の確認項目:
   - **net Sharpe**（主指標・コスト後）
   - **最大DD**
   - **ターンオーバー**
   - **フォールバック発動率**
   - gross/net 両方のリターン
   - コスト内訳（slippage / financing / borrow / reverse）

3. スリッページ感度分析（オプション）:

```
python3 src/research/scripts/backtest/run_production_backtest.py --config configs/production/production.yaml --start-date 2015-01-05 --slippage-bps 3
python3 src/research/scripts/backtest/run_production_backtest.py --config configs/production/production.yaml --start-date 2015-01-05 --slippage-bps 7
```

4. CLI経由のバックテスト（代替）:

```
python3 -m leadlag.cli backtest --start-date 2015-01-05
```

## 注意事項

- `start_date` は **2015-01-05 以降**を維持（ベースライン期間 2010-2014 の分離）
- コストは片道5bps + 金利・貸株・逆日歩を含む **net** で評価
- 長時間実行の可能性があるため、タイムアウト付きで実行すること
- 結果は `reports/<sprint名>/` に markdown で記録
