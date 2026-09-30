# ML overlay cost sensitivity

判定: **PENDING_REAL_EXECUTION_COSTS**

- 固定OOS期間: 2024-12-23〜2026-07-29 (368日)
- 比較: Stage 5のML有効とStage 6のML無効。ウェイト・gross収益は固定し、slippageだけ片道5/10/20bpsへ再価格付け。
- financing / borrow / reverse は既存モデル値を据え置く。実約定・非約定・ロット丸め・impactは含まない。
- bootstrap: paired日次net差、20日non-circular block、5,000回、seed=20260924。

| 片道slippage | ML on net合計 | ML off net合計 | 差 (on−off) | ML on Sharpe | ML off Sharpe | 差の平均95%区間 | ML on cost | ML off cost |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5bps | 1.9646 | 1.9214 | +0.0432 | 5.164 | 5.180 | +0.000006 .. +0.000230 | 0.9070 | 0.9035 |
| 10bps | 1.1862 | 1.1465 | +0.0397 | 3.115 | 3.089 | -0.000003 .. +0.000220 | 1.6855 | 1.6785 |
| 20bps | -0.3707 | -0.4034 | +0.0327 | -0.971 | -1.084 | -0.000022 .. +0.000201 | 3.2423 | 3.2284 |

同一slippage前提のML損益分岐点: **66.980 bps/片道**

## 判定

この感度分析では価格・費用モデルの仮定を変えた影響を確認できるが、broker約定費用込みのML価値はまだ判定できない。既知368日を新しいforward OOSとは扱わない。現行のML有効版は実費優位が確認されるまで研究上pendingとする。

## 検証

- 5bpsの再構成slippageと保存系列の最大差: 9.910e-17
- ML on gross−cost−netの最大差: 1.110e-16
- ML off gross−cost−netの最大差: 1.041e-16
- 実約定との対応: なし。機能の実費込み採用判定は保留。

日次結果: `daily_cost_sensitivity.csv`; machine-readable summary: `summary.json`。
