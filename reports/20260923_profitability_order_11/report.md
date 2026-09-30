# 収益改善順序11：LightGBM期待リターンと注文別コスト判断

実行日時: 2026-09-23T08:51:33.156042+00:00。発注・本番設定・モデルartifactの変更は行っていない。

## 仮説と固定設計

LightGBMの回帰出力から学習時に控除した固定往復コストを戻して銘柄別の期待グロスリターンを得る。前日の選択ウェイトから当日のcanonical目標へ動かす注文について、期待増分利益が注文コストを上回る場合だけLPで部分約定相当のサイズを選ぶ。市場中立・model gross≤2.0を保つ。
- 評価期間: 2024-12-23〜2026-07-29 (368営業日)。既存stage5と同じ既知OOSであり、未使用の独立holdoutではない。
- LGBM artifact `20260922T011723828589Z-7e1a81ab2617` 学習期間: 2015-01-05〜2024-12-20。過去データの実取得時刻証拠: `False`。
- 対象銘柄: TOPIX-17。設定: `configs/production/production.yaml`。gap入力: `var/live/pipeline_data/gap_adjusted_distribution/20260731_024303`。ML overlay: `models/ml_order_overlay/production_20260923`。
- 実行コマンド: `.venv/bin/python reports/20260912_workspace_audit/watchdog.py 1800 .venv/bin/python src/research/scripts/experiments/experiment_lgbm_order_cost_20260923.py > /tmp/20260923_lgbm_order_cost.log 2>&1`。中心scenarioの解決済み設定: slippage=12.5bps/side, financing=2.50%/年, borrow=1.15%/年, reverse=2.0bps/日, overnight alpha long/short=0.75/0.50, side leverage=1.50。
- 学習targetは `sign(score) × 実現9:10→大引けリターン − 0.1000%`。回帰予測へ固定控除額を加え、signal方向を戻して生リターン単位へ変換。
- LGBM予測診断: 6256銘柄日、MAE=0.00871、相関=0.1549、方向一致率=55.20%、予測平均=-0.00017、実現平均=-0.00047。
- 注文費用は、注文ウェイト×実効レバレッジ×予想売買側数に対し、full spreadの半分/片道、明示手数料、impactを別項目で計算。carryは判断費用に重ねず、PnLのfinancing/borrow/reverseで別計上。
- 中心はfull spread 0.25%（12.5bps/side）、±20%感度は0.20%/0.30%。過去の真の9:10気配値がないため全銘柄共通の仮定。手数料・impactはデータ不足で0仮置き。
- Backtest損益には同じシナリオのslippageと解決済みのfinancing/borrow/reverseを計上。previous weightは実在庫ではない。
- 部分約定、板厚、ticker別spread、impact、lot丸めは未モデル化。費用関数はticker別入力を受けられるが今回の過去系列に有効な気配値はない。

## 正本baselineの再現

- canonical weights最大差: 9.714e-17。
- 5bps/side netリターン最大差: 9.986e-17。
- canonicalウェイトと5bps/sideのnet損益をStage 5成果物へ照合。LGBMのraw回帰出力は既存artifactから直接再計算した。

## 結果（同じ費用前提のbaselineとの比較）

| gate cost (bps/side) | variant | net Sharpe | gross Sharpe | max DD | mean turnover | cost sum | mean net Δ/day | block-bootstrap 95% CI |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 10.0 | baseline | 3.1152 | 7.5529 | -36.81% | 1.2871 | 1.685455 | — | — |
| 10.0 | gated | 1.4687 | 4.9469 | -34.54% | 0.4935 | 1.306098 | -0.0017313 | [-0.0029406, -0.0005827] |
| 12.5 | baseline | 2.0919 | 7.5529 | -42.94% | 1.2871 | 2.074668 | — | — |
| 12.5 | gated | -0.0248 | 4.0952 | -39.95% | 0.3925 | 1.510024 | -0.0021903 | [-0.0036056, -0.0008133] |
| 15.0 | baseline | 1.0695 | 7.5529 | -48.48% | 1.2871 | 2.463881 | — | — |
| 15.0 | gated | -0.9371 | 4.1047 | -47.28% | 0.3088 | 1.690415 | -0.0019556 | [-0.0036273, -0.0004095] |

中心シナリオの注文診断: 期待利益が推定コストを超えた銘柄注文 2231件、実際に動かした銘柄注文 1466件、変更があった日 319 / 368日。部分注文を含む推定実行費用合計 0.719085。モデル制約の最大違反量は net 5.829e-16, gross超過 4.441e-16。side leverage=1.50適用後の平均実効grossはbaseline 2.629、gated 2.989、最大 3.00。

## コスト内訳（中心シナリオ）

| 系列 | baseline | gated | 差分 |
|---|---:|---:|---:|
| slip | 1.946066 | 1.364036 | -0.582031 |
| financing | 0.039529 | 0.044874 | +0.005344 |
| borrow | 0.012122 | 0.013761 | +0.001639 |
| reverse | 0.076950 | 0.087354 | +0.010404 |

## 統計評価と判定

- 20営業日paired block bootstrap: 1,000回、seed=42。区間は日次平均net差の不確実性で、Sharpe差の有意性を意味しない。
- 名目DSR (中心variant; 同一検証内の3コスト水準×baseline/gatedの6候補): 0.010450723120571683。実行前のExperimentRegistryには32件あるが、未登録試行とOOSの事前閲覧分を完全には数えられないため、採否確定の証拠にしない。
- 事前ゲート: 中心12.5bpsでnet Sharpeが同費用baseline以上、max DDが悪化せず、turnoverとcost sumが減ること。結果: `False`。
- 研究判定: **rejected**。中心シナリオが事前のSharpe/DD/turnover/cost条件を満たさなかった。
- 0.25% spread換算は約定実績ではない。現行データでは真の9:10板・fill価格が不足しているため、仮に数値ゲートを通過しても本番採用は保留。

## 再現成果物

- script: `src/research/scripts/experiments/experiment_lgbm_order_cost_20260923.py`
- results: `reports/20260923_profitability_order_11/results.json`
- candidate weights: `reports/20260923_profitability_order_11/daily_gate_weights.csv`
- per-day gate diagnostics: `reports/20260923_profitability_order_11/daily_gate_decisions.csv`
- per-order estimated costs and decisions: `reports/20260923_profitability_order_11/daily_orders.csv`
- registry: `var/experiments/registry.jsonl`
