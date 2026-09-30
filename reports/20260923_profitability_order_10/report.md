# 収益改善順序10：期待利益対注文コストの一次検証

実行日時: 2026-09-23T07:27:58.707908+00:00。発注・本番設定・モデルartifactの変更は行っていない。

## 仮説と固定設計

V2の9:10時点予測 `mu_gap` を使い、前日の選択ウェイトから当日のcanonical目標ウェイトへ動かす各銘柄の予測増分利益が、推定注文コストを上回る場合だけその方向の注文を許す。注文サイズは0〜目標差分の範囲で選び、市場中立・model gross≤2.0を保つ。
- 評価期間: 2024-12-23〜2026-07-29 (368営業日)。既存stage5と同じ既知OOSであり、未使用の独立holdoutではない。
- 対象銘柄: TOPIX-17。設定: `configs/production/production.yaml`。gap入力: `var/live/pipeline_data/gap_adjusted_distribution/20260731_024303`。ML overlay: `models/ml_order_overlay/production_20260923`。
- 実行コマンド: `.venv/bin/python reports/20260912_workspace_audit/watchdog.py 1800 .venv/bin/python src/research/scripts/experiments/experiment_order_cost_gate_20260923.py > /tmp/20260923_order_cost_gate.log 2>&1`。中心scenarioは slippage=12.5bps/side, financing=2.50%/年, borrow=1.15%/年, reverse=2.0bps/日, overnight alpha long/short=0.75/0.50, side leverage=1.50。
- 予測利益: `side_leverage × Δweight_i × mu_gap_i`。実現リターンはゲートに使わない。
- 注文費用代理: 0.25%をfull spread幅と見て12.5bps/sideを中心に設定。日中取引のentryとclose部分、増分carryをコスト比較に含む。±20%感度は10/15bps/side。
- 約定損益は各費用シナリオと同じ片道slippageを共有PnL計算へ渡し、financing/borrow/reverseも同じ解決済み設定で計上。
- 部分約定、板厚、価格インパクト、ticker別spread、lot丸めは未モデル化。prev weightは実在庫ではない。
- 注文に紐づくbroker手数料・明示feesも、過去の約定連結データがないため未計上。

## 正本baselineの再現

- canonical weights最大差: 9.714e-17。
- 5bps/side netリターン最大差: 9.986e-17。
- 予測は各日のV2 `mu_gap` から回収し、全日についてbaselineの予測ポートフォリオ平均と照合した。

## 結果（同じ費用前提のbaselineとの比較）

| gate cost (bps/side) | variant | net Sharpe | gross Sharpe | max DD | mean turnover | cost sum | mean net Δ/day | block-bootstrap 95% CI |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 10.0 | baseline | 3.1152 | 7.5529 | -36.81% | 1.2871 | 1.685455 | — | — |
| 10.0 | gated | 0.0483 | 3.5831 | -35.39% | 0.5172 | 1.286427 | -0.0031761 | [-0.0054734, -0.0012554] |
| 12.5 | baseline | 2.0919 | 7.5529 | -42.94% | 1.2871 | 2.074668 | — | — |
| 12.5 | gated | -0.8783 | 3.4405 | -41.64% | 0.4905 | 1.558780 | -0.0030195 | [-0.0052899, -0.0010722] |
| 15.0 | baseline | 1.0695 | 7.5529 | -48.48% | 1.2871 | 2.463881 | — | — |
| 15.0 | gated | -1.8349 | 3.2799 | -50.68% | 0.4607 | 1.816366 | -0.0028636 | [-0.0050726, -0.0008206] |

中心シナリオの注文診断: 閾値を超えた銘柄注文 2534件、実際に動かした銘柄注文 1730件、変更があった日 292 / 368日。モデル制約の最大違反量は net 4.580e-16, gross超過 8.882e-16。side leverage=1.50適用後の平均実効grossはbaseline 2.629、gated 2.904、最大3.00。turnoverが下がってもgated側は平均grossとfinancing/borrow/reverse費が増えた。

## コスト内訳（中心シナリオ）

| 系列 | baseline | gated | 差分 |
|---|---:|---:|---:|
| slip | 1.946066 | 1.416796 | -0.529270 |
| financing | 0.039529 | 0.043643 | +0.004113 |
| borrow | 0.012122 | 0.013384 | +0.001261 |
| reverse | 0.076950 | 0.084957 | +0.008007 |

## 統計評価と判定

- 20営業日paired block bootstrap: 1,000回、seed=42。区間は日次平均net差の不確実性で、Sharpe差の有意性を意味しない。
- 名目DSR (中心variant; 同一検証内の3コスト水準×baseline/gatedの6候補): 4.869651867086058e-05。既存の関連no-trade試行とこのOOSの事前閲覧分を完全に網羅した総試行数ではないため、採否を確定する証拠としては使わない。
- 事前ゲート: 中心12.5bpsでnet Sharpeが同費用baseline以上、max DDが悪化せず、turnoverとcost sumが減ること。結果: `False`。
- 研究判定: **rejected**。中心シナリオが事前のSharpe/DD/turnover/cost条件を満たさなかった。
- 0.25% spread換算は約定実績ではない。現行データでは真の9:10板・fill価格が不足しているため、仮に数値ゲートを通過しても本番採用は保留。

## 再現成果物

- script: `src/research/scripts/experiments/experiment_order_cost_gate_20260923.py`
- results: `reports/20260923_profitability_order_10/results.json`
- candidate weights: `reports/20260923_profitability_order_10/daily_gate_weights.csv`
- per-day gate diagnostics: `reports/20260923_profitability_order_10/daily_gate_decisions.csv`
- registry: `var/experiments/registry.jsonl`
