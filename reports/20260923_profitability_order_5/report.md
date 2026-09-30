# 収益改善順序5：日中と持越しの分解

発注・取消・再送を行わない、canonical V2 backtestの診断です。

判定: **PENDING_STAGE_5_COMPLETION**

## 固定条件

- 期間: 2024-12-23 -> 2026-07-29 (368日)
- OOS境界: 2024-12-23 (after overlay training end 2024-12-20)
- config: configs/production/production.yaml
- active overlay: models/ml_order_overlay/production_20260923
- gap input: var/live/pipeline_data/gap_adjusted_distribution/20260731_024303
- 年率換算: 252営業日
- 評価は全368日（fallback日を除外しない）
- 今回のdf_exec再構築では前回出力にあった2025-10-28がなく、前回369日から1日減少。推測補完せず、順序6-8も今回の同一日付集合で再計算する。
- 急変日は実現gross絶対値のOOS上位5%で分類する事後診断であり、予測時点の特徴量ではない

## 帰属結果

| 系列 | 年率リターン | 年率vol | 年率Sharpe | 合計 |
|---|---:|---:|---:|---:|
| net | 1.345329 | 0.260539 | 5.163636078955636 | 1.964607 |
| intraday gross | 1.952173 | 0.218309 | 8.942232092464707 | 2.850793 |
| overnight carry | 0.014272 | 0.113907 | 0.12529906852684516 | 0.020842 |
| cost (negative) | -0.621117 | 0.007536 | -82.41584928105914 | -0.907028 |
| no-carry net counterfactual | 1.289646 | 0.217458 | 5.930543802393315 | 1.883293 |

コスト累計（return単位）: slip=0.778427, financing=0.039529, borrow=0.012122, reverse=0.076950

## 会計整合性

- cost内訳最大誤差: 1.960e-16
- PnL帰属最大誤差: 1.110e-16
- shared PnL replay net最大誤差: 9.986e-17
- shared PnL replay overnight最大誤差: 9.931e-17
- shared PnL replay cost最大誤差: 9.888e-17

## 週末・連休・急変時

{
  "session_gap_groups": {
    "next_session_1d": {
      "n": 270,
      "intraday_sum": 2.1131915951626383,
      "overnight_sum": 0.10867388663415502,
      "cost_sum": 0.6284119789925064,
      "net_sum": 1.5934535028042747,
      "mean_net": 0.005901679640015832,
      "worst_net": -0.0394613012322883
    },
    "weekend_or_holiday": {
      "n": 97,
      "intraday_sum": 0.7128843232256338,
      "overnight_sum": -0.0878315738019865,
      "cost_sum": 0.27608243254495357,
      "net_sum": 0.34897031687868996,
      "mean_net": 0.0035976321327699998,
      "worst_net": -0.0239782425193392
    },
    "terminal": {
      "n": 1,
      "intraday_sum": 0.0247166996665438,
      "overnight_sum": 0.0,
      "cost_sum": 0.0025335447965369,
      "net_sum": 0.0221831548700068,
      "mean_net": 0.0221831548700068,
      "worst_net": 0.0221831548700068
    }
  },
  "realized_stress_group": {
    "other": {
      "n": 349,
      "intraday_sum": 2.126423123622457,
      "overnight_sum": -0.14405531795560145,
      "cost_sum": 0.8606189098301273,
      "net_sum": 1.1217488958367132,
      "mean_net": 0.0032141802172971725,
      "worst_net": -0.0335531715933555
    },
    "top_5pct_abs_gross": {
      "n": 19,
      "intraday_sum": 0.7243694944323589,
      "overnight_sum": 0.16489763078777003,
      "cost_sum": 0.0464090465038696,
      "net_sum": 0.8428580787162582,
      "mean_net": 0.04436095151138201,
      "worst_net": -0.0394613012322883
    }
  }
}

## 完了条件と採否

| 条件 | 結果 |
|---|---|
| 日中/夜間/費用のPnL帰属 | PASS（出力系列とshared PnL replayで整合） |
| 週末・連休の持越し分解 | PASS（calendar day gap別に集計） |
| 急変時の損失分解 | DIAGNOSTIC（実現値による事後分類） |
| 真の9:10価格・実約定との整合 | 未証明（順序2を保留したまま） |
| carry方針の本番変更 | 実施なし |

順序5の診断計算は完了したが、実行可能価格・実約定・費用を使った正式なbaseline再評価が未完了のため、順序5の経済的な採否は保留する。ユーザー依頼に基づき順序6-8の既存診断は今回の368日データで再計算したが、順序2の実約定証拠や順序3の実行可能baselineの代替にはならず、本番採用判断は引き続き保留する。

ExperimentRegistry: stage5_intraday_carry_decomposition_20260923, decision=pending, trials=3

再現JSON: reports/20260923_profitability_order_5/stage5_summary.json
日次分解: reports/20260923_profitability_order_5/daily_decomposition.csv
