# 銘柄別オーバーナイト在庫 sizing：V2単独の長期比較

## 仮説・固定条件

2026-10-08時点で固定した252営業日adaptive carry policyを、V2単独の長期ウェイト列でfixed alphaと比較する。期間を広げるだけの再検証とし、policy選択・閾値調整は行わない。

- 生成・評価対象: 2015-01-05–2026-09-25（2777営業日）。評価: 2016-02-02–2026-09-25（2524営業日、252遷移warm-up）。
- V2単独: resolved production `ml_overlay_enabled=False`、overlay model dir empty。V2 audit status counts={'PASSED|PASSED': 2777}。
- df_exec source=2009-01-07–2026-09-25。最新market row 2026-10-09 はmacro snapshot期限に合わせて除外。
- macro snapshot=2009-01-07–2026-09-25、source SHA256=e4a0c2126eba507f63493b4104b81ede305bab8c5e74678213b9051caf24b7c6。provenanceは `historical_provider_available_at_proven=False`。各決定は当該日closeより前のmacro行だけを使う。
- gap storeは9日（2026-08-07–2026-09-25）。その他の期間はresolved V2 on-demand設定で生成。評価fallback率=0.00%。
- fixed carry alpha=0.75/0.50、side leverage=1.30、slippage=5.0bp/side、financing=2.50%/年、borrow=1.15%/年、reverse=2.0bp/暦日。
- inventoryはflat start・連続状態・最終引け全清算。execution volumeはopening/closing inventory flow合計、turnoverはその1/2。gross/net PnL、MDD、Sharpeは全評価日を含む。

## 全期間比較

PnL列は日次return fractionの合計/複利値。

| Policy | n | Net Sharpe | MDD | Gross sum / comp. | Net sum / comp. | Slip | Carry | Turnover/day | Reuse | Fallback | Carry net exp. | Carry gross exp. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_alpha_0.75_0.50 | 2524 | 6.952 | -31.52% | 18.8499 / 12099408472.72% | 13.1085 / 40273441.51% | 4.9532 | 0.7882 | 1.9624 | 30.47% | 0.00% | 0.303 | 1.514 |
| adaptive_252 | 2524 | 6.576 | -22.41% | 18.8047 / 11383426660.54% | 12.9647 / 34334601.90% | 5.6729 | 0.1671 | 2.2476 | 27.08% | 0.00% | 0.643 | 0.650 |

### 年別評価（2026年はYTD）

| 年/期間 | Policy | n | Net Sharpe | MDD | Net sum | Slip | Carry | Turnover/day | Reuse | Carry net exp. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2016 | fixed_alpha_0.75_0.50 | 218 | 14.520 | -1.26% | 3.1077 | 0.4733 | 0.0732 | 2.1712 | 26.38% | 0.325 |
| 2016 | adaptive_252 | 218 | 14.114 | -1.91% | 2.8718 | 0.5370 | 0.0148 | 2.4632 | 20.65% | 0.659 |
| 2017 | fixed_alpha_0.75_0.50 | 244 | 8.895 | -4.42% | 1.1113 | 0.5202 | 0.0786 | 2.1321 | 28.77% | 0.325 |
| 2017 | adaptive_252 | 244 | 7.679 | -7.08% | 1.1519 | 0.5780 | 0.0224 | 2.3688 | 24.80% | 0.930 |
| 2018 | fixed_alpha_0.75_0.50 | 249 | 5.332 | -8.71% | 0.8013 | 0.5271 | 0.0800 | 2.1170 | 29.64% | 0.325 |
| 2018 | adaptive_252 | 249 | 3.350 | -10.33% | 0.4720 | 0.6215 | 0.0100 | 2.4961 | 25.30% | 0.402 |
| 2019 | fixed_alpha_0.75_0.50 | 233 | 6.298 | -4.07% | 0.8475 | 0.5012 | 0.0797 | 2.1509 | 27.63% | 0.325 |
| 2019 | adaptive_252 | 233 | 4.175 | -11.15% | 0.5023 | 0.5943 | 0.0040 | 2.5508 | 29.22% | 0.132 |
| 2020 | fixed_alpha_0.75_0.50 | 234 | 8.277 | -4.19% | 1.7556 | 0.4932 | 0.0791 | 2.1078 | 30.22% | 0.325 |
| 2020 | adaptive_252 | 234 | 8.705 | -7.62% | 1.6318 | 0.5858 | 0.0091 | 2.5035 | 26.59% | 0.333 |
| 2021 | fixed_alpha_0.75_0.50 | 237 | 6.274 | -5.03% | 0.9306 | 0.4306 | 0.0690 | 1.8168 | 31.35% | 0.282 |
| 2021 | adaptive_252 | 237 | 6.437 | -6.10% | 1.1323 | 0.4704 | 0.0242 | 1.9848 | 27.70% | 0.985 |
| 2022 | fixed_alpha_0.75_0.50 | 235 | 7.978 | -3.60% | 1.1726 | 0.4225 | 0.0681 | 1.7980 | 31.85% | 0.281 |
| 2022 | adaptive_252 | 235 | 7.679 | -3.76% | 1.2026 | 0.4823 | 0.0159 | 2.0521 | 28.15% | 0.689 |
| 2023 | fixed_alpha_0.75_0.50 | 238 | 4.285 | -4.44% | 0.4938 | 0.4270 | 0.0691 | 1.7941 | 33.14% | 0.283 |
| 2023 | adaptive_252 | 238 | 4.428 | -4.77% | 0.5258 | 0.5002 | 0.0110 | 2.1015 | 33.08% | 0.484 |
| 2024 | fixed_alpha_0.75_0.50 | 236 | 6.848 | -6.33% | 1.0564 | 0.4269 | 0.0716 | 1.8091 | 33.70% | 0.286 |
| 2024 | adaptive_252 | 236 | 6.253 | -8.91% | 1.1826 | 0.4790 | 0.0207 | 2.0297 | 30.86% | 0.848 |
| 2025 | fixed_alpha_0.75_0.50 | 230 | 7.321 | -3.77% | 1.3293 | 0.4080 | 0.0674 | 1.7740 | 31.92% | 0.277 |
| 2025 | adaptive_252 | 230 | 7.939 | -6.49% | 1.4924 | 0.4664 | 0.0169 | 2.0279 | 26.94% | 0.693 |
| 2026 | fixed_alpha_0.75_0.50 | 170 | 2.763 | -31.52% | 0.5024 | 0.3231 | 0.0522 | 1.9004 | 31.84% | 0.294 |
| 2026 | adaptive_252 | 170 | 3.650 | -22.41% | 0.7992 | 0.3581 | 0.0182 | 2.1063 | 25.88% | 1.020 |

### 固定候補の感度確認

| Policy | Net Sharpe | MDD | Net sum | Slip | Carry | Reuse | Long/short alpha |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_alpha_0.75_0.50 | 6.952 | -31.52% | 13.1085 | 4.9532 | 0.7882 | 30.47% | 0.750/0.500 |
| adaptive_252 | 6.576 | -22.41% | 12.9647 | 5.6729 | 0.1671 | 27.08% | 0.534/0.003 |
| adaptive_lookback_202 | 6.517 | -22.29% | 12.8664 | 5.6776 | 0.1670 | 27.16% | 0.523/0.006 |
| adaptive_lookback_302 | 6.455 | -23.38% | 12.9025 | 5.6612 | 0.1695 | 27.36% | 0.543/0.003 |
| adaptive_reversal_cost_0.8x | 6.577 | -22.45% | 13.0241 | 5.6620 | 0.1719 | 27.03% | 0.547/0.004 |
| adaptive_reversal_cost_1.2x | 6.575 | -22.37% | 12.9046 | 5.6841 | 0.1623 | 27.13% | 0.520/0.003 |

### 連休・翌朝反転

| Policy | Segment | n | Net sum | Mean net | Overnight | Slip | Next-open slip delta vs flat | Carry | Reuse |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_alpha_0.75_0.50 | calendar_gap_1d | 1873 | 9.5600 | 0.00510 | 1.73600 | 3.67518 | 0.56023 | 0.37917 | 30.25% |
| fixed_alpha_0.75_0.50 | weekend_or_short_holiday_2_3d | 535 | 3.1073 | 0.00581 | 0.27714 | 1.04942 | 0.15246 | 0.30704 | 31.20% |
| fixed_alpha_0.75_0.50 | extended_holiday_4d_plus | 115 | 0.4382 | 0.00381 | 0.02256 | 0.22546 | 0.03383 | 0.10195 | 30.63% |
| fixed_alpha_0.75_0.50 | terminal_final_close | 1 | 0.0031 | 0.00305 | 0.00000 | 0.00316 | 0.00000 | 0.00000 | 0.00% |
| fixed_alpha_0.75_0.50 | next_signal_reversal | 6632 | 8.1956 | 0.00124 | 7.09979 | 1.13853 | 0.49471 | 0.20154 | 0.00% |
| fixed_alpha_0.75_0.50 | next_signal_same_direction | 8463 | 0.8442 | 0.00010 | -6.04526 | 1.39280 | -0.49972 | 0.27470 | 87.60% |
| fixed_alpha_0.75_0.50 | next_signal_flat | 10135 | 4.8171 | 0.00048 | 0.98117 | 1.66727 | 0.75152 | 0.31193 | 0.00% |
| adaptive_252 | calendar_gap_1d | 1873 | 9.0633 | 0.00484 | 1.45500 | 4.18177 | 0.29119 | 0.08824 | 27.01% |
| adaptive_252 | weekend_or_short_holiday_2_3d | 535 | 3.2740 | 0.00612 | 0.36923 | 1.22137 | 0.07134 | 0.06048 | 27.16% |
| adaptive_252 | extended_holiday_4d_plus | 115 | 0.6241 | 0.00543 | 0.16625 | 0.26686 | 0.01363 | 0.01836 | 28.10% |
| adaptive_252 | terminal_final_close | 1 | 0.0033 | 0.00326 | 0.00000 | 0.00295 | 0.00000 | 0.00000 | 0.00% |
| adaptive_252 | next_signal_reversal | 6632 | 4.2039 | 0.00063 | 3.20151 | 1.38844 | 0.22284 | 0.04506 | 0.00% |
| adaptive_252 | next_signal_same_direction | 8463 | 4.5712 | 0.00054 | -2.08198 | 1.84884 | -0.17241 | 0.05491 | 81.69% |
| adaptive_252 | next_signal_flat | 10135 | 4.5120 | 0.00045 | 0.87095 | 2.10703 | 0.32573 | 0.06711 | 0.00% |

### Borrow stressと不確実性

Borrow stressはuniformな仮定であり、tickerごとの実費ではない。

| Borrow annual | Policy | Net Sharpe | MDD | Net sum | Borrow | Reverse | Mean short alpha |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10% | fixed_alpha_0.75_0.50 | 6.650 | -32.59% | 12.5368 | 0.64604 | 0.47161 | 0.500 |
| 10% | adaptive_252 | 6.573 | -22.41% | 12.9570 | 0.00050 | 0.00036 | 0.001 |
| 30% | fixed_alpha_0.75_0.50 | 5.963 | -34.96% | 11.2447 | 1.93811 | 0.47161 | 0.500 |
| 30% | adaptive_252 | 6.571 | -22.41% | 12.9531 | 0.00000 | 0.00000 | 0.000 |

Adaptive minus fixed paired daily net delta=-0.0000570; 20-session circular block bootstrap CI95=[-0.00032671188797494985, 0.00020321044784086135] (5000 resamples, seed 20261009).

DSR trial estimate=15; current same-period candidate Sharpe variance=0.030582. This remains approximate because earlier candidate return series are not available on the same dates and the complete historical trial family is unknown.

## 判定と限界

判定: **PENDING**。この長期再生は約10年の相場環境と多数の通常日・連休・反転遷移を含むため、159日評価の期間依存性を確認する材料になる。ただし、現時点ですでに観測可能な歴史データを使うため、将来の未観測データによるprospective OOSではない。約定価格、銘柄別借株・逆日歩、share/cash drift台帳は未取得。macro snapshotのhistorical availability provenanceも未証明。したがって長期化だけで本番採用条件は満たさない。

再現コマンド:

```sh
mkdir -p reports/20261009_adaptive_overnight_inventory_v2_long
timeout -k 10s 8100s env MPLCONFIGDIR=/private/tmp/leadlag-mpl-cache .venv/bin/python src/research/scripts/experiments/experiment_adaptive_overnight_inventory_v2_long_20261009.py 2>&1 | tee reports/20261009_adaptive_overnight_inventory_v2_long/run.log
```

Artifacts: `metrics.json`, `policy_summary.csv`, `yearly_summary.csv`, `borrow_stress_summary.csv`, `calendar_and_reversal_segments.csv`, `ticker_carry_diagnostics.csv`, `daily_weights.csv`, `daily_policy_replay.csv`.

Registry: record_id=`f0b6f61a80f94546bb704792708343fd`, decision=`pending`, DSR=1.0, study_id=`adaptive-overnight-inventory-v2-long-2026-10-09`.
