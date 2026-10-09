# 銘柄別オーバーナイト在庫 sizing 研究

2026-10-09統合時の注記: 以下の数値・CSVはinventory-v3統合前の研究記録として保持する。再現コードは受入日の持越し損益、価格変動後の在庫/NAV、入ってくる暦日区間の費用、09:10価格区間へ更新したが、実データでの全再評価は未実行。この表を現行会計の成績・採用根拠にしない。DSRの探索履歴の完全性も未確認である。

## 仮説と事前条件

固定alphaの持越し在庫を、銘柄ごとの翌朝継続確率・継続時に再利用できる期待量・再売買回避費用から計算し、予想overnight PnLと保有費用、反転/flat時の手仕舞い費用を差し引いて決める。日次の同じV2ウェイトを両方式に使い、carry配分だけを比較した。ML overlayは入力不足で全行skipとなり、生成weightはoverlay適用前のV2値。

- Production baseline: `overnight_alpha_long=0.75`, `short=0.50`。本番設定は変更していない。
- 期間: モデル生成 2024-12-23–2026-10-06、評価 2026-02-02–2026-10-06（159日）。評価開始までは252本の過去遷移を使うwarm-up。年率化245日、全評価日を含む。
- 解決済み本番 side leverage=1.30、片道slippage=5.0bp、long financing=2.50%/年、short borrow=1.15%/年、reverse=2.0bp/暦日。
- Overlay: `models/ml_order_overlay/production_20260923` の `HISTORY.json` に従い、各期間で学習cutoffが未来にならないartifactを使用。gap storeは2026-08-07–2026-09-25の9日分で、それ以外は有効設定のon-demand BLPX経路を使った。評価期間fallback率=0.00%。
- Overlay適用状態: skipped 412/412、理由 `adr_not_supplied`。本比較の全weightはML overlay適用前。
- Macro入力: run-owned価格の読込に失敗し、412/412日でmacro data insufficient (0 rows)。macro調整は適用されていない。


- 現行overlayの promotion record にはoperator overrideと未通過gateが記録されている。これはcarryの研究replayであり、overlayや本番リスク停止の受入/解除を意味しない。
- inventoryは開始時flatから全日連続でreplay。終端は最終引け全清算、terminal alphaを0にして売買量と片道費用を計上した。execution volumeは実効opening/closing inventory flow合計、turnoverはその1/2。

### 時点制約

各日のcarry決定は、当日closeまでに分かる現在weightの符号、当日09:10までに確定した過去gap、そして決定日前日までに終わった銘柄別weight遷移だけを使う。次営業日のweight・target return・gapはalpha計算関数へ渡していない。次営業日weightは再利用率と反転分類の事後評価だけに使う。

同方向時の費用便益は `P(same) × E[overlap] × 2 × one-way slippage`。これへ方向付き過去平均gap returnを加え、暦日financing/borrow/reverse、反転時の持越し在庫exit slip、flat時のexit slipを引いた値をround-trip slipで割り、0–1へclipする。実際のPnLにはovernight markとcore ledgerの売買費用を一度だけ計上する。

## OOS結果

すべてのPnL値はポートフォリオreturn fraction。gross/net PnLは日次単純returnの合計と複利値を並記。MDDは初期wealth=1から計算。

| Policy | n | Net Sharpe | MDD | Gross PnL sum / comp. | Net PnL sum / comp. | Slip cost | Carry cost | Turnover/day | Reused inventory | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_alpha_0.75_0.50 | 159 | 1.767 | -35.13% | 0.6873 / 93.65% | 0.3218 / 34.50% | 0.3149 | 0.0506 | 1.9806 | 31.76% | 0.00% |
| adaptive_252 | 159 | 2.511 | -26.32% | 0.9139 / 140.02% | 0.5485 / 66.78% | 0.3477 | 0.0178 | 2.1867 | 26.28% | 0.00% |

コストのreturn fraction内訳:

| Policy | Slippage | Financing | Borrow | Reverse | Total costs | Overnight PnL |
|---|---:|---:|---:|---:|---:|---:|
| fixed_alpha_0.75_0.50 | 0.31491 | 0.01554 | 0.00477 | 0.03026 | 0.36548 | -0.11151 |
| adaptive_252 | 0.34768 | 0.01777 | 0.00000 | 0.00000 | 0.36545 | 0.11509 |

### 事前パラメータ感度

| Policy | Net Sharpe | MDD | Net PnL sum | Slip | Carry cost | Reuse ratio | 平均alpha(long/short) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_alpha_0.75_0.50 | 1.767 | -35.13% | 0.3218 | 0.3149 | 0.0506 | 31.76% | 0.746/0.498 |
| adaptive_252 | 2.511 | -26.32% | 0.5485 | 0.3477 | 0.0178 | 26.28% | 0.870/0.000 |
| adaptive_lookback_202 | 2.455 | -26.23% | 0.5444 | 0.3464 | 0.0179 | 26.93% | 0.873/0.000 |
| adaptive_lookback_302 | 2.453 | -27.31% | 0.5376 | 0.3475 | 0.0178 | 26.38% | 0.869/0.000 |
| adaptive_reversal_cost_0.8x | 2.508 | -26.36% | 0.5484 | 0.3475 | 0.0179 | 26.25% | 0.875/0.000 |
| adaptive_reversal_cost_1.2x | 2.514 | -26.28% | 0.5485 | 0.3479 | 0.0177 | 26.32% | 0.864/0.000 |

### Short borrow stress

Borrow fee is a hypothetical uniform annual rate on short carried exposure; historical ticker-level borrow/reverse charges were unavailable.

| Borrow annual | Policy | Net Sharpe | MDD | Net PnL sum | Borrow cost | Reverse cost | Avg short alpha | Reuse ratio |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 10% | fixed_alpha_0.75_0.50 | 1.565 | -36.22% | 0.2852 | 0.04145 | 0.03026 | 0.498 | 31.76% |
| 10% | adaptive_252 | 2.511 | -26.32% | 0.5485 | 0.00000 | 0.00000 | 0.000 | 26.28% |
| 30% | fixed_alpha_0.75_0.50 | 1.108 | -38.63% | 0.2023 | 0.12435 | 0.03026 | 0.498 | 31.76% |
| 30% | adaptive_252 | 2.511 | -26.32% | 0.5485 | 0.00000 | 0.00000 | 0.000 | 26.28% |

### 連休・翌朝反転の事後区分

区分は次営業日の実現date gap・実現weightを使った事後診断で、carry decisionには使っていない。calendar区分はportfolio session単位、same/reversal/flat区分は銘柄×session単位の加算可能な台帳。日次PnLはoutgoing trade dateへ帰属し、overnight markを含む。`Next-open slip delta vs flat` は次営業日寄りの実slippageを、carry在庫なしで同じtargetを売買した反実仮想と比べた増減費用（負値は節約）。

| Policy | Segment | Unit | n | Gross sum | Net sum | Mean net / unit | Overnight | Actual slip | Next-open slip delta vs flat | Carry cost | Reuse ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_alpha_0.75_0.50 | calendar_gap_1d | portfolio_sessions | 117 | 0.6088 | 0.3526 | 0.00301 | -0.01036 | 0.23196 | 0.03549 | 0.02422 | 30.41% |
| fixed_alpha_0.75_0.50 | weekend_or_short_holiday_2_3d | portfolio_sessions | 33 | 0.0379 | -0.0451 | -0.00137 | -0.09643 | 0.06407 | 0.00683 | 0.01885 | 36.34% |
| fixed_alpha_0.75_0.50 | extended_holiday_4d_plus | portfolio_sessions | 8 | 0.0372 | 0.0129 | 0.00161 | -0.00472 | 0.01683 | 0.00214 | 0.00750 | 32.99% |
| fixed_alpha_0.75_0.50 | terminal_final_close | portfolio_sessions | 1 | 0.0034 | 0.0014 | 0.00137 | 0.00000 | 0.00206 | 0.00000 | 0.00000 | 0.00% |
| fixed_alpha_0.75_0.50 | next_signal_reversal | ticker_session_transitions | 401 | 0.5602 | 0.4826 | 0.00120 | 0.53416 | 0.06583 | 0.03022 | 0.01180 | 0.00% |
| fixed_alpha_0.75_0.50 | next_signal_same_direction | ticker_session_transitions | 554 | -0.1765 | -0.2889 | -0.00052 | -0.71035 | 0.09318 | -0.03384 | 0.01925 | 88.82% |
| fixed_alpha_0.75_0.50 | next_signal_flat | ticker_session_transitions | 625 | 0.3001 | 0.1750 | 0.00028 | 0.06468 | 0.10561 | 0.04808 | 0.01952 | 0.00% |
| adaptive_252 | calendar_gap_1d | portfolio_sessions | 117 | 0.8021 | 0.5374 | 0.00459 | 0.18292 | 0.25596 | 0.03141 | 0.00877 | 25.46% |
| adaptive_252 | weekend_or_short_holiday_2_3d | portfolio_sessions | 33 | 0.0594 | -0.0185 | -0.00056 | -0.07485 | 0.07143 | 0.00692 | 0.00655 | 29.70% |
| adaptive_252 | extended_holiday_4d_plus | portfolio_sessions | 8 | 0.0490 | 0.0283 | 0.00354 | 0.00702 | 0.01821 | 0.00207 | 0.00245 | 24.92% |
| adaptive_252 | terminal_final_close | portfolio_sessions | 1 | 0.0034 | 0.0013 | 0.00135 | 0.00000 | 0.00208 | 0.00000 | 0.00000 | 0.00% |
| adaptive_252 | next_signal_reversal | ticker_session_transitions | 401 | 0.4800 | 0.3998 | 0.00100 | 0.45396 | 0.07567 | 0.02245 | 0.00459 | 0.00% |
| adaptive_252 | next_signal_same_direction | ticker_session_transitions | 554 | 0.1180 | -0.0009 | -0.00000 | -0.41589 | 0.11286 | -0.01579 | 0.00607 | 77.21% |
| adaptive_252 | next_signal_flat | ticker_session_transitions | 625 | 0.3125 | 0.1821 | 0.00029 | 0.07702 | 0.12322 | 0.03375 | 0.00711 | 0.00% |

### 比較の不確実性と判定

Adaptive minus fixed-alpha paired daily net delta: 0.001425; 20-session circular block bootstrap 95% CI [0.00044269394787089535, 0.0023871824843978116] (5000 resamples, seed 20261008). CI is an uncertainty interval, not a probability of strategy superiority.

DSR trial count estimate: 9 (3 prior fixed-alpha variants found in existing result files plus 6 current policies). Same-window annual Sharpe values used for the cross-trial variance proxy: `1.7675, 2.5108, 2.4551, 2.4527, 2.5078, 2.5138`. Earlier historical candidates do not have matching-date Sharpe series, so DSR remains approximate; the exact trial family may be larger.

判定: **PENDING**。OOSは159日で単一区間、実約定価格・ticker別borrow/逆日歩の実費がなく、carry accountingも数量・cash driftではなくweight-based simulated notionalであるため本番採用判断の証拠ゲートに達していない。

再現コマンド:

```sh
timeout -k 10s 3600s .venv/bin/python src/research/scripts/experiments/experiment_adaptive_overnight_inventory_20261008.py 2>&1 | tee reports/20261008_adaptive_overnight_inventory/run.log
timeout -k 10s 180s .venv/bin/python src/research/scripts/experiments/refresh_adaptive_overnight_inventory_report_20261008.py
```

Artifacts: `metrics.json`, `policy_summary.csv`, `borrow_stress_summary.csv`, `calendar_and_reversal_segments.csv`, `ticker_transition_attribution.csv`, `ticker_carry_diagnostics.csv`, `ticker_summary.csv`, `daily_policy_replay.csv`.

Registry result will be appended below.

Registry: record_id=`03c34337835b4d15bbb819ed49eecee9`, decision=`pending`, DSR=0.9570694686887353, study_id=`adaptive-overnight-inventory-pit-2026-10-08`.
