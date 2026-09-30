# 収益改善順序12：LGBM予測の時系列校正診断

実行日時: 2026-09-23T11:54:18.039998+00:00。既存予測を再利用し、モデル・本番設定・注文経路は変更していない。

## 仮説と固定設計

既存LGBMの予測順位に情報があるなら、過去日だけで推定した切片・傾きによる拡大型校正で、未使用の次日予測の誤差と固定コスト注文ゲートが改善する。予測ラベル・価格データ・中心コスト12.5bps/片道は順序11と揃える。校正候補は過去50/63/76営業日の最低履歴で開始し、63日を事前主候補とする。校正は銘柄日をまとめたOLS、当日より前の予測・実現値のみ使用。

- 対象: 2024-12-23〜2026-07-29、368営業日。順序11で既に確認済みで、fresh holdoutではない。
- 価格・ラベル: `20260920_structural_completion/r3_capture_gap/exact_df_exec.pkl` と `var/market_data/etf_prices.sqlite` の5分足を読み、順序11と同じターゲット計算を再現。5分足1629.Tはloaderの読込時分割調整を適用。ラベルは欠損9:10入力で9:00→大引けにフォールバックする。
- 保有: 前日のモデルウェイトを実在庫proxyとして継続する既存研究実装。約定・板厚・手数料・impactは今回も実測されない。
- 9:10入力の評価期間内カバレッジ: 完全17銘柄 8/368日、少なくとも1銘柄の値あり 96/368日、セル充足率 22.2%。
- 学習artifact metadataはhistorical provider `available_at`の証跡を持たない。リークの証明ではないが、完全なPIT証明でもない。

## 再現確認

- exact_df_execと5分足cacheによるcanonical 5bps baselineの日次リターン最大差: 9.986e-17。
- 保存済み12.5bps raw gateとのweight最大差: 9.645e-16。

## 予測校正結果

| 最低履歴日数 | 評価日数 | raw MAE | 校正MAE | raw slope | 校正後slope | 校正−raw 日次MAE差 | 95% block CI |
|---:|---:|---:|---:|---:|---:|---:|---|
| 50 | 318 | 0.00892 | 0.00892 | 0.641 | 0.619 | -0.000005 | [-0.000038, +0.000025] |
| 63 | 305 | 0.00889 | 0.00889 | 0.640 | 0.627 | -0.000008 | [-0.000041, +0.000025] |
| 76 | 292 | 0.00871 | 0.00870 | 0.530 | 0.540 | -0.000014 | [-0.000047, +0.000017] |

MAE block CIは日付を単位にした20営業日circular block bootstrap、1000回、seed=42。63日主候補のMAE差は−0.08bp/銘柄日で、95%区間は0をまたぎます。共通期間のraw最上位十分位は予測+43.0bp、実現+9.8bp、校正後予測も+41.5bpで、過大さが残りました。全候補期間は過去に閲覧済みであり、区間は独立OOSの採否根拠ではありません。分位診断は `raw_reliability_deciles.csv`、係数系列は `calibration_coefficients.csv`。

## 固定コストの注文ゲート（共通期間）

校正startup感度を同じ共通期間 2025-04-25〜2026-07-29 (292日) で比較。

| 方法 | net Sharpe | gross Sharpe | max DD | turnover | cost sum | mean model gross |
|---|---:|---:|---:|---:|---:|---:|
| canonical baseline | 0.948 | 6.904 | -42.94% | 1.291 | 1.6509 | 1.759 |
| raw forecast gate | -1.179 | 3.419 | -39.95% | 0.407 | 1.2095 | 1.996 |
| calibrated, warmup 50d | -0.725 | 3.789 | -39.72% | 0.429 | 1.2232 | 1.994 |
| calibrated, warmup 63d (primary) | -0.725 | 3.789 | -39.72% | 0.429 | 1.2232 | 1.994 |
| calibrated, warmup 76d | -0.719 | 3.801 | -39.72% | 0.430 | 1.2237 | 1.994 |

Primary 63d gate minus same-cost baseline paired mean daily net difference: -0.001569, 20-day block-bootstrap 95% CI [-0.003217, -0.000112]。名目同期間DSR=0.0355、比較候補Sharpe数=5。過去の試行と既知OOSの再閲覧を完全には補正できないため、DSRは採否に使わない。

## 判定と制約

**保留（開発診断のみ）**。予測校正・同じ既知期間内の注文ゲートの変化は観察できたが、データを既に閲覧済みであり、9:10値は全期間の多くでopen-to-close fallback、実在庫と実約定費用もないため、新方式の採用可否は判定しない。未使用OOSの独立検証は、PIT取得時刻証拠・9:10入力・約定台帳を整え、校正方式を固定した後に行う。

## 成果物

- `results.json`: 全指標と固定条件
- `daily_forecast_calibration.csv`: 日次・銘柄別予測と実現ラベル
- `calibration_coefficients.csv`: 当日より前で推定した校正係数
- `raw_reliability_deciles.csv`: raw予測分位と実現リターン
- `daily_portfolio_returns.csv`: baseline/raw/calibrated gateの日次netリターン
- `weights_calibrated_*.csv`, `daily_gate_calibrated_*.csv`, `orders_calibrated_*.csv`: gate経路再現
