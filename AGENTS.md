# 日米リードラグ・ファンド改善ガイド

## ⚠️ 実行規約（必須・違反禁止）

- **`python3 -c "..."` のインライン実行は禁止** — ハング・スタックの主要原因。必ずスクリプトファイル（`scripts/experiments/` 配下等）を作成して `python3 scripts/...` で実行すること
- 長時間実行コマンドには必ずタイムアウトを設定すること
- 詳細は `docs/スタック再発防止策.md` 参照

## 戦略概要

米国セクターETF（15銘柄: SPDR 11 + Style 4）の当日リターンから、翌営業日の日本 TOPIX-17 セクターETF（17銘柄）の **9:10→大引けリターン** を予測する日次マーケットニュートラル戦略。

- **本番モデル**: `ProductionV2Model` (Residual-BLPX-RA v2) — `src/leadlag/models/production_v2.py`
  - BLPX 構造化投影シグナル + gap調整予測分布 + `mu_over_sigma` ランキング + RuleD 動的グロス（PIT三分位: Low→0.75x, Mid/High→1.00x）
- **フォールバック**: gapデータ欠損時・監査失敗時は **フラットポジション（w_final=0）** を返す（V1フォールバックは2026-07に廃止）
- **本番config**: `configs/production/production.yaml`（正本）。`production_v2_primary_ruleD.yaml` は旧版（overnight holding・multi-horizon blend・rank reversal overlay 未含む）
- **アーキテクチャ詳細**: `docs/ARCHITECTURE.md`、数理仕様: `docs/モデル技術仕様書.md`

## データ整合規約（最重要・変更禁止）

`df_exec` の行 t の意味:
- **US列 (`us_cc_*`)**: 米国営業日 D_t のクローズ・トゥ・クローズリターン（JST 翌朝に確定）
- **JP列（ターゲット）**: 取引日 D_{t+1} の 9:10→大引けリターン（`compute_jp_target_returns` in `src/leadlag/data/preprocessor.py`）
- **`jp_gap_*`**: 取引日の寄付ギャップ（9:10 判定時点で既知 → シグナルに使用可）
- 相関窓は `all_returns[window_start:current_index]` で **当日行を除外**（`src/leadlag/core/signal.py`）。この規約を崩すと即リークになる

## 不変条件（改善時に絶対に守ること）

1. **ルックアヘッド禁止**: すべてのローリング統計・ベータ・PITビニングは strictly historical。`ComplianceAuditor`（`src/leadlag/compliance/auditor.py`, `v2_auditor.py`）の監査項目（`check_pit_binning_lookahead`, `check_residualization_leakage` 等）を無効化しない
2. **ベースライン期間の分離**: 事前分布・基準相関 `c_full` は 2010–2014 固定（`compute_baseline_correlation`）。バックテスト `start_date` は 2015-01-05 以降を維持。`_prepare_residual_prior` のフォールバック（先頭1260行）が発動する構成は作らない
3. **テストを弱めない**: 変更後必ず全テストを通す。推奨は並列実行 `bash scripts/run_tests_parallel.sh`（約8分、ログは `/tmp/pytest_parallel/`）。直列 `python3 -m pytest tests/ -v` は約32分。unit + integration（`test_leakage_audit.py`, `test_production_residual_blpx.py` 等）
4. **市場中立制約**: net exposure ±0.05、gross ≤ 2.0（RuleD 適用後）。リスク正本は `src/leadlag/core/risk.py`、グロス調整正本は `src/leadlag/core/portfolio.py::adjust_gross_exposure()`
5. **ティッカー定義**: `src/leadlag/data/tickers.py` が単一正本（N_U=15, N_J=17, 計32次元）で、感応度ラベル `w3`–`w6` も `SENSITIVITY_LABELS` として同ファイルに保持。`core/correlation.py` はレジストリから生成するため、ユニバース変更時は `tickers.py` のみ更新すればよい
6. **前日gap行列の使用禁止**: 当日のgap行列（`mu_gap_{YYYYMMDD}.npy` / `omega_gap_{YYYYMMDD}.npy`）が存在しない場合は **フラットポジション（w_final=0）** を返すこと。前日行列をコピーして当日日付で使用してはならない（誤ったポジションで発注するリスクがある）。`load_gap_matrices`（`production_v2.py`）は当日日付のファイルのみを検索し、シェルスクリプト側のフォールバックコピー（2026-07-14に廃止）に依存しない

## 改善ワークフロー

1. **仮説→実験**: 実験スクリプトは `scripts/experiments/` に作成（本番パス `src/leadlag/` に直接実験コードを入れない）。実験用モジュールは `src/experiments/` へ
2. **バックテスト・日次決済**: 本番 V2 は `BacktestEngine.run_v2_backtest()`（`src/leadlag/execution/backtester.py`）を使用。CLI `backtest` / `decision` は V2 一本化。V1 SRE モデル `SectorRelativeEnsembleModel` は `archive/legacy_src/models/sre.py` に移設。レガシー V1 汎用 `BaseModel` バックテストは `research.backtest_v1.run_v1_backtest()` に移設。コストは片道5bps + 金利・貸株・逆日歩を含む **net** で評価
3. **過学習ガード（必須）**:
   - このリポジトリには過去の実験config・スクリプトが大量にあり（`archive/experiments/` 約30本）、同一ヒストリー上での反復選択が既に多い。**新パラメータ追加は原則避け、追加時はパラメータ±摂動の感度分析と Deflated Sharpe（試行回数補正）を必ずレポートに含める**
   - ウォークフォワード検証（先例: `reports/phase3_walkforward_validation_report.md`）で OOS 確認
4. **シャドー運用**: 昇格前に `tools/validation/monitor_residual_blpx_shadow_performance.py` / `shadow_runs/` でライブ整合を確認
5. **本番昇格**: `configs/production/` の config 更新 + `docs/ARCHITECTURE.md` のリファクタリング履歴へ追記
6. **レポート**: `reports/<sprint名>/` に markdown で結果を残す（既存 sprint0–3b の形式に倣う）

## 既知の落とし穴（コードレビュー指摘済み）

- **`sre.py` のインスタンス状態一時書き換え（解決済み）**: `build_c0_from_v0` グローバル差し替えは `c0_override` 引数経由に修正済み。`self.k` の一時書き換えも `k_override` 引数経由に修正済み（2026-07-13）
- **グローバルキャッシュ（解決済み）**: `_PRODUCTION_SIGNAL_CACHE` 等のモジュールレベルdictはインスタンス属性 (`self._production_signal_cache` 等) に移行済み（2026-07-13）。`predict_signals` 開始時にインスタンスキャッシュは clear される。`core/correlation.py` の `_ROLLING_CORR_CACHE` / `_BASELINE_CORR_CACHE` は関数レベルで管理されサイズ上限あり
- **9:10 価格近似**: 5分足 09:10 バーの (High+Low)/2 を執行価格としており楽観側。コスト検証時は実約定ログと突合すること
- **金利コスト日割り（解決済み）**: `backtester.py` は `calendar_days` で暦日数を計算し、`financing_daily * days_held` で課金。週末（金曜→月曜=3日）も正しく加算される
- **VaR99 の不安定性**: 250日窓の99%は尾部標本 ~2.5個。stop 判定の変更時は注意
- **ハング既知パターン**（CLI実行時）: yfinance ダウンロード、`cache.py` の fcntl ファイルロック、`close.py` の auto-close 無限待機、API再試行バックオフ。詳細は `docs/スタック再発防止策.md`。長時間実行はタイムアウト付きで
- **yfinanceのティッカー別NaN欠損**: yfinanceダウンロード時に特定ティッカー（IJR等）のデータが日付以降全てNaNになることがある。`preprocess_data()` のNaNチェック（`preprocessor.py:264-272`）で1ティッカーでもNaNがあると該当日の全レコードがスキップされ、df_execが途中で切断される。`etf_data.pkl` の異常は `preprocess_data` 呼び出し前に検査・修正すること
- **config dictのshallow copy**: `base_cfg.copy()` はネストした dict（`cfg["blpx"]` 等）を共有参照する。比較実験で2つのモデルに異なるconfigを渡す際は `copy.deepcopy(base_cfg)` を使うこと。shallow copy だと一方の変更が他方に伝播し、両モデルが同一設定になる（実例: Robust PCA 比較実験で両モデルが Robust PCA 有効化されシグナルが完全一致した）

## よく使うコマンド

```bash
# テスト（並列・推奨、約8分）
bash scripts/run_tests_parallel.sh

# テスト（直列、約32分）
python3 -m pytest tests/ -v

# 日次本番実行（v2）
python3 tools/production/run_daily_production_v2.py

# gap調整分布の事前計算（v2 の入力）
python3 tools/research/compute_gap_adjusted_distribution.py

# 本番 V2 バックテスト（gap 行列が事前計算済みの場合は --gap-dir 指定）
python3 src/research/scripts/backtest/run_production_backtest.py --start-date 2015-01-05

# CLI経由 V2 バックテスト（--config / --gap-dir / --slippage-bps 等）
python3 -m leadlag.cli backtest --start-date 2015-01-05

# CLI経由 V2 本番決済（--config / --gap-dir / --live-dir / --api-enable 等）
python3 -m leadlag.cli decision --config configs/production/production.yaml --gap-dir var/live/pipeline_data/gap_adjusted_distribution/latest --api-enable --capital-from-wallet

# 構文チェック（CLIスタック防止: python3 -c は使わずスクリプト経由で）
python3 -m compileall src/leadlag tests tools scripts src/research
```

## 評価指標の約束事

- 主指標: **net Sharpe**（コスト後）、最大DD、ターンオーバー、フォールバック発動率
- gross/net 両方を報告し、コスト内訳（slippage / financing / borrow / reverse）を分解
- 「Sharpe改善なし」の結論も価値がある（例: Health Score によるサイズ調整は検証の結果不採用、`docs/ARCHITECTURE.md` Phase 9 参照）。不採用の実験も必ずレポート化して二重検証を防ぐ
- **不採用実験の記録**（再検証防止用）:
  - **Robust PCA伝播行列**（2026-07）: B_struct を低ランク+スパース分解（L+S）で置換する方針を検証。セクター事前知識（M_sector）とPCA事前分布（B_pca）の統合が失われ、confidence weighting の inv_A_tikh も単位行列フォールバックになった結果、Sharpe -35%、IC -32%と大幅劣化。チューニングでは埋められない構造的欠陥が原因。コードは全て破棄済み
  - **V1フォールバック廃止**（2026-07）: gapデータ欠損時のV1ウェイトフォールバックを廃止しフラットポジション化。理由: (1) `production_v2_writer.py` がV2実行のたびに `v1_baseline_weights.csv` を `w_v1` で上書きする循環参照があり、V1ウェイトが新規計算されず凍結化していた (2) 一度ゼロになると永久にゼロになる（実例あり） (3) データパイプライン障害時に古いシグナルで取引するより取引を見送る方がリスク管理として健全
  - **前日gap行列フォールバック廃止**（2026-07-14）: `run_gap_distribution.sh` の前日行列コピー機能を廃止。当日のgap行列が生成できない場合、前日行列を当日日付でコピーして使用していたが、これにより誤ったポジションで発注するリスクがあった（2026-07-14に発生: 手動フル再計算が `latest` シンボリックリンクを上書きし、フォールバックなしで `mu_gap_20260714.npy` が不在になりフラットポジション化）。廃止後は当日行列不在時にフラットポジションを返すのが正しい挙動。`preprocess_data` を修正し `r_oc` NaNの行も0埋めで残すことで、大引け前に当日のgap行列を計算可能にしたことで再発防止
  - **Fractional Differentiation採用**（2026-07-21）: US ETFリターンにLópez de Prado (2018)の分数階差分（d=0.1）を適用。ウォークフォワード検証（2015-2026、12年次ウィンドウ）でd=0.1がd=1.0ベースラインを12/12ウィンドウで上回る（平均Sharpe 8.87 vs 7.75）。d=0.5も12/12で上回るが改善幅が小さい（+0.67 vs +1.12）。実験スクリプトは `archive/experiments/` にアーカイブ、検証レポートは `reports/fractional_diff_walkforward_audit_report.md`
  - **Transfer Entropy統合**（2026-07-22）: TEによる共分散行列調整（diagonal congruence transform `D @ Omega @ D`）を検証。ウォークフォワード（252日訓練/63日テスト、6窓）で6窓中5窓の動的最適alpha=0.0（TE無効化が最適）。TE-Static（固定alpha=0.5）はSharpe 7.02 vs Base 7.66と劣化。MDD改善はポジション縮小の副作用でリスク調整後リターンは悪化。本番未統合。コードは実験用として保持（`src/leadlag/features/transfer_entropy.py`、`src/leadlag/models/production_v2_te.py`）、詳細レポートは `reports/te_walkforward_experiment.md`
  - **非線形シュリンケージ不採用**（2026-07-24）: Ledoit & Wolf (2020)の解析的非線形シュリンケージ（MPベース・Empiricalベース）で `regularize_correlation` の Stage 1（線形LW）を置換する方針を検証。ウォークフォワード（2015-2026、12年次ウィンドウ）でNL-MPは0/12窓で劣化（平均Sharpe 8.04 vs 8.87、Δ=-0.83）、NL-Empiricalは7/12窓で微改善（平均Sharpe 8.91、Δ=+0.04、統計的に有意でない）。MP法はシグナル固有値を過度に縮小、Empirical法は縮小が弱く線形LWと同等。既存の2段階正則化（線形LW + 構造事前分布C0）がN=32/T=504で十分に最適化されており、Stage 2（構造事前分布）が性能の主要因。実験コードは `src/experiments/nonlinear_shrinkage.py`、詳細レポートは `reports/nonlinear_shrinkage_walkforward_report.md`
  - **時変gap係数（Kalman filter）不採用**（2026-07-24）: Dangl & Halling (2012)に基づき固定gap_open_coef=0.70/topix_beta_coef=0.60をKalman filter+BMAによる時変係数に置換する方針を検証。ウォークフォワード（2015-2026、12年次ウィンドウ）でKalmanは1/12窓のみ固定を上回る（平均Sharpe 7.58 vs 8.88、Δ=-1.30）。感度分析は安定（0.4% range）。劣化原因: Kalmanが予測誤差最小化を目指すのに対し、実際のパフォーマンスはcross-sectional ranking→weight→net returnの多段階プロセスで目的関数がミスマッチ。実験コードは `src/experiments/tv_gap_coef.py`・`scripts/experiments/experiment_tv_gap_coef.py`、詳細レポートは `reports/tv_gap_coef/tv_gap_coef_walkforward_report.md`
  - **USセクター横断面ランク入力不採用**（2026-07-29）: BLPX投影前にUS 15銘柄のリターンを横断面ランク（[-1,1]スケール）に変換する案を検証。ウォークフォワード（2018-2026、9年次ウィンドウ）でベースラインに対し1/9窓のみ勝利、プールSharpe 6.45 vs 7.20、Rank IC 0.226 vs 0.229と劣化。ボラティリティ規模を除去すると共分散構造・PCA/BLPX事前分布との整合性が損なわれ、相対順位のみでは予測力が低下。実験コードは破棄済み、レポートは `reports/us_cs_rank_blpx/us_cs_rank_blpx_report.md`
  - **ML order overlay US 特徴量追加不採用**（2026-08-01）: `ticker × score / gap` の per-ticker 交互作用ベース LightGBM overlay に、米国セクター close-to-close 特徴量（mapped US return / US 横断面 zscore・rank / 60日 rolling correlation 等）を追加する案を検証。basic US も refined US も per-ticker only と比べて OOS 改善なし（pooled OOS Sharpe: per-ticker 6.6992、basic US 6.7043、refined US 6.6904）。US 特徴量は統計的に有意な追加効果をもたらさず、本番モデル `models/ml_order_overlay/phase2_8`（per-ticker 交互作用のみ）を継続。レポートは `reports/ml_order_decision/us_features_experiment_decision_20260801.md`
  - **US→JP gap 予測精度ベースの信頼度オーバーレイ不採用**（2026-08-01）: 各JP銘柄の寄りギャップ（`jp_gap`）を米国セクターcc + `topix_night` からrolling OLSで予測し、その外れ具合でBLPXシグナルをスケール/キルする案を検証。2015-2026 全期間バックテストで、baseline net Sharpe 4.57 → daily 4.06 → per-asset 1.73、MDD -7.20% → -21.60% と大幅劣化。直近5日ドローダウンは軽減できた（-6.51% → -5.45%）が全期間でリスク調整後リターンが悪化。原因: (1) 日次スカラー信頼度は `build_weights` のスケール不変性で実質0/1キルに等しく、28%の取引日を落としてαを損失 (2) 銘柄別信頼度はgap予測誤差が本戦略の residual 9:10→大引け予測と弱相関。実験スクリプト・レポートは本セッションで一時ファイルとして削除済み。
  - **Gap-Sign コンシステンシー不採用**（2026-08-01）: BLPX 自身の close-to-close 残差予測 `r_hat_jp_cc` と実際の `jp_gap` の符号一致性を信頼度にする案を検証。Baseline Sharpe 4.57 → per-asset 3.45 → daily majority 2.42 → per-asset+daily majority 1.71 と劣化。直近5日（2026-07-27〜31）のドローダウンは `-6.51% → -2.02%` と大幅軽減できたが、全期間で MDD -7.20% → -23.41% と悪化しポジションを過度に削減。原因: 寄りギャップは 9:10→大引け ターゲットとは別の時間スケールであり、符号不一致が必ずしも当日の予測ミスを示さない。実験スクリプト・レポートは本セッションで一時ファイルとして削除済み。
  - **セクター連動度6施策 単体評価**（2026-08-01）: 本命（rolling IC, common-factor R²）、対抗（gap overreaction, overnight shock）、大穴（macro conduit, idiosyncratic gap）の6種 per-asset confidence を独立に評価。結果：5/6は baseline（net Sharpe 4.57, MDD -7.20%）を下回る。唯一 `counter_overnight_shock`（`exp(-|r_hat_jp_cc - jp_gap| / (scale_factor * target_vol))`）が marginal 改善を示した（scale=2, vol=63: Sharpe 4.63, MDD -5.99%, AR 123.2%）。パラメータ感度（scale_factor ∈ {0.5,1.0,1.5,2.0,3.0,4.0,5.0}、vol_window ∈ {21,63,126,252}、28組）では `scale=2.00, vol=252` が net Sharpe 4.69、`scale=1.50, vol=252` が MDD -5.94% / Sharpe 4.68 / DSR 39.31 とバランス良好。ただし AR は 145.8% → 113-122% と低下。他は `main_rolling_ic`（3.75 / -15.47%）、`main_common_factor_r2`（3.89 / -6.09%）、`counter_gap_overreaction`（3.41 / -11.08%）、`longshot_idiosyncratic_gap`（4.23 / -6.00%）、`longshot_macro_conduit`（4.60 / -7.81%）。`counter_overnight_shock` 以外は不採用。採用可否はウォークフォワード検証後に再判定。実験スクリプト・レポートは本セッションで一時ファイルとして削除済み。
  - **counter_overnight_shock 不採用**（2026-08-01）: 年次OOS（12年）で平均 net Sharpe は `scale=2.00, vol=63` で 5.61（baseline 5.34）、MDD -3.33%（baseline -4.01%）と改善するも、AR は平均 137.8% → 116.7% と大幅低下。対 baseline の年次 Sharpe の paired t-test p値はすべて > 0.05（scale=2.00/vol=63: p=0.27、scale=2.00/vol=252: p=0.20、scale=1.50/vol=252: p=0.55）で統計的有意差なし。α 損失に対するリスク調整後リターン改善が限界のため本戦略には採用しない。実験スクリプト・レポートは本セッションで一時ファイルとして削除済み。
  - **Overnight Reversal 追加検証不採用**（2026-08-01）: `counter_overnight_shock` の逆仮説として「gap が r_hat を大きく上回る（過反発）銘柄に高信頼度」を試した。`reversal_distance`（4.48 / -9.88%）と `reversal_overshoot`（4.53 / -11.13%）はいずれも baseline（4.57 / -7.20%）と `counter_overnight_shock`（4.63 / -5.99%）を下回った。大きなショックがある日に BLPX シグナルを強めるのはリスク上昇だけでリスク調整後リターンは改善しない。実験スクリプト・レポートは本セッションで一時ファイルとして削除済み。
  - **US-JP per-asset linkage overlay 不採用**（2026-08-01）: US-JP 連動度を per-asset 信頼度に変換する3案（per-asset us-gap IC, per-asset us-rhat IC, per-asset US t-stat）を検証。`per_asset_us_rhat_ic_inv`（window=63）が全期間で最も良好（net Sharpe 4.72, MDD -7.21%, AR 142.06%）に見えたが、ウォークフォワード（2015-2026、12年次）では baseline（平均 Sharpe 5.34）と `rhat_inv_w63`（5.35, p=0.91, wins 5/12）、`rhat_inv_w76`（5.36, p=0.88, wins 7/12）で有意差なし。パラメータ感度（window 50/63/76/126）も安定した改善を示さず、OOS ロバスト性がないため不採用。実験スクリプト・レポートは本セッションで一時ファイルとして削除済み。
  - **LGBM overlay 日米VIX特徴量追加不採用**（2026-08-02）: `ticker × score / gap` の per-ticker 交互作用ベース LightGBM overlay に、lagged 60日 log z-score の `us_vix_z` / `jp_vix_z` / `vix_spread_z` と各種交互作用を追加する案を検証。ウォークフォワード（2022-2024、3年次）で VIX overlay は no-VIX overlay と比べて marginal 改善に留まり（pooled OOS Sharpe: no-VIX 7.0280、VIX 7.0348、Δ=+0.0068）、年次勝率は2/3、平均日次リターン差は -0.000010（p=0.4245）で統計的有意差なし。さらに 2026 年後半（データは 2026-07 まで）の直近 7 日（2026-07-23〜31）で大きな DD（baseline MDD -10.66%）が発生したが、VIX overlay は no-VIX  overlay とほぼ同一（MDD -10.8864% vs -10.8842%）で下落保護効果は認められず、かつ overlay 全般が baseline よりも DD を拡大。VIX 特徴量は一部 importance を示すものの、実質的な追加 α/ダウンサイド保護は小さく、既存 `phase2_8` per-ticker 交互作用モデルを継続。実験コードは `scripts/experiments/experiment_ml_order_vix_walkforward.py` および `scripts/experiments/experiment_ml_order_vix_2026h2.py`、レポートは `reports/ml_order_decision/vix_overlay_walkforward/vix_overlay_walkforward_report.md` および `reports/ml_order_decision/vix_overlay_2026h2/report.md`
