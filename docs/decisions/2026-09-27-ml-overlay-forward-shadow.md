# ML overlay 増分価値の前向きシャドー仕様（2026-09-27）

## 目的と状態

既知期間のコスト感度だけでは、現在のML overlayの実行後の増分価値を判定できない。2026-09-28以降、現行のoverlay有効設定と同一入力からML無効の対照ウェイトを生成し、将来データを追記保存する。開始時点の判定は **PENDING_REAL_EXECUTION_COSTS** とする。研究上の採用、モデル切替、注文変更はこの仕様では行わない。

## 固定する比較

- **ML有効**: 継承解決した `configs/production/production.yaml` と、開始時点のartifact `20260926T192555935698Z-ee306a32f3ec`（train end 2025-12-30、SHA-256 `c1a5b9354dae4b6507b41433c3b1c6a90451a81bcf0d3bc78b07939f421cb1e9`）。
- **ML無効対照**: 同一の `DecisionInputs`、同一V2設定・費用・リスク・fallbackを使い、`ml_overlay_enabled`だけをfalseにした `ProductionRunner`。
- 有効/無効の両方について、日付、09:10時点、入力fingerprint、effective gap distribution hash、config fingerprint、モデルartifact metadata、17 ETF weights/scores、fallback/監査状態、現在値・前日終値、資本額を記録する。
- 17 ETFの09:10 bid/askと板情報は読み取り専用captureとして別の追記JSONLに保存する。shadow結果とcaptureはtrade date・時刻をキーに引け後に照合する。
- 比較対象artifact/configが変わった後の結果は同一系列へ混ぜない。記録済みデータは編集・削除せず、同日再実行は別attemptとして残す。

## 実費と評価単位

モデルの `CostBreakdown` は実費とみなさない。損益は口座資本に対する日次returnで評価し、ML有効側は注文intent・部分約定・fill price・fee明細・引け時点建玉をruntime ledgerとbroker記録で照合する。ML無効対照の約定数量と費用は、同一資本、同一ロット/注文制約、当日板・約定可能量を使うcounterfactual execution replayで推定し、実測値と推定値を別欄にする。板・close quote・fee/borrow等の根拠が欠ける日は採否用の完全paired日数に含めず、欠損理由を残す。ML有効だけが実際に執行されるため、counterfactual ML無効の費用は厳密な実約定ではなく、校正済み推定値であると報告する。

主比較は全flat日を含むpaired daily net PnL差（ML有効−ML無効）。費用をslippage、手数料、financing、borrow、reverseに分解し、実測・推定を区別する。gross/net Sharpe、最大DD、turnover、fill率、fallback率も両方について出す。現行モデルの5/10/20bps stressは仮定値として残し、実測spreadや約定費用へ置き換えない。

## 判定条件

次をすべて満たすまで有効性の採否を出さない。

1. 現行artifactの学習終端2025-12-30より後の、完全なpaired forward評価が250取引日以上ある。
2. paired日次net PnL差の20日non-circular moving-block bootstrap（5,000回、seed 20260924）の95%区間下限が0より大きい。
3. 年率net Sharpe差が0より大きく、最大DD悪化が1.0 percentage point以内、turnover増加が10%以内。
4. 約定・実費照合の適格率、quote/板充足率、non-fill/partial-fill内訳を開示し、欠損や失敗日を落として良い結果だけを選ばない。
5. 数値・リーク監査が全paired日にPASSし、fallback率を増やさず、RuleD後のmodel net ±0.05 / gross ≤2.0を守る。side leverage適用後の実効gross/netは別に確認する。

どれかを満たさない場合は `PENDING` または `REJECTED` とし、既知368日再集計や今回のシャドー記録だけを理由に本番設定を変えない。本番artifactが変わる場合は新しい比較期間を別登録する。

## 実装と限界

`scripts/batch/run_decision_v2.sh` が通常のlive decision時にシャドー保存を有効化する。ML無効対照計算または保存が失敗しても、本番ML有効の結果・発注ウェイトは変更せずエラーをログへ残す。読み取り専用captureは09:10:00〜09:10:30 JSTのquote/board取得だけを行い、失敗時は本番decisionを継続する。出力先は `var/shadow_runs/ml_overlay_value/`。

**現行実装の追記（2026-09-29）:** 上記のcapture失敗時にdecisionを継続する計画は、actual-liveのquote入力契約としては採用しない。現行actual-liveは完全な当日frozen quote snapshotがなければ処理を中止する。ML shadow保存の失敗を注文判断から分離する方針とは別の制御である。詳細は [2026-09-29 decision](2026-09-29-frozen-0910-account-risk-forward-eval.md)。

2026-09-28 00:39 JSTに既存の `com.leadlag.decision` が平日09:10にロード済みと確認した。専用shadow Agentを同じ時刻に追加すると `live:production_v2` guardを競合するため、試験登録した `com.leadlag.ml-overlay-shadow` は直ちにbootoutし、plistを削除した。既存 `run_decision_v2.sh` が同一live decision中にペアフックを実行する。追加した `--shadow-only` は安全に発注経路を迂回するCLI機能だが、独立Agentは登録していない。既存decision Agentの直近終了コードは1で、9/25ログではgap計算失敗、VaR/ES履歴計算timeout、risk stopによる発注停止が確認された。ペアフック導入後の初回記録はまだなく、9/28の実行後にログとJSONLを確認する。引け後のcandidate実約定・費用台帳とbaseline counterfactual execution replayの照合器は別途必要であり、現在の保存レコードには将来実現収益をまだ付与しない。
