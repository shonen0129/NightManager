# DSR入力契約とstudy探索履歴

- Date: 2026-10-08
- Status: accepted
- Scope: Issue #41 / F16・F28。研究記録と統計入力の変更。

## 指標契約

正本は [experiment_registry.py](../../src/leadlag/experiment_registry.py)。`daily-v1`、`metric_status=valid`、Sharpe frequency、正の整数の年率係数を明示する。全評価日の有限な1次元return系列とTの一致を必要とし、欠損を削除してTを使い回さない。N>1には全N件のtrial Sharpe（ddof=1）、または明示した分散推定を必要とする。年率varianceは年率係数で、年率Sharpeはその平方根で日次へ変換する。双方があるときは頻度変換後の一致も検証する。全候補が同じ値のゼロ分散は許容し、負・非有限分散を拒否する。単一試行のnull varianceを試行間varianceの代わりに使わない。N=1はPSR。

外部分散には `trial_sharpe_variance_basis=external_estimate`、全候補標本分散には `cross_trial_ddof1` を明記する。外部推定の方法・対象試行・対象期間・相関による有効試行数の根拠はレポートに記録する。宣言だけで推定の妥当性や独立性を証明しない。成功例だけのtrial Sharpeを全N試行のvarianceと扱わない。

[研究helper](../../src/research/experiment_utils.py) の `extra_metrics` はcomputed field・status・観測数・DSR・年率入力を上書きできない。年率の変更は `metrics_spec=MetricsSpec(...)`、外部計算済み指標の明示入力は `record_simple_experiment(metrics=...)` を使う。computed fieldを追加metricからoverrideする契約は設けない。結果がない場合も同じ制約を適用する。schema未記録、returns未保存、探索件数不明はDSR `None` とする。数値が変わる場合は研究判断を再確認する。

## 事前登録とライフサイクル

同じJSONL正本に `study_registered` / `trial_started` / `study_selection` イベントを追記する。実験recordの既存readerはイベントを除外し、legacy ID生成と追記訂正を保持する。別のstudy正本ファイルは作らない。単一writerで研究記録を直列化する（並列workerは評価だけを行い、親writerが記録する）。

1. `register_study(study_id, hypothesis, candidates, protocol, history_complete, historical_trials_lower_bound)` を実行前に呼ぶ。protocolはtarget/cost schema、IS/OOS start/end（IS終了<OOS開始）、purge/embargo session数と理由、選択基準、daily/all-flat-days MetricsSpecを固定する。登録後の変更は新studyを作る。同じ仮説familyの先行探索が不明なら `history_complete=False`。過去に見た期間を新しい独立OOSとして宣言しない。
2. 候補ごとに `start_trial` で、safe parameters、code snapshot SHA-256、全入力data SHA-256を評価前に記録する。config hashとprotocol（期間・target/costを含む）hashはregistryが計算する。別候補名でも同studyで集約する。再実行は別のattempt、指標の訂正は同trialのcorrection。
3. helperへstudy IDとtrial IDを渡し、`completed` / `failed` / `aborted` のoutcomeと研究判定・reasonを追記する。中断・棄却も試行数へ含める。強制終了でoutcomeを記録できない試行はrunningで残り、明示した中断記録が付くまで選択を拒否する。起動されなかった候補を実行済み試行にしない。
4. `select_study` は全開始試行のoutcome、未開始候補、選択時点、選択trial、理由、探索下限を保存する。選択時点の全試行数でDSRを再検証し、最初の成功時点の値を使い回さない。外部分散または全N系列がない・history不明なら選択DSRはNone。選択後の追加試行は新studyにする。

[テンプレート](../../src/research/scripts/experiments/_template.py) は `--study-plan` / `--candidate-id` / `--code-hash` を必須とし、planの固定評価期間・MetricsSpecで実行する。df_exec（index/列を含む）とgap入力のhashを記録する。実験ごとに他の入力artifactも含む完全なdata hashを用意する。既存の事後記録helperは下限の証跡を残すが、開始記録なしに完全な探索履歴と主張できない。

## 過去索引と既存DSR

[履歴点検tool](../../tools/research/audit_study_history.py) は過去Markdown reportをhash付きで索引化する。MLと感応度のfamilyタグは事後の索引で、事前登録ではない。各reportを独立試行と数えず、重複し得るreportの件数を加算しない。名前の根拠を読めた候補だけを列挙し、完全な探索数・未報告中断・過去の実行hash・選択時刻・IS/OOS/purgeはunknownとして残す。索引はreportの参照資料であり、架空のExperimentRecordを追加しない。

```sh
PYTHONPATH=src timeout -k 10s 120s .venv/bin/python tools/research/audit_study_history.py \
  --output reports/study_review/history_index.json
```

既存registryに保存DSRがある行をrecord IDごとに入力・保存値・探索履歴で点検する。数値入力に問題がないが探索が不明な行は `unverified_search_history`。全既存DSRが数学的に誤りだとは判定しない。まずpreviewを確認し、必要時に同じコマンドへ `--write-corrections` を付けて追記する。元行・legacy ID・target訂正履歴を保持し、再実行は現在viewを使って二重訂正しない。訂正の影響を受ける研究判断は個別に再確認する。

Git管理外の実registryは今回の取得環境に存在しないため、[作業証跡](../../reports/20261008_study_governance/report.md) は個別の本番履歴点検を未実施と明記する。所有環境でpreview/追記訂正を完了するまでIssue全体の完了とはしない。
