# 運用面改善 実施報告

## 対象

`reports/20260912_workspace_audit/report.md` の運用面指摘を対象に、発注経路の停止期限・canonical gap store 検証・実行証跡・引け後照合の可視化を実装した。既存のユーザー変更は保持し、実brokerへの発注・取消・再送とlaunchd登録変更は行っていない。

## 実装内容

### 1. 工程期限と単一実行

- `src/leadlag/execution/phase_deadline.py` を追加し、各工程を独立 process group として実行する工程期限、TERM/KILL、JSON証跡を実装。
- `run_decision_v2.sh`、`run_gap_distribution.sh`、`run_close_positions.sh`、`run_pnl_report.sh` に工程期限を追加。
- 既存の `job_guard` による全体期限・SQLite lease・重複実行防止と組み合わせ、期限超過を未発注の証拠として扱わない経路を維持。

### 2. canonical gap store の照合

- `src/leadlag/execution/gap_store_check.py` を追加。
- 本番設定が参照する `gap_store.sqlite` から対象取引日の μ・Ω・metadata・bundle manifest を同一読込で取得し、日付・horizon・storage format・SHA-256を検証。
- `run_gap_distribution.sh` の終了前に同検証を実行し、営業日の欠損・世代不一致は非ゼロ終了、休日は成功扱いとした。互換 `.npy` 出力をcanonical storeの代用にはしない。

### 3. decision / execution manifest

- `src/leadlag/execution/runtime_manifest.py` を追加。
- decisionごとに trade date、as-of、最大観測時刻、ticker順、入力fingerprint、code revision / dirty diff hash、resolved config hash、model / overlay artifact provenance、gap source/version/reason、fallback・PIT・監査・alert・weights・net/grossを `run_manifest.json.runtime` に保存。
- 発注後・close後に同じmanifestへ run ID、注文ID、fill/pending/failed、要求数量・約定数量、照合エラー、建玉の符号付き数量・net/grossを `execution` として追記。
- `results_format.py` のmanifest更新を fsync + atomic replace に変更し、部分JSONを公開しない。
- closeでは、durable checkpointで見つかった照合エラーをmanifest公開前に反映する。

### 4. 文書・回帰テスト

- `docs/SCHEDULER_SETUP.md` に工程期限、復旧前照合、canonical store、manifestの運用手順を追記。
- `tests/unit/test_operational_runtime.py` に工程timeout、manifest、数量・建玉、日付束検証の回帰ケースを追加。
- 既存のCI定義で compileall、Ruff、mypy、import-linter、文書リンク、wheel検査、全テストを実行できる状態を確認。

## 検証結果

- 関連テスト: 141 passed
- 全テスト: 879 passed / 0 failures / 0 errors / 0 skipped
- `compileall src/leadlag tests tools scripts src/research`: PASS
- Ruff（production、tests、maintained tools、ML training entry）: PASS
- mypy `src/leadlag`: PASS
- 文書リンク検証: 89 links verified

## 運用上の残課題

- 実schedulerのlaunchd登録状態、実brokerの注文・約定・建玉照合結果はこの変更では検証していない。
- 実運用へ戻す前に、dry-runで当日bundleのcanonical検証、工程JSON、runtime/execution manifest、引け後 `reconcile --pending` の一連の証跡を確認する。
- 期限超過・lease競合・未約定・照合エラー発生時の再送は自動化せず、保存台帳を照合して未処理数量だけを再計画する既存方針を維持する。
