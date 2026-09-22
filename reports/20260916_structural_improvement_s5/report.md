# S5a / S5b 実施報告

実施日: 2026-09-16  
対象: `reports/20260915_structural_improvement/plan.md` の S5a（永続台帳・復旧・排他）と S5b（batch正本化・deadline・引け後照合）

## 判定

S5a と S5b のコード実装を完了した。brokerとの原子的 transaction や exactly-once 発注は
主張せず、送信後の状態が不明な場合は再送を止めて照合候補として残す設計にした。
実口座API、実注文、実schedulerの登録操作は行っていない。

## S5a: 永続台帳・復旧・排他

### 実装

- `src/leadlag/execution/state_store.py`
  - schema version 1 を明示した SQLite WAL の `execution_runs`、`order_intents`、`order_observations`、
    `execution_reconciliations`、`execution_leases` を追加。
  - run状態を `prepared → executing → completed` または
    `reconciliation_required` として保持し、送信前の失敗だけを
    `failed_before_submission` として再開可能にした。
  - `ExecutionPlan` の全注文意図を最初のbroker呼出前に commitする。
  - broker order ID・status・filled/remaining quantityを観測として追記し、
    同じ結果snapshotの再記録は observation key で冪等化する。
  - 約定・建玉・余力・journalの照合checkpointを追記し、
    `executing` と `reconciliation_required` を復旧候補として列挙する。
  - `(account, strategy, trade_date, job_type)` を job key とし、完了済み・実行中・
    照合待ちrunの同一再送を拒否する。
  - `live:production_v2` の lease を decision / close で共有する。

- `src/leadlag/execution/broker_ops.py`
  - 既存の新規・返済注文経路へstate storeを接続し、送信前intent、結果観測、
    submission時点のreconciliationを保存する。各batchのbroker応答はpoll前に
    台帳へ保存し、保存に失敗した場合は後続batchを送信しない。

- `src/leadlag/execution/post_decision.py`
  - fill price、position、wallet、daily journalの完了後に最終checkpointを保存する。
  - journal照合に失敗した場合は、完了済みrunも `reconciliation_required` へ戻す。

- `src/leadlag/execution/close.py`
  - 引け注文にも同じplan・観測・checkpoint・復旧候補の契約を接続する。

- `src/leadlag/config/paths.py`
  - state DBの正本を `var/live/pipeline_data/execution/execution_state.sqlite` に固定。

### 復旧規則

プロセスが送信後に停止した場合、runは `executing` のまま残る。再起動した同一jobは
state storeで拒否され、`list_recovery_candidates()` で注文・約定・建玉を照合する対象を
取得できる。照合が完了するまで自動再送しない。leaseの期限切れだけで再送を許可しない。

## S5b: batch・deadline・引け後照合

- `src/leadlag/execution/job_guard.py` を追加した。
  - process group全体を `start_new_session` で起動し、monotonic deadlineを監視。
  - 期限超過時は TERM、猶予後に KILL、exit code 124 と guard JSONを保存。
  - lease競合は exit code 73 とし、child commandを起動しない。
  - retryは行わず、S5aの照合を要求する。
- `scripts/batch/run_decision_v2.sh` をdecisionのmacOS正本入口とし、
  gap生成を含む全体を `job_guard` で監視する。
- `scripts/batch/run_close_positions.sh` と `run_gap_distribution.sh` も同じ期限・
  process-group guardを使用する。decisionとcloseは `live:production_v2` leaseを共有する。
- 旧 `scripts/batch/run_decision.sh` は撤去し、scheduler・手順書・Architectureの参照を
  `run_decision_v2.sh` へ切り替えた。plistはもともとv2入口を参照している。
- close CLIは、注文結果に加えてposition/wallet/journalを保存し、照合エラー時は
  summaryを保持して非成功終了する。

## 実装漏れの確認

| 確認項目 | 結果 |
|---|---|
| 送信前に注文意図が永続化される | 実装・テスト済み |
| 送信後のbroker観測が追記される | 各batchのpoll前保存を実装・テスト済み |
| process停止後の再送をブロックする | `executing` / `reconciliation_required` で実装・テスト済み |
| decisionとcloseの同一口座・戦略排他 | 共通leaseで実装・テスト済み |
| 期限超過で子process groupを停止する | TERM→KILLの実装・テスト済み |
| 引け後の建玉・余力・journal照合 | 既存処理をstate checkpointへ接続済み |
| 旧batch入口の参照 | v2へ切替、旧macOS入口を撤去 |
| 実schedulerへの登録反映 | 未実施（ローカルコード変更の範囲外。plist適用後に運用環境で確認する） |

コード上のS5a/S5b実装漏れは確認されなかった。実schedulerの登録状態と実口座での
照合は、発注権限を伴う運用作業として別途確認が必要である。

## 検証

| 検証 | 結果 |
|---|---|
| S5対象回帰・関連回帰 | **92 passed** |
| S5追加テスト（state store / job guard） | **7 passed**（上記に含む） |
| 変更対象 Ruff | PASS |
| 変更対象 mypy | PASS（6 source files） |
| `compileall src/leadlag tests tools scripts src/research` | PASS |
| import-linter | **4 contracts kept / 0 broken**（174 files / 444 dependencies） |
| batch `bash -n` | PASS |
| `git diff --check` | PASS |
| `tests/` 全体 | **705 passed, 17 warnings, 377.53秒** |

全体テストは `job_guard` により process group の1800秒期限・10秒猶予を付けて実行し、
guardのexit codeは0だった。警告は既存研究fixtureの定数入力・ゼロ除算に関するもので、
S5変更による失敗ではない。
