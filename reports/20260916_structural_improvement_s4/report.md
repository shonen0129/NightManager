# S4a / S4b 構造改善 実施報告

- 実施日: 2026-09-16
- 対象: `reports/20260915_structural_improvement/plan.md` の S4a / S4b
- 判定: 実装完了。対象回帰・静的検査・全体回帰 PASS。

## S4a: 意思決定から注文計画への型付き境界

### 実装

- `src/leadlag/execution/contracts.py`
  - `ExecutionPlan`を追加した。
  - 取引日、決定ID、確認済み既存建玉、返済注文、新規注文、目標net/grossをimmutableな値として保持する。
  - 決定IDはプロセス依存の`hash()`を使わず、正規化した注文内容のSHA-256から生成する。
  - 既存のJSON/CSV writerへ渡す`to_dict()`を提供する。
  - `OrderObservation`は要求数量・累積約定数量・残数量・観測時刻・raw参照を保持する。残数量はbroker記録を優先し、なければ要求数量と約定数量から導出する。
  - `ExecutionReport`と`report_from_records()`を追加し、accepted/filled/partial/failed/unresolved、close失敗、記録・照合エラーを分離する。
- `src/leadlag/execution/broker_ops.py`
  - `build_execution_plan()`を追加し、既存のdelta計算を注文計画へ変換する。
  - `submit_orders_via_api()`は互換のDataFrame引数を維持しつつ、`execution_plan`を受け取れる。
  - execution summaryに計画とtyped reportのJSON表現を保存する。
- `src/leadlag/execution/post_decision.py`
  - decision DataFrameにCSVへ出さない取引日属性を付与し、発注前に`ExecutionPlan`を構築して渡す。
  - fill/建玉/wallet/journalの照合エラーを`ExecutionReport`へ反映する。
- `src/leadlag/execution/close.py`
  - 引け処理のsummaryにも`ExecutionReport`を保存する。

旧dict/CSV/JSON形式を内部の正本にせず、既存利用者の互換境界として残した。残存建玉の実測値は既存のposition snapshot/journalを正本参照とし、S4aで注文台帳や再起動復旧まで追加していない（S5の対象）。

## S4b: 新規注文・引け決済のpoll共通化

`src/leadlag/execution/order_lifecycle.py`に`poll_order_statuses()`を追加した。broker固有の応答形式はgetter/setter adapterで吸収し、次の規則を一箇所へ集約した。

- `SUBMITTED` / `PARTIALLY_FILLED`をpendingとして扱う。
- brokerのstatus照会をprobeしてから、monotonic deadlineまでpollする。
- `NotImplementedError`ではpollを無効化し、照会例外では未解決として保持する。
- 期限後の未約定・部分約定を成功へ変換しない。

`broker_ops._wait_for_fills_sync()`と`close._wait_for_close_fills_sync()`はこの共通部品の薄いadapterになった。注文目的ごとのログラベル、timeout、結果表現は呼出側に残している。

## S4後半: `PortfolioDecision` dict互換の撤去

`PortfolioDecision`は属性専用のimmutable domain typeへ整理した。mapping protocolと`from_dict`/`to_dict`を削除し、モデル、執行、backtest、研究overlay、fallback結果の利用者を属性参照へ移した。overlayは`dataclasses.replace`で更新後の値を返す。

legacy mappingの変換は`reporting/production_v2_writer.py::_coerce_decision`だけに残した。JSON/CSV/Markdown出力の入口でのみ変換し、内部経路へmappingを戻さない。`tests/unit/test_portfolio_decision_contract.py`で属性専用契約とwriter境界の変換を固定した。

## 漏れ確認

### 実装接続

- 発注経路: `v2_bridge → post_decision → build_execution_plan → submit_orders_via_api`
- 新規・返済poll: `broker_ops` / `close`の両方が`poll_order_statuses`を使用
- summary: 発注・引けの両方が`execution_report`を出力
- 互換入口: DataFrame引数、既存order result dict、既存CSV/JSONキーを維持
- `rg`でexecution内のbroker status poll本体を確認し、残るstatus照会はbroker adapter自身と共通pollのみであることを確認

### 追加テスト

`tests/unit/test_order_lifecycle.py`で、共通pollのOrderResult/dict両表現、partial→filled遷移、ExecutionPlanの返済/新規分割、ExecutionReportの照合エラーを検証した。

## 検証

| 検証 | 結果 |
|---|---|
| S4a/S4b対象回帰＋追加テスト | **61 passed** |
| S4後半対象回帰＋追加テスト | **100 passed** |
| 変更対象 Ruff | PASS |
| 変更対象 mypy | PASS |
| 対象 compileall | PASS |
| import-linter | **4 contracts kept, 0 broken**（172 files / 435 dependencies） |
| 構造改善計画の文書検証 | PASS（相対リンク68件、ステージS0–S8確認） |
| `git diff --check` | PASS |
| 全`tests/`（S4a/S4b時点） | **696 passed, 17 warnings** |
| 全`tests/`（S4後半反映後） | **698 passed, 17 warnings** |

実口座API、実注文、実schedulerの変更は行っていない。S5の永続台帳・復旧・排他は未実施であり、今回のS4完了をもって本番昇格可とは判定しない。
