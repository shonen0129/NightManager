# 段階A〜Cレビュー指摘の修正報告 — 2026-09-13

## 結果

レビューで挙がったR01〜R08を修正した。実口座への発注・決済・外部API操作は行っていない。

| 指摘 | 対応 |
|---|---|
| R01 同日US終値のprovisional行 | `preprocessor.py`のprovisional対象をJP取引日の後日に限定し、同日US closeを日本9:10入力へ流用しないようにした。回帰テストを追加。 |
| R02 部分約定・未約定の成功扱い | Tachibana/Kabuの状態判定、通常注文・引け決済のpollingを修正。`SUBMITTED`/`PARTIALLY_FILLED`を終端まで追跡し、完了数・部分数・pending数・失敗数を分離。未完了の新規注文はlive経路で例外化し、引け決済ログとCLIは完了表示を出さない。 |
| R03 実signal日を監査へ渡さない | Gap bundle metadataの`sig_date`/`signal_date`を分布読込みからV2 leakage auditへ伝播。未来のsignal日を含むbundleは監査失敗時にflat化される回帰テストを追加。 |
| R04 GapStore readerの混在 | μ・Ω・metadataをhorizon付きの単一SQLite transactionで保存・取得する`save_horizon`/`load_horizon`を追加し、本番の`load_gap_bundle`経路とh=3/5保存を接続。 |
| R05 VaR fingerprint不足 | SQLite本体に加えてWAL/SHM、gap内容、ML artifact、解決済み設定、`src/leadlag`の内容digestをcache keyへ含めた。 |
| R06 ML in-sample適用 | `trade_date <= train_end`またはlegacy provenance artifactを例外化。Backtestは暗黙に通常成績へ混ぜずflat fallbackとして記録する。 |
| R07 DD系列の不一致 | 初期wealth=1.0を含めてからhigh-water markを計算する共有`compute_drawdown_series`を追加し、metrics・backtester・reporting chartで共通利用。 |
| R08 regression monkeypatch残留 | `monkeypatch.setattr`へ変更し、テスト終了時にmacro関数を復元。 |

決済ログは集計後に保存するようにし、`close_execution_log.json`にも`filled_orders_count`、`partial_orders_count`、`pending_orders_count`、`failed_orders_count`、`close_incomplete`を残す。これにより、受付済みでも約定未確認の注文を「完了」として記録しない。

## 検証

- 全テスト: **591 passed, 17 warnings**, 704.21秒。プロセス全体のwatchdog deadlineは1800秒。ログ: [pytest_full_final.log](./pytest_full_final.log)
- 追加対象テスト: **26 passed**。ログは標準出力で確認。
- `compileall`: PASS
- Ruff: PASS
- mypy (`src/leadlag` 120 files): PASS
- `git diff --check`: PASS
- import-linter: `.venv`にモジュールがないため未実行（`pyproject.toml`の設定は変更していない）。

設計上の安全境界は[ADR](../../docs/decisions/2026-09-13-stage-abc-safety-boundaries.md)に記録した。

## 残る運用前提

既存の`models/ml_order_overlay/phase2_8`はprovenance metadataを持たないため、production runnerは安全側に拒否する。これは今回の修正で緩めていない。provenance付きartifactを再学習し、適用可能期間を満たすartifactへ置き換えるまで、本番ML経路の稼働準備は未完了である。
