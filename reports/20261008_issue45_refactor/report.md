# Issue #45 実施記録

## 範囲

`v2_bridge.run_v2_decision`、`close.close_all_positions`、`data.preprocessor.preprocess_data`、`var_history.get_hist_returns_for_risk`を、入力選択・pure calculation・計画・副作用・照合の境界に分割した。Quote/account-risk preflightを型付きrun契約にし、設計理由は[Issue #45 ADR](../../docs/decisions/2026-10-08-issue-45-run-boundaries.md)へ記録した。実口座やbrokerへの発注、live data更新は行っていない。

## 検証

| 検査 | 結果 |
|---|---|
| 変更前関連unit baseline | 71 passed |
| 変更後close回帰 | 27 passed |
| 変更後VaR history/cache回帰 | 12 passed |
| 変更後preprocessor回帰 | 8 passed |
| preflight/account-risk契約 | 16 passed |
| 全pytest（`tests/regression/test_v2_baseline.py`を除く） | 1,151 passed、159.30秒 |
| V2 baseline regression | 1 passed |
| Ruff（変更したPython module/test） | passed |
| `compileall`（src/leadlag、tests、tools、scripts、src/research） | passed |
| Operational Python import boundary | 2 entry points passed |
| Markdown参照検査（architecture、roadmap、ADR index/new ADR） | 84 references passed |
| `git diff --check` | passed |

pytestが報告した警告は、research QAの定数系列Spearman相関と既存fixtureのDataFrame fragmentationの2件。変更範囲で新しい失敗は確認されなかった。

この環境には`mypy`と`lint-imports`がないため、両検査は未実施。GitHub CIで確認する。
