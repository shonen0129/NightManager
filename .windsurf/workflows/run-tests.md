---
description: pytest でユニットテスト・統合テストを実行し、リーク監査・コンプライアンスを確認する
---

# テスト実行

コード変更後にテストを必ず通す。特にリーク防止・コンプライアンス監査は最優先。

## 手順

1. 全テスト実行（並列・推奨、約8分）:

```
// turbo
bash scripts/run_tests_parallel.sh
```

7プロセス並行実行: 重いテスト4つ（sprint0_diagnostics, sprint0_qa, sprint1::backtest, sprint1::calibration）を各1プロセスに分散し、残りをpytest-xdistで並列化。ログは `/tmp/pytest_parallel/` に出力。

2. 全テスト実行（直列・非推奨、約32分）:

```
python3 -m pytest tests/ -v
```

3. 個別実行（必要に応じて）:

```
# リーク監査テスト
python3 -m pytest tests/integration/test_leakage_audit.py -v

# 本番モデル統合テスト
python3 -m pytest tests/integration/test_production_residual_blpx.py -v

# バックテスターテスト
python3 -m pytest tests/unit/test_backtester_910.py -v
```

4. 構文チェック（CLIスタック防止）:

```
python3 _check_syntax.py
```

## 注意事項

- unit 26本 + integration テストが全て通ることを確認
- テストを弱めたり削除したりしない
- `ComplianceAuditor` の監査項目（`check_pit_binning_lookahead`, `check_residualization_leakage` 等）を無効化しない
- `python3 -c "..."` はスタックしやすいので避け、スクリプト経由で実行すること
