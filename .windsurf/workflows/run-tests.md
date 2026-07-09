---
description: pytest でユニットテスト・統合テストを実行し、リーク監査・コンプライアンスを確認する
---

# テスト実行

コード変更後にテストを必ず通す。特にリーク防止・コンプライアンス監査は最優先。

## 手順

1. 全テスト実行:

```
python3 -m pytest tests/ -v
```

2. 個別実行（必要に応じて）:

```
# リーク監査テスト
python3 -m pytest tests/integration/test_leakage_audit.py -v

# 本番モデル統合テスト
python3 -m pytest tests/integration/test_production_residual_blpx.py -v

# バックテスターテスト
python3 -m pytest tests/unit/test_backtester_910.py -v
```

3. 構文チェック（CLIスタック防止）:

```
python3 _check_syntax.py
```

## 注意事項

- unit 26本 + integration テストが全て通ることを確認
- テストを弱めたり削除したりしない
- `ComplianceAuditor` の監査項目（`check_pit_binning_lookahead`, `check_residualization_leakage` 等）を無効化しない
- `python3 -c "..."` はスタックしやすいので避け、スクリプト経由で実行すること
