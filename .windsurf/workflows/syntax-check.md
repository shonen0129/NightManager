---
description: Pythonファイルの構文チェックを ruff で実行し、CLIスタックを防止する
---

# 構文チェック（ruff 実行）

CLIで `python3 -c "..."` を実行するとスタックする傾向があるため、`ruff` コマンドを使用すること。

## 手順

1. 仮想環境に `ruff` が含まれていることを確認（`pyproject.toml` の `dev` 依存に記載）
2. 以下のいずれかで実行:

```bash
# システムの python3 を使う場合
python3 -m ruff check src/

# uv 環境の場合
uv run ruff check src/
```

3. `All checks passed!` を確認

## 注意事項

- `python3 -c "..."` は長いコードの場合スタックしやすいので避ける
- 設定は `pyproject.toml` の `[tool.ruff]` / `[tool.ruff.lint]` に集約
- 以前の `_check_syntax.py` は廃止した。AST 構文確認が必要な場合は `python3 -m compileall src/ research/` でも可能
