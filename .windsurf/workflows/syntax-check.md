---
description: Pythonファイルの構文チェックをスクリプト経由で実行し、CLIスタックを防止する
---

# 構文チェック（スクリプト実行）

CLIで `python3 -c "..."` を実行するとスタックする傾向があるため、必ずスクリプトファイル経由で実行すること。

## 手順

1. プロジェクトルートに `_check_syntax.py` が既に存在する場合はそのまま使用
2. 存在しない場合は作成（内容は下記テンプレート参照）
3. 以下のコマンドで実行:

```
python3 _check_syntax.py
```

4. 結果の `N/N OK` を確認

## テンプレート

```python
#!/usr/bin/env python
"""Syntax check for all Python files under scripts/ and src/experiments/."""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

dirs = [
    ROOT / "scripts" / "backtest",
    ROOT / "scripts" / "experiments",
    ROOT / "src" / "experiments",
]

files = []
for d in dirs:
    if d.exists():
        files.extend(sorted(d.glob("*.py")))

ok = 0
for f in files:
    try:
        ast.parse(f.read_text())
        ok += 1
    except SyntaxError as e:
        print(f"ERR {f}: {e}")

print(f"{ok}/{len(files)} OK")
sys.exit(0 if ok == len(files) else 1)
```

## 注意事項

- `python3 -c "..."` は長いコードの場合スタックしやすいので避ける
- 一時的なチェック用スクリプトは使い回す（毎回作成・削除しない）
- チェック対象ディレクトリを追加したい場合は `dirs` リストに追記する
