---
description: 正本のcompileallとCI静的検査を実行する
---

# 構文・静的検査

共通規約は [AGENTS.md](../../AGENTS.md)。長時間実行には [hang-prevention](../../.agents/skills/hang-prevention/SKILL.md) のプロセス全体の停止期限を設定する。

検証範囲は [AGENTS.md](../../AGENTS.md) と [CI手順](../../docs/CI.md) に従う。既存 `.venv` で `python -m compileall -q src/leadlag tests tools scripts src/research` を実行し、CI指定のRuff・mypy・import契約も確認する。

`python3 -c` のインライン実行は禁止。Pythonコードが必要ならスクリプトへ保存する。Ruff成功だけで構文確認・全テスト完了を宣言しない。
