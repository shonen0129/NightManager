---
description: 現行運用手順で日次decisionとcloseを確認する
---

# 日次本番実行

共通規約は [AGENTS.md](../../AGENTS.md)。長時間実行には [hang-prevention](../../.agents/skills/hang-prevention/SKILL.md) のプロセス全体の停止期限を設定する。

[日次運用手順書](../../docs/日次運用手順書.md) を使い、[leadlag-fund-improvement](../../.agents/skills/leadlag-fund-improvement/SKILL.md) で入力・設定・出力を特定する。

入口はCLI `decision` / `close`、設定は `configs/production/production.yaml`。当日gap、frozen 09:10 quote、実口座risk、durable reconciliationの実際のgateを確認する。モデルflatは口座全解消を意味しない。通常closeは持越し設定に従い、約定と残建玉の照合まで確認する。

実発注・再送・全解消は依頼の承認範囲内でのみ行う。調査のために `--api-enable` を追加しない。
