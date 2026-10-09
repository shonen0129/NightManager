---
description: 本番設定でV2バックテストを検証する
---

# バックテスト

共通規約は [AGENTS.md](../../AGENTS.md)。長時間実行には [hang-prevention](../../.agents/skills/hang-prevention/SKILL.md) のプロセス全体の停止期限を設定する。

[leadlag-fund-improvement](../../.agents/skills/leadlag-fund-improvement/SKILL.md) で現行経路・入力・継承設定を特定してV2評価を行い、[backtest-report](../../.agents/skills/backtest-report/SKILL.md) で証跡を残す。性能比較を伴う場合は [experiment-design](../../.agents/skills/experiment-design/SKILL.md) に従う。

公式入口はCLI `backtest`、本番正本は `configs/production/production.yaml`。評価開始は2015-01-05以降、固定2010–2014 priorと分離する。全評価日、コスト内訳、持越しとモデル/実効exposureを報告する。
