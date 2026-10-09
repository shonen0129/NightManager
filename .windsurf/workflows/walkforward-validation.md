---
description: 事前設計したウォークフォワードOOSを評価する
---

# ウォークフォワード検証

共通規約は [AGENTS.md](../../AGENTS.md)。長時間実行には [hang-prevention](../../.agents/skills/hang-prevention/SKILL.md) のプロセス全体の停止期限を設定する。

[experiment-design](../../.agents/skills/experiment-design/SKILL.md) を正本として、仮説・候補群・区間・purge/embargo・採否基準を実験前に定める。既存registryと [不採用索引](../../docs/experiment_graveyard.md) を検索し、未登録試行もDSRに含める。

各評価日の前に利用可能なデータだけで学習・選択する。全期間BTの事後分割は、パラメータをISだけで決めた場合を除き参考値であり真のOOSと呼ばない。CLI `backtest` にIS/OOS分割フラグがあると仮定しない。

結果は [backtest-report](../../.agents/skills/backtest-report/SKILL.md) に従って `reports/`、不採用索引は `docs/experiment_graveyard.md` に残す。Skillへ成績や完了履歴を追記しない。
