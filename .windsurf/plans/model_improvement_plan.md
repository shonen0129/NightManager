# モデル改善計画の入口

不変条件・研究配置・本番昇格の範囲は [AGENTS.md](../../AGENTS.md) を正本とする。

- 現行モデル・継承設定・日次/BT経路の特定は [leadlag-fund-improvement](../../.agents/skills/leadlag-fund-improvement/SKILL.md)。
- 仮説・採用基準・試行予算を事前に固定し、[experiment-design](../../.agents/skills/experiment-design/SKILL.md) に従ってOOS・感度分析・DSRを設計する。
- 研究コードは `src/research/scripts/experiments/` / `src/research/experiments/`、研究設定は `configs/research/` に置く。研究採用だけで本番設定を更新しない。
- 結果は [backtest-report](../../.agents/skills/backtest-report/SKILL.md) に従ってreportsへ、不採用索引は `docs/experiment_graveyard.md` へ記録する。

[旧案](../../reports/20261008_issue47/legacy_model_improvement_plan.md) は歴史資料として保存する。旧class・旧config・成績・未検証の期待効果を現行実装や達成済みPhaseとして扱わない。構造Phaseの進捗を扱う場合は `docs/refactor_roadmap.md` と対象ADRも照合する。
