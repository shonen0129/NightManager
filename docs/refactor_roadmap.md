# Current Structure and Open Work

この文書は現行の構造と作業backlogへの入口を示す。実装状態の正本は current main のコード、継承解決後の設定、受入レポートとする。過去のPhase一覧を現在の未完了作業として扱わない。

## Current Architecture

- 本番経路は V2 同期パイプライン。入口は `src/leadlag/cli.py`、本番モデルは `src/leadlag/models/production_v2.py`。
- 本番設定は `configs/production/production.yaml` とその `__base__` 継承先。
- 本番・バックテスト共通のモデル組立は `src/leadlag/runner/model_factory.py`。
- CI は `.github/workflows/ci.yml`、現在のrequired checkと検査手順は [`docs/CI.md`](CI.md)。
- 現行ディレクトリは [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) を参照。

## Current Backlog

プロジェクト全体の残件は [GitHub tracker #22](https://github.com/shonen0129/NightManager/issues/22) が正本。個別issueで範囲と完了条件を管理する。

| Issue | Topic | Status |
|---|---|---|
| [#18](https://github.com/shonen0129/NightManager/issues/18) | main branch protection と required CI | Open |
| [#21](https://github.com/shonen0129/NightManager/issues/21) | README・architecture・roadmapの現行化 | Open |
| [#23](https://github.com/shonen0129/NightManager/issues/23) | ML overlay の事前登録済みforward評価 | Open |
| [#24](https://github.com/shonen0129/NightManager/issues/24) | VaR/ES超過の原因分析とリスク低減 | Open |
| [#25](https://github.com/shonen0129/NightManager/issues/25) | actual-account risk snapshot producer | Open |
| [#27](https://github.com/shonen0129/NightManager/issues/27) | 実取引日の運用受入 | Open |

## Historical Roadmap

旧master listと当時のPhaseメモは [`docs/history/refactor_roadmap_legacy_2026-08.md`](history/refactor_roadmap_legacy_2026-08.md) に保存した。実施済みPhaseの詳しい根拠は [構造改善計画](../reports/20260915_structural_improvement/plan.md)、[実装確認報告](../reports/20260917_structural_completion_review/report.md)、[本番受入報告](../reports/20260922_production_acceptance/report.md) を参照する。
