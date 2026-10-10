# 旧文書・参考資料索引

このディレクトリは当時の記述を保存する履歴資料であり、現行の運用・実装指示ではない。現行仕様は [文書索引](../../docs/README.md) と [AGENTS.md](../../AGENTS.md) を参照する。

2026-10-10の整理では下表の旧文書を本文を変更せず移設した。旧API、V1/PCAフォールバック、移行中の互換層、当時の設定値・実験成績は現在の仕様や受入結果を保証しない。研究スクリプトが参照する設計案も、実装採用や完了を意味しない。

| 元の場所 | 保存先・内容 |
|---|---|
| docs/MODE_USAGE_GUIDE.md | [旧CLIガイド](MODE_USAGE_GUIDE.md) |
| docs/model_summary_for_improvement.md | [旧モデル要約](model_summary_for_improvement.md) |
| docs/運用方針書.md | [旧運用方針案](運用方針書.md) |
| docs/研究メモ202606.md | [2026年6月研究メモ](研究メモ202606.md) |
| docs/quants.md | [量的取引の参考資料](quants.md) |
| docs/design/A_theory_design_specs.md | [理論面の改善設計案](design/A_theory_design_specs.md) |
| docs/design/B_refactor_specs_and_memos.md | [旧コード改善仕様・メモ](design/B_refactor_specs_and_memos.md) |
| docs/design/B3_composition_refactor_proposal.md | [旧composition提案](design/B3_composition_refactor_proposal.md) |
| docs/design/B8_leak_freedom_analysis.md | [当時のルックアヘッド解析](design/B8_leak_freedom_analysis.md) |
| docs/design/C_validation_frameworks.md | [旧検証フレームワーク案](design/C_validation_frameworks.md) |
| docs/design/subsector_refinement_plan.md | [サブセクター精緻化計画](design/subsector_refinement_plan.md) |
| archive/docs/README.md | [旧PCA-Ensemble概要](pca_ensemble_overview.md) |

[運用方針書オリジナル](運用方針書オリジナル.md) も過去版として保存する。

無効化済みの旧ネスト設定は [configs/archive/production_nested_legacy_20260813.yaml](../../configs/archive/production_nested_legacy_20260813.yaml) に保存した。本番設定は `configs/production/production.yaml` とその継承先を使う。

歴史レポートに記載された旧パスは、当時の証跡として保持する。移設一覧と本文のSHA-256は [整理記録](../../reports/20261010_repository_cleanup/cleanup_manifest.json) を参照する。
