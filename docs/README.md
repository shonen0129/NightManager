# 文書索引

現行の仕様・手順は以下を入口にする。挙動はコード・継承解決後の設定・テストで確認し、共通の不変条件は [AGENTS.md](../AGENTS.md) に従う。

| 用途 | 文書 |
|---|---|
| セットアップ・主要CLI | [プロジェクトREADME](../README.md) |
| 構造・責務・本番経路 | [ARCHITECTURE.md](ARCHITECTURE.md) |
| 数式・パラメータ・モデル契約 | [モデル技術仕様書](モデル技術仕様書.md) |
| 日次実行・監査・照合・復旧 | [日次運用手順書](日次運用手順書.md) |
| スケジューラ | [SCHEDULER_SETUP.md](SCHEDULER_SETUP.md) |
| 検証・CI | [CI.md](CI.md) |
| 研究環境 | [RESEARCH_ENV.md](RESEARCH_ENV.md) |
| 設計判断 | [ADR索引](decisions/README.md) |
| 現在の構造・作業backlog | [refactor_roadmap.md](refactor_roadmap.md) |
| 停止・ハングの診断 | [スタック再発防止策](スタック再発防止策.md) |
| 出力ディレクトリ・移行手順 | [var_convention.md](var_convention.md) |
| API仕様 | [kabu STATION](api/kabu_STATION_API.yaml) / [立花証券](api/立花証券API.md) |

実験・受入の証跡は [reports](../reports/README.md)、不採用実験の入口は [experiment_graveyard.md](experiment_graveyard.md) を参照する。

過去の実装記録は [history.md](history.md)、旧ロードマップは [history/](history/refactor_roadmap_legacy_2026-08.md)、旧CLI・モデル要約・運用方針案・設計案・研究メモは [旧文書索引](../archive/docs/README.md) に保存している。履歴資料のコマンド・設定値・監査結論・完了状態を現在の仕様として使わない。
