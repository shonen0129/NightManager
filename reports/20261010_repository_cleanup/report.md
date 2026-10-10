# 不要コード・旧文書の整理（2026-10-10）

## 範囲と判断

Git管理対象のコード・設定・Docs・テスト・スクリプト・CI・履歴レポートを静的に調査した。変更前の作業ツリーはclean。本番モジュールはimport・公開関数/クラスの参照候補を調べたが、削除を確定できる未使用モジュールは見つからなかった。動的ロードや手動実行もあり、静的参照がないことだけを削除根拠にしない。

## 整理内容

| 対象 | 整理と根拠 |
|---|---|
| scripts/migrate_production_config.py | 削除。2026年8月のネスト→flat移行用入口。コード・設定・テスト・バッチ・CIから呼び出しなし。参照は当時の移行計画のみ。現行の設定読込・正規化処理は保持 |
| src/research/scripts/experiments/check_df_exec_range.py | 削除。同じcacheを読む scripts/check_df_exec_range.py と日付範囲確認が重複。呼び出しなし。手動診断の入口は後者を保持 |
| configs/production/production.yaml.legacy.disabled | configs/archive/production_nested_legacy_20260813.yaml へ移設。本文不変・有効設定からの参照なし |
| 旧Docs 12件 | 旧CLI・モデル要約・運用方針案・研究メモ・参考資料・設計案11件を現行docsからarchiveへ移設。旧archive READMEも別名で保存し、新索引へ差替え。本文不変 |
| 現行文書・研究docstring | 新しい文書索引を追加。architecture・README・運用/技術仕様と4本の研究スクリプトの設計参照を更新。旧方針案を現行規約として参照せずAGENTS.mdへ案内 |
| docs/history.md | 同文のPhase 32記載2件を1件にし、当時のAPI・成績・完了状態と現行仕様の区別を明記 |

移設元・先・本文のSHA-256と削除ファイルのSHA-256は [cleanup_manifest.json](cleanup_manifest.json) に記録した。旧文書内の旧パス・コマンドは当時の記録として残す。歴史レポートの旧パスも改変しない。現行の入口は [docs/README.md](../../docs/README.md)、旧資料の入口は [archive/docs/README.md](../../archive/docs/README.md)。判断は [ADR](../../docs/decisions/2026-10-10-repository-cleanup.md) に記録した。

## 保持対象と限界

本番計算・監査・フォールバック・発注経路、有効設定、ティッカー定義、過去の実験コード・設定・レポート・registry、運用/市場データは保持した。`var/`、旧出力ディレクトリ、仮想環境、build/cacheは削除していない。未参照のADRや研究資料は由来・再現性が必要なため、参照件数だけでは不要と判定しない。本作業は不要入口と文書構成の整理であり、Phase全体の完了・本番受入・戦略性能の再評価ではない。

## 検証

変更前：設定読込・検証・文書検査の既存テスト26件PASS。`validate_docs.py --current` はリンク135件、現行path/symbol249件PASS。

変更後：

| 検査 | 結果 |
|---|---|
| 対象の既存テスト | [26件PASS](pytest_targeted.txt) |
| tests/全体（固定V2 baselineを含む） | [1264件PASS、警告2件](pytest_full.txt)。pytest 154.75秒、skip・失敗なし |
| 構文 | [compileall PASS](compileall.txt)：src/leadlag、tests、tools、scripts、src/research |
| lint | [Ruff PASS](ruff.txt)：CIの本番・テスト・運用tool・研究コード・文書検査パス |
| import契約 | [7契約PASS](import_linter_cached.txt)、破損cacheは再構築され0 broken |
| 運用Python入口の依存境界 | [2入口PASS](operational_imports.txt) |
| 文書 | [現行リンク・path/symbol検査PASS](docs_after.txt)。新文書索引・旧索引・新ADR・整理報告も指定して検査 |
| 移設・挙動維持 | [13件の本文一致](integrity.json)。本番ソース・有効設定は差分なし。研究4本はmodule docstringを除いたASTが変更前後で一致 |
| 型検査 | [mypy：4 errors / 3 files](mypy_cached.txt)。変更のない本番コードの既存指摘として分離。PASS扱いしない |

全テストは既存 `.venv/bin/python` で直列実行した。pytest-xdistのモジュールがないため `-n auto` は使っていない。`scripts/tools/phase_deadline.py` でプロセスグループ全体1800秒・終了猶予10秒、pytestの各ケース300秒を設定した。[deadline証跡](test_deadline.json) はcompleted / return_code=0 / cleanup=already_exited、全体経過156.051秒を記録している。

ローカル環境ではmypy/import-linterのconsole入口がある一方、対応moduleがなく [初回実行は失敗](environment_failures.txt)。実行中のテスト環境やグローバル環境には依存を追加せず、既存のuvキャッシュからlockfileと同じversionの検査tool・関連依存を読み込んだ（[mypy環境](mypy_cache_environment.json)、[import-linter環境](imports_cache_environment.json)）。本番全依存をlockfileから再構築したCI環境の代替ではない。

mypy指摘は `src/leadlag/core/market_calendar.py:111,187` の戻り値Any、`src/leadlag/core/gap_adjustment.py:195` の戻り値Any、`src/leadlag/models/blpx/model_meta.py:82` のlist要素型。型検査の対象本番ソースは全て変更前と同一であり、整理に起因する差分はない。依存環境の再構築やこれらの型修正は本整理の範囲外として残した。検証集計は [validation.json](validation.json) に保存した。
