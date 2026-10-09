# Issue #47 — 運用文書・agent/IDE手順の現行契約

対象: [Issue #47](https://github.com/shonen0129/NightManager/issues/47)（F26 / F27）。
検証日: 2026-10-08 JST。

日次手順を持越し・監査・実行gateの実契約へ揃え、文書の具体的なコード参照と意味の対応をCIへ追加した。本番モデル・設定・発注処理は変更していない。研究の新規採否、Phase全体の完了、本番昇格、実発注はこの作業の対象外。

## 完了条件への対応

| 条件 | 対応・確認 |
|---|---|
| 持越し・残在庫・照合/復旧 | 日次手順§1.3/§3を更新。通常closeはlong0.75/short0.50を保持し、単元丸め・未約定残を区別する。held_overnightは予定数量、net0/モデルflatは口座全解消ではない。全解消は承認範囲と専用carry0設定を要し、各建玉と未解決注文を照合する |
| V2監査・停止結果・保証外 | 日次手順§2の対応表を実leakage/numerical/fallbackへ具体化。日差・PIT・gap読込フラグの検査と実時計/計算窓の保証外を区別。数値再監査PASSED、leakage FAILED保持、取得fallback時FLATをAudit case表と実監査テストで照合する |
| 技術仕様の現行map | 既に修正されていたProductionV2Model/V2 bridge/BT mapを確認し、公開入口をpath.py::symbolで記載。§3.1を継承解決後の属性表へ更新。BLPX504/120、vol_adjusted_target=false、現行overlay/MinVar等を照合。offline寄付proxyとactual-liveの当日09:10必須条件を分離する |
| leak-audit / experiment-design / debugging / Devin | 着手時点で主要修正済み。両FAILED経路のflat化、helperの明示trials保持・study_id、market_data_cacheの正規参照、Devinの正本Skill参照を現行コードと照合。4 Skillのfrontmatter検査も成功。重複修正はしていない |
| AGENTSのrunner・履歴分離 | 着手時点でfeatures/固定regressionを含む説明へ修正済み。10-process runnerの実際の列挙と照合。新規directory/regression追加時の漏れ確認を維持。旧Windsurf案の成績/期待効果は[歴史report](legacy_model_improvement_plan.md)へ保存し、Skillへ蓄積しない |
| IDE正本参照・参照検査 | WindsurfのBT/daily/shadow/syntax/walkforward/planを短い正本参照へ更新。廃止済みwrapper/test/tool、旧config、V1 fallbackの指示を除去。run-tests/Devinの既修正は保持。current文書をglobで発見し、通常prose/backtick/fenced commandの具体的path・宣言symbolをASTで検査。歴史reports/ADRへ現在のAPI適合を強制しない |
| 意味の検証とCI | Config attribute表と継承解決AppConfig、Audit case表と実監査、carry表とcloseの単元丸めをオフライン検証。CIに--currentを追加。既存actual-live gate・照合/復旧テストは全testsで検証する |

設計理由は [文書契約ADR](../../docs/decisions/2026-10-08-maintained-document-contracts.md)、現行検査範囲は [CI手順](../../docs/CI.md)。AGENTS/Skillを実績台帳へ変更していない。

## 検証

- 文書validatorと運用契約の対象回帰: 13 passed。初回のテストfixtureはnumpy文字列が日時入力として不適格で1 failedとなり、datetime64の有効PIT日へ修正して再検証した。実監査はモックしていない。
- current対象の相対参照・具体的path/宣言symbol検査、Skill frontmatter検査: 成功。不存在path、公開methodの削除、履歴と現行の区別、placeholderとroot外参照、Skill追加時の発見に回帰ケースを追加した。
- 必須5 treeのcompileall、CI指定production/tests/maintained toolsとresearch両treeのRuff、変更validatorのRuff: 成功。
- import-linter: 7契約成功。scheduled Python import境界: 成功。
- wheel:一時build rootでno-deps/no-build-isolationのofflineビルド、source module manifest/research除外、隔離インストール後のCLI・ML artifact推論・ADR producer・shared BLPXのsmoke成功。
- 固定baselineは1 passed（単独実行）、残り全testsは1,171 passed / 2 warnings（4 worker、148秒）。合計1,172 passed / 2 warnings。JUnit証跡は同directoryの `tests.xml`。
- 最終差分でCI対象Ruff・compileall・current文書reference validatorを再実行し成功。import-linter 7契約とscheduled Python import境界も成功。`git diff --check` 成功。

既存`.venv/bin/python`を優先し、xdist/検証toolが必要なチェックには直前作業で作られた `/tmp/nightmanager-issue-env`（system-site-packagesを利用する一時環境）を再利用した。新たな依存の追加はしていない。全testsはbaselineを先に単独実行し、残りを4 workerで実行。外側に1800秒、TERM後10秒の猶予を設定し、pytestにも1テスト300秒の期限を設定した。wheel/静的検査にも外側期限を設定した。

mypyは既存4件（market_calendar 2件、gap_adjustment 1件、model_meta 1件）でexit1。直前の同環境baselineと同じで、今回のproduction source/config差分は0。mypy全成功とは扱わない。uvがローカル環境にないためuv sync/lock --checkとHosted CIの新規実行は未実施。ローカルwheelはHosted CIのuv isolated buildの完全な再現ではない。lockfile環境でのHosted quality-and-testsを最終merge gateとする。

文書検査は静的参照と明示した表の契約までであり、自由文全体・計算窓全体の非リーク・provider公表時刻・実約定完了を証明しない。実broker呼出、本番cache更新、scheduler変更、再送は実施していない。実運用受入や口座risk producer等の範囲外の残件を、このIssueの文書修正で完了扱いしない。
