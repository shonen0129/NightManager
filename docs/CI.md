# CI と構造境界の検証

`.github/workflows/ci.yml` は Python 3.12 と `uv.lock` を使い、production
runtime と開発用検査を同じ解決結果から構築する。CI は次を順番に検査する。

1. `uv sync --locked --extra dev --extra ml-overlay` と `uv lock --check`。`ml-overlay` は
   LightGBM境界テストだけを満たし、研究用の`research` extraに含まれる
   SHAP/llvmlite依存はHosted production CIの解決対象から外す。[研究環境手順](RESEARCH_ENV.md)
   で研究依存を分離する。
2. `compileall`、Ruff、mypy、import-linter、および `scripts/ci/check_operational_imports.py` によるscheduled Python batch入口のresearch import禁止。shellが起動する研究診断toolの分離は別の残件。
3. architecture/ADR/plan の相対リンク検査と `scripts/ci/validate_docs.py --current`。
   `--current` はAGENTS、README/architecture/CI/scheduler/research環境/roadmap、運用/技術仕様、`.agents/skills/`、Devin Skill、Windsurf workflow/planを対象に、通常のprose/backtick/fenced commandの具体的なrepository pathと `path.py::Class.method` をASTで照合する。runtime出力・placeholder・globは静的存在検査の対象外。履歴reports/ADRには現在のAPI適合を強制しない。現行文書に残す歴史節は `<!-- docs:historical -->` / `<!-- docs:current -->` で範囲を明示する。
   `tests/unit/test_operational_doc_contracts.py` は仕様のConfig attribute表を継承解決後の設定へ、運用Audit case表を実監査のフラット化・statusへ対応付ける。closeの単元丸めと予定残数量も検査する。実時計・計算窓・口座の約定完了までは文書検査で保証しない。
4. production wheel のビルド、`research`混入検査、隔離インストール後のCLI・artifact推論
5. `tests/` 全体（unit / integration / research / regression / features）。回帰baselineは
   並列worker間の状態影響を受けないよう単独で実行し、残りを並列化する。

S0で固定したコード版・入力fingerprint・回帰基準は、CIの構造比較manifest artifactとして
毎回保存する。全体テストのJUnit結果も保存する。S0 manifestの保存だけでは、当該PRの
数値比較を実行したことにはならない。モデル挙動を変更するPRでは、別コード版による
before/after数値diffを追加し、比較機能・入力・除外条件を明示する。

Ruff の対象は `src/leadlag tests tools/production tools/validation scripts/ci/validate_docs.py` と
`src/research tools/research`。研究ツリーの既存 backlog を整理したため、学習入口だけの
個別検査から両研究ツリー全体へゲートを拡大する。新規エラーの免除は追加しない。

wheel検査は `scripts/ci/verify_wheel.py` が行う。本番CLIで使う推論依存は
production側に残し、学習・実験用の `src/research/` はwheelへ含めない。さらにsource配下の
Python module manifestと照合し、削除済みmoduleの混入とproduction moduleの欠落を検出する。
wheelは `scripts/ci/build_wheel_clean.py` でworktree外の一時buildディレクトリへ生成し、
反復ビルドで削除済みモジュールが古い `build/lib` から混入しないようにする。
`scripts/ci/smoke_installed_wheel.py`は既存の依存環境を使い、repositoryのeditable sourceを
import pathから除外する。wheelからCLIを起動し、合成のversioned `MLOrderOverlayModel`
artifactを保存・読込して推論値の一致を検査する。本番artifact再生成や本番/BT一致は別の検証である。

Sprint診断テストは`tests/research/conftest.py`の固定市場入力と一時診断CSVを使う。
外部取得・live cache読込だけを差し替え、モデル・診断計算と校正の非空結果を検査する。
ローカルcacheがある環境だけで成功するテストをCIの受入証拠にしない。

ローカルでCI相当の検査を行う場合（`uv` が利用できる環境）は次を実行する。

```bash
uv sync --locked --extra dev --extra ml-overlay
uv lock --check
uv run --locked python -m compileall -q src/leadlag tests tools scripts src/research
uv run --locked ruff check src/leadlag tests tools/production tools/validation scripts/ci/validate_docs.py
uv run --locked ruff check src/research tools/research
uv run --locked mypy --config-file pyproject.toml src/leadlag
uv run --locked lint-imports
uv run --locked python -m pytest tests/regression/test_v2_baseline.py
.venv/bin/python scripts/tools/phase_deadline.py --label ci_tests --timeout 1800 --grace 10 -- \
  .venv/bin/python -m pytest tests --ignore=tests/regression/test_v2_baseline.py -n auto --junitxml=var/ci/tests.xml
```

長時間になるbuild・静的検査にも同じwatchdog等で全体期限を設定する。
2026-09-18のローカル実ビルド・検査結果は[実装確認報告](../reports/20260917_structural_completion_review/report.md)を参照する。

## main の required check

Current workflow の表示名は `leadlag-ci`、required job/check 名は `quality-and-tests`（GitHub の branch protection 設定では `quality-and-tests`、workflow `leadlag-ci` として表示）です。main の branch protection または repository ruleset では、pull request を必須にし、このcheckの成功をmerge条件にします。direct pushとbypassは許可せず、administratorも例外にしません。

`.github/workflows/ci.yml` のworkflow名または `quality-and-tests` job ID を変更する場合は、同じ変更でGitHub側required check selectorと本節を確認・更新します。設定後はGitHubのbranch protection/ruleset APIまたはSettings画面で、mainへの適用対象とrequired checkを読み取り確認します。ローカルのCI成功だけではGitHub側の保護設定を証明しません。

2026-10-08 の [main branch API](https://api.github.com/repos/shonen0129/NightManager/branches/main)
で `protected=true`、`quality-and-tests` 必須、`enforcement_level=everyone`、
check の `app_id=15368`（GitHub Actions）を確認した。管理設定はユーザーが実施した。
このbranch APIの応答はPR必須・strict・bypass対象の全設定を列挙しない。
同日にユーザーがSettings画面でPR必須・up-to-date・bypass禁止がON、
approval要求がOFF、bypass対象なしとして保存済みであることを確認した。

wheel smokeはtemporaryなdeployment rootを `LEADLAG_RUNTIME_ROOT` に指定し、installed packageのcode位置から独立したADR/macro/相対model/varの解決を検査する。運用配置ではimport前に同変数へ既存の絶対directoryを指定し、そのroot配下へconfig/model/dataを配置する。詳しくは [runtime境界ADR](decisions/2026-10-06-audit-boundaries.md) を参照。
