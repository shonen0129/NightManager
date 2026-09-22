# CI と構造境界の検証

`.github/workflows/ci.yml` は Python 3.12 と `uv.lock` を使い、production
runtime と開発用検査を同じ解決結果から構築する。CI は次を順番に検査する。

1. `uv sync --locked --extra dev` と `uv lock --check`。研究用の`research`/`nonlinear`
   extraはHosted production CIの解決対象から外し、[研究環境手順](RESEARCH_ENV.md)で分離する。
2. `compileall`、Ruff、mypy、import-linter
3. architecture/ADR/plan の相対リンク検査
4. production wheel のビルド、`research`混入検査、隔離インストール後のCLI・artifact推論
5. `tests/` 全体（unit / integration / research / regression / features）

S0で固定したコード版・入力fingerprint・回帰基準は、CIの構造比較manifest artifactとして
毎回保存する。全体テストのJUnit結果も保存する。S0 manifestの保存だけでは、当該PRの
数値比較を実行したことにはならない。モデル挙動を変更するPRでは、別コード版による
before/after数値diffを追加し、比較機能・入力・除外条件を明示する。

研究ツリー全体には過去からの Ruff backlog があるため、CI の厳格な対象は
production と保守対象の研究学習入口に限定する。既存 backlog を理由に
production の新規エラーを免除しない。研究コードを変更するPRでは、変更対象を
個別に Ruff へ追加し、既存エラーと新規エラーを分けて記録する。

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
uv sync --locked --extra dev --extra nonlinear
uv lock --check
uv run --locked python -m compileall -q src/leadlag tests tools scripts src/research
uv run --locked ruff check src/leadlag tests tools/production tools/validation
uv run --locked mypy --config-file pyproject.toml src/leadlag
uv run --locked lint-imports
.venv/bin/python reports/20260912_workspace_audit/watchdog.py 1800 \
  .venv/bin/python -m pytest tests -n auto --junitxml=var/ci/tests.xml
```

長時間になるbuild・静的検査にも同じwatchdog等で全体期限を設定する。
2026-09-18のローカル実ビルド・検査結果は[実装確認報告](../reports/20260917_structural_completion_review/report.md)を参照する。
