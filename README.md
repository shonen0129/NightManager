# 日米ラグ・ファンド

US セクター ETF から JP TOPIX-17 セクター ETF の翌営業日 9:10→大引けリターンを予測する日次マーケットニュートラル戦略。

## Setup

Python 3.12 と uv を使い、lockfile に固定した開発・CI依存を導入します。

```bash
uv sync --locked --extra dev --extra ml-overlay
```

## Daily operation

```bash
# 朝（09:15 前）= decision、以降 = close を自動実行
python3 -m leadlag.cli daily --config configs/production/production.yaml

# 日次本番実行
python3 -m leadlag.cli decision --trade-date latest --api-enable
```

本番実行前の安全条件、認証、fail-closed時の扱いは [日次運用手順書](docs/日次運用手順書.md) を参照してください。

## Backtest

```bash
python3 -m leadlag.cli backtest --start-date 2015-01-05
```

## Tests and CI

GitHub CI は Python 3.12 と uv.lock を使い、compileall、Ruff、mypy、import-linter、文書リンク・パス検査、production wheel の分離検査と smoke、回帰 baseline、tests 全体を実行します。required check 名と実行方法は [docs/CI.md](docs/CI.md) を参照してください。

```bash
uv run --locked python -m compileall -q src/leadlag tests tools scripts src/research
uv run --locked python -m pytest tests/regression/test_v2_baseline.py
uv run --locked python -m pytest tests --ignore=tests/regression/test_v2_baseline.py -n auto
```

[Architecture](docs/ARCHITECTURE.md) と [AGENTS.md](AGENTS.md) に現行構造・不変条件を記載しています。過去Phaseの記録は [docs/history.md](docs/history.md)、現在の未解決作業は [GitHub tracker #22](https://github.com/shonen0129/NightManager/issues/22) が正本です。
