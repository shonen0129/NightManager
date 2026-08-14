# 日米ラグ・ファンド

US セクター ETF から JP TOPIX-17 セクター ETF の翌営業日 9:10→大引けリターンを予測する日次マーケットニュートラル戦略。

## Setup

```bash
uv sync
# or
pip install -e .
```

## Daily operation

```bash
# 朝（09:15 前）= decision、以降 = close を自動実行
python3 -m leadlag.cli daily --config configs/production/production.yaml

# 日次本番実行
python3 -m leadlag.cli decision --trade-date latest --api-enable
```

## Backtest

```bash
python3 -m leadlag.cli backtest --start-date 2015-01-05
```

## Tests

```bash
python3 -m compileall src/leadlag tests tools scripts src/research
python3 -m pytest tests/unit/ -v
```

See `docs/ARCHITECTURE.md` for architecture and `AGENTS.md` for invariants.
