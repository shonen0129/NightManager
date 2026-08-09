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

# or 従来の本番スクリプト
python3 tools/production/run_daily_production_v2.py
```

## Backtest

```bash
python3 -m leadlag.cli backtest --start-date 2015-01-05
```

## Tests

```bash
python3 _check_syntax.py
python3 -m pytest tests/unit/ -v --ignore=tests/unit/test_health_score.py
```

See `docs/ARCHITECTURE.md` for architecture and `AGENTS.md` for invariants.
