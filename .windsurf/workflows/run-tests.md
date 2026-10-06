---
description: 正本の検証規約に沿って全テストを実行する
---

# テスト実行

実行規約・対象範囲・完了判定は [AGENTS.md](../../AGENTS.md) と [test-gen Skill](../../.agents/skills/test-gen/SKILL.md) に従う。`.venv`を使い、プロセス全体のdeadlineを設定する。

対象回帰を通した後、以下のCIと同じ全体検証を行う。

```bash
timeout -k 10s 300s .venv/bin/python -m pytest tests/regression/test_v2_baseline.py
timeout -k 30s 1800s .venv/bin/python -m pytest tests --ignore=tests/regression/test_v2_baseline.py -n 4
timeout -k 10s 120s .venv/bin/python -m compileall -q src/leadlag tests tools scripts src/research
```

macOSでtimeoutがなければ [hang-prevention Skill](../../.agents/skills/hang-prevention/SKILL.md) のprocess-group guardを使う。既存の `bash scripts/run_tests_parallel.sh` は10プロセスでunit/integration/research/featuresと固定regressionを分割し、全体deadlineと `/tmp/pytest_parallel/` ログを持つ。新しいtest directoryやregression追加時に手動列挙の漏れを確認する。

失敗・未実行をPASSにせず、テスト・監査のassertionを弱めない。
