#!/usr/bin/env python
"""Syntax check for all Python files under scripts/ and src/experiments/."""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

dirs = [
    ROOT / "scripts" / "backtest",
    ROOT / "scripts" / "experiments",
    ROOT / "src" / "experiments",
]

files = []
for d in dirs:
    if d.exists():
        files.extend(sorted(d.glob("*.py")))

ok = 0
for f in files:
    try:
        ast.parse(f.read_text())
        ok += 1
    except SyntaxError as e:
        print(f"ERR {f}: {e}")

print(f"{ok}/{len(files)} OK")
sys.exit(0 if ok == len(files) else 1)
