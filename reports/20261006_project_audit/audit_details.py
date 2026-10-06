"""Additional read-only coverage, harness, and artifact statistics."""
from __future__ import annotations

import ast
import collections
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent


def main():
    out = {}
    test_counts = collections.Counter()
    for name in ("baseline.xml", "tests.xml"):
        for node in ET.parse(OUT / name).iter("testcase"):
            parts = node.get("classname", "").split(".")
            category = parts[1] if len(parts) > 1 else "unknown"
            test_counts[category] += 1
    out["executed_test_counts"] = dict(test_counts)
    out["regression_test_modules"] = [str(p.relative_to(ROOT)) for p in (ROOT / "tests/regression").glob("test_*.py")]
    harness_files = [ROOT / "AGENTS.md"]
    for directory in (".agents", ".devin", ".windsurf"):
        harness_files.extend((ROOT / directory).rglob("*.md"))
    out["harness_files"] = [{"path": str(p.relative_to(ROOT)), "lines": len(p.read_text().splitlines())} for p in harness_files]
    missing = []
    for p in harness_files:
        for number, line in enumerate(p.read_text().splitlines(), 1):
            for m in re.finditer(r"`((?:src|tests|tools|scripts|docs|configs)/[^`]+)`", line):
                reference = m.group(1).split("::")[0]
                if any(c in reference for c in ("*", "{", " ", "<", ":")):
                    continue
                if not (ROOT / reference).exists():
                    missing.append({"file": str(p.relative_to(ROOT)), "line": number, "reference": reference})
    out["harness_missing_literal_paths"] = missing
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    manifest = {}
    for name in tracked:
        p = ROOT / name
        if name and p.is_file() and (name.startswith(("src/", "tests/", "tools/", "scripts/", "configs/", ".agents/", ".devin/", ".windsurf/", ".github/", "docs/")) or name in ("AGENTS.md", "README.md", "pyproject.toml", "uv.lock")):
            manifest[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT / "audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    out["manifest_files"] = len(manifest)
    shell_files = [p for p in (ROOT / "scripts").rglob("*.sh")]
    out["shell_deadline_markers"] = [{"path": str(p.relative_to(ROOT)), "deadline_marker_present": bool(re.search(r"job_guard|phase_deadline|watchdog|timeout|gtimeout", p.read_text()))} for p in shell_files]
    out["production_ast"] = {"files": 0, "long_functions": 0}
    for p in (ROOT / "src/leadlag").rglob("*.py"):
        out["production_ast"]["files"] += 1
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.end_lineno - node.lineno + 1 >= 100:
                out["production_ast"]["long_functions"] += 1
    (OUT / "audit_details.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
