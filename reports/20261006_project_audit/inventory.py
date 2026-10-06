"""Read-only project audit inventory; no production actions or cache updates."""
from __future__ import annotations

import ast
import collections
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from leadlag.execution.config import load_config_from_yaml


def main() -> None:
    tracked = subprocess.check_output(["git", "-c", "core.quotepath=false", "ls-files"], cwd=ROOT, text=True).splitlines()
    cfg = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
    out = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "python": sys.version,
        "tracked_counts": dict(collections.Counter(p.split("/")[0] for p in tracked)),
        "effective_config": {
            "strategy": cfg.strategy.model_dump(mode="json"),
            "risk": cfg.risk.model_dump(mode="json"),
            "v2": cfg.v2.model_dump(mode="json"),
            "gap_distribution_dir": cfg.gap_distribution_dir,
        },
    }
    files = []
    functions = []
    clones = collections.defaultdict(list)
    sources = {}
    for prefix in ("src/leadlag", "src/research", "tests", "scripts", "tools"):
        for path in sorted((ROOT / prefix).rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            rel = str(path.relative_to(ROOT))
            sources[rel] = hashlib.sha256(source.encode()).hexdigest()
            try:
                tree = ast.parse(source)
            except SyntaxError as exc:
                files.append({"path": rel, "syntax_error": str(exc)})
                continue
            nodes = list(ast.walk(tree))
            files.append({"path": rel, "lines": len(source.splitlines()),
                          "classes": sum(isinstance(n, ast.ClassDef) for n in nodes),
                          "functions": sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in nodes)})
            for node in nodes:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    lines = node.end_lineno - node.lineno + 1
                    branches = sum(isinstance(n, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.IfExp, ast.Match)) for n in ast.walk(node))
                    entry = {"path": rel, "name": node.name, "line": node.lineno, "lines": lines, "branches": branches}
                    functions.append(entry)
                    if lines >= 12:
                        body = node.body
                        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                            body = body[1:]
                        signature = ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)
                        clones[hashlib.sha256(signature.encode()).hexdigest()].append(entry)
    out["source_counts"] = {prefix: {"files": sum(f["path"].startswith(prefix + "/") for f in files),
                                     "lines": sum(f.get("lines", 0) for f in files if f["path"].startswith(prefix + "/"))}
                            for prefix in ("src/leadlag", "src/research", "tests", "scripts", "tools")}
    out["largest_files"] = sorted(files, key=lambda f: f.get("lines", 0), reverse=True)[:35]
    out["largest_functions"] = sorted(functions, key=lambda f: f["lines"], reverse=True)[:40]
    out["branchiest_functions"] = sorted(functions, key=lambda f: f["branches"], reverse=True)[:25]
    out["exact_ast_clone_groups"] = [v for v in clones.values() if len(v) > 1]
    out["harness_files"] = [str(p.relative_to(ROOT)) for p in (ROOT / ".agents").rglob("SKILL.md")]
    registry = ROOT / "var/experiments/registry.jsonl"
    if registry.exists():
        rows = [json.loads(line) for line in registry.read_text().splitlines() if line.strip()]
        out["registry"] = {"rows": len(rows), "names": dict(collections.Counter(r.get("name") for r in rows)),
            "decisions": dict(collections.Counter(r.get("decision") for r in rows)),
            "missing_study_id": sum(not r.get("study_id") for r in rows),
            "missing_record_id": sum(not r.get("record_id") for r in rows),
            "dsr_non_null": sum(r.get("metrics", {}).get("deflated_sharpe") is not None for r in rows),
            "trials_missing_or_one": sum(r.get("metrics", {}).get("trials", 1) == 1 for r in rows),
            "missing_report_paths": sorted({r["report_path"] for r in rows if r.get("report_path") and not (ROOT / r["report_path"]).exists()}),
            "record_dates": dict(collections.Counter(str(r.get("start_time", ""))[:10] for r in rows))}
    (Path(__file__).parent / "inventory.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    (Path(__file__).parent / "source_manifest.json").write_text(json.dumps(sources, indent=2))
    print(json.dumps({k: out[k] for k in ("git_head", "python", "source_counts", "largest_files", "registry")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
