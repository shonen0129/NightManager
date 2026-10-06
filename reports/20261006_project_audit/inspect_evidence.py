"""Summarize retained local evidence without revealing broker credentials."""
from __future__ import annotations

import ast
import collections
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    out = {}
    findings = json.loads((OUT / "ruff_all.log").read_text())
    out["ruff"] = {
        "total": len(findings),
        "codes": dict(collections.Counter(f["code"] for f in findings)),
        "trees": dict(collections.Counter(str(Path(f["filename"]).relative_to(ROOT)).split("/")[0] for f in findings)),
        "undefined_names": [{"path": str(Path(f["filename"]).relative_to(ROOT)), "line": f["location"]["row"], "message": f["message"]} for f in findings if f["code"] == "F821"],
    }
    inventory = json.loads((OUT / "inventory.json").read_text())
    out["exact_ast_clones"] = inventory["exact_ast_clone_groups"]
    prod_functions = []
    for path in (ROOT / "src/leadlag").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                prod_functions.append({"path": str(path.relative_to(ROOT)), "line": node.lineno, "name": node.name,
                                       "lines": node.end_lineno - node.lineno + 1,
                                       "branches": sum(isinstance(n, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.IfExp, ast.Match)) for n in ast.walk(node))})
    out["largest_production_functions"] = sorted(prod_functions, key=lambda row: row["lines"], reverse=True)[:20]
    out["long_production_function_count"] = sum(f["lines"] >= 100 for f in prod_functions)
    out["harness_refs"] = []
    for path in [ROOT / "AGENTS.md", *(ROOT / ".agents").rglob("*.md")]:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            for match in re.finditer(r"`((?:src|tests|tools|scripts|docs|configs)/[^`]+)`", line):
                reference = match.group(1).split("::")[0]
                if "*" in reference or "{" in reference or " " in reference:
                    continue
                if not (ROOT / reference).exists():
                    out["harness_refs"].append({"path": str(path.relative_to(ROOT)), "line": number, "missing": reference})
    out["capture_records"] = []
    for path in sorted((ROOT / "var/shadow_runs/ml_overlay_value/microstructure").glob("capture_*.json")):
        raw = json.loads(path.read_text())
        attempts = raw.get("attempts", [])
        out["capture_records"].append({"path": str(path.relative_to(ROOT)),
            **{k: raw[k] for k in ("trade_date", "status", "observed_count", "lob_count", "window_valid") if k in raw},
            "attempt_count": len(attempts),
            "error_types": sorted({str(a.get("error_type")) for a in attempts}),
            "disclosure_unread_flag_observed": any("sKinsyouhouMidokuFlg=1" in str(a.get("error")) for a in attempts)})
    out["forward_files"] = {name: {"exists": (ROOT / "var/shadow_runs/ml_overlay_value" / name).exists(),
                                  "lines": len((ROOT / "var/shadow_runs/ml_overlay_value" / name).read_text().splitlines()) if (ROOT / "var/shadow_runs/ml_overlay_value" / name).exists() else 0}
                            for name in ("daily.jsonl", "outcomes.jsonl")}
    out["frozen_quotes"] = len(list((ROOT / "var/shadow_runs/ml_overlay_value/microstructure").glob("frozen_*.json")))
    out["account_risk_producer"] = {"snapshot_exists": (ROOT / "var/live/pipeline_data/account_risk/latest.json").exists()}
    current_root = ROOT / "models/ml_order_overlay/production_20260923"
    version = (current_root / "CURRENT").read_text().strip()
    raw = json.loads((current_root / "versions" / version / "metadata.json").read_text())
    out["overlay_metadata"] = {"CURRENT": version, **raw}
    out["sqlite"] = {}
    for path in [ROOT / "var/live/pipeline_data/gap_adjusted_distribution/gap_store.sqlite", ROOT / "var/live/pipeline_data/execution/execution_state.sqlite"]:
        if not path.exists():
            continue
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5)
        tables = [r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        summary = {}
        for table in tables:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
                continue
            columns = [row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')]
            info = {"columns": columns, "rows": connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]}
            for column in ("trade_date", "date", "status"):
                if column in columns:
                    if column == "status":
                        info[column] = dict(connection.execute(f'SELECT "{column}", COUNT(*) FROM "{table}" GROUP BY "{column}"'))
                    else:
                        info[column + "_range"] = list(connection.execute(f'SELECT MIN("{column}"), MAX("{column}") FROM "{table}"').fetchone())
            summary[table] = info
        connection.close()
        out["sqlite"][str(path.relative_to(ROOT))] = summary
    (OUT / "evidence.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    print(json.dumps({k: out[k] for k in ("ruff", "harness_refs", "capture_records", "forward_files", "frozen_quotes", "account_risk_producer")}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
