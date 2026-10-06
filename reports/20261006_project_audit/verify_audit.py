"""Verify audit references, preserved tracked inputs, and evidence summaries."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent


def main():
    text = (OUT / "report.md").read_text()
    manifest = json.loads((OUT / "audit_manifest.json").read_text())
    changed = [name for name, expected in manifest.items() if not (ROOT / name).is_file() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected]
    issues = []
    axis = {**{i: "pnl_metrics_statistics" for i in [1, 11, 12, 13, 14, 15, 16, 17, 18, 19]},
        2: "credential_boundary", **{i: "operations" for i in [3, 4, 5, 6]},
        **{i: "model_profitability_evidence" for i in [7, 8, 9, 10]},
        20: "provider_contract", **{i: "structure_quality" for i in [21, 22, 23, 24, 25]},
        26: "operational_document", 27: "harness", 28: "experiment_governance", 29: "repository_governance", 30: "data_quality", 31: "calendar"}
    matches = list(re.finditer(r"^### (F\d+) \[(P\d)/([^\]]+)\] (.+)$", text, re.MULTILINE))
    bad_refs = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else text.index("## 8. 維持")
        body = text[match.end():end]
        refs = []
        for ref in re.finditer(r"((?:src|scripts|tools|docs|\.agents|\.devin|\.windsurf)/[^`\s]+\.(?:py|md|sh)):(\d+)", body):
            name, line = ref.group(1), int(ref.group(2))
            path = ROOT / name
            if not path.exists() or line > len(path.read_text().splitlines()):
                bad_refs.append({"issue": match.group(1), "path": name, "line": line})
            refs.append({"path": name, "line": line})
        issues.append({"id": match.group(1), "priority": match.group(2), "kind": match.group(3), "title": match.group(4),
            "axis": axis[int(match.group(1)[1:])], "report_line": text[:match.start()].count("\n") + 1, "source_references": refs})
    (OUT / "findings.json").write_text(json.dumps(issues, ensure_ascii=False, indent=2))
    jobs = []
    for label in ("com.leadlag.microstructure-0910", "com.leadlag.update-market-data"):
        result = subprocess.run(["launchctl", "print", f"gui/{os.getuid()}/{label}"], capture_output=True, text=True, timeout=10)
        record = {"label": label, "query_exit": result.returncode}
        for field in ("state", "runs", "last exit code"):
            match = re.search(r"^\s*" + re.escape(field) + r"\s*=\s*([^\n]+)", result.stdout, re.MULTILINE)
            if match:
                record[field] = match.group(1).strip()
        jobs.append(record)
    probes = json.loads((OUT / "probes.json").read_text())
    json_files = [p for p in OUT.glob("*.json") if p.name != "verification.json"]
    for path in json_files:
        json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonstandard JSON: {value}")))
    docs = subprocess.run([str(ROOT / ".venv/bin/python"), "scripts/ci/validate_docs.py", str(OUT / "report.md")], cwd=ROOT, capture_output=True, text=True, timeout=30)
    out = {"tracked_manifest_files": len(manifest), "changed_or_missing_tracked_files": changed,
        "findings": len(issues), "priority_counts": {level: sum(i["priority"] == level for i in issues) for level in ("P0", "P1", "P2", "P3")},
        "invalid_line_references": bad_refs, "probe_categories": len(probes), "strict_json_files": len(json_files),
        "docs_validation": {"exit": docs.returncode, "output": docs.stdout.strip()}, "scheduler_snapshot": jobs,
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)}
    (OUT / "verification.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if changed or bad_refs or docs.returncode or len(issues) != 31:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
