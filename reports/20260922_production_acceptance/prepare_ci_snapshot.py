"""Prepare a source-only CI branch without modifying the user's index."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import unquote

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BRANCH = "codex/structural-acceptance-20260922"


def run(*args, cwd=ROOT):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True, timeout=120).stdout


def main():
    destination = Path(tempfile.mkdtemp(prefix="leadlag-acceptance-ci-")) / "source"
    run("git", "worktree", "add", "--detach", str(destination), "HEAD")
    roots = ["src", "tests", "tools", "scripts", "configs", "docs", ".github"]
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".DS_Store", "*.egg-info", "*.pkl")
    for name in roots:
        target = destination / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(ROOT / name, target, ignore=ignored)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(ROOT / name, destination / name)
    documents = ["docs/ARCHITECTURE.md", "docs/CI.md", "docs/SCHEDULER_SETUP.md",
                 "docs/decisions/2026-09-15-structural-improvement-boundaries.md",
                 "reports/20260915_structural_improvement/plan.md"]
    required = set(documents)
    for name in documents:
        source = ROOT / name
        for raw in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", source.read_text()):
            value = raw.strip().split("#", 1)[0].split("?", 1)[0]
            if not value or "://" in value or value.startswith("mailto:"):
                continue
            path = (source.parent / unquote(value)).resolve()
            if path.is_file():
                required.add(str(path.relative_to(ROOT)))
            elif not path.exists():
                raise FileNotFoundError(path)
    required.update({"reports/20260915_structural_improvement/s0_baseline_manifest.json",
                     "reports/20260915_structural_improvement/s0_execution.json",
                     "reports/20260915_structural_improvement/s0_feature_matrix.md"})
    for name in required:
        if name.startswith(("var/", "creds/", ".env")) or Path(name).suffix in {".log", ".sqlite", ".pkl"}:
            raise ValueError(f"Private artifact requested by a CI documentation link: {name}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    run("git", "add", "-A", "--", *roots, "pyproject.toml", "uv.lock", *sorted(required), cwd=destination)
    names = run("git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z", cwd=destination).split("\0")
    secrets = [(key, str(value).encode()) for key, value in dotenv_values(ROOT / ".env").items()
               if value and len(value) >= 12 and re.search(r"PASSWORD|SECRET|TOKEN|API_KEY", key)
               and not re.search(r"PATH|FILE|URL", key)
               and not (str(value).endswith((".json", ".pem")) and (ROOT / str(value)).is_file())]
    manifest = {}
    for name in filter(None, names):
        content = (destination / name).read_bytes()
        matches = [key for key, value in secrets if value in content]
        if matches or re.search(rb"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----", content):
            raise ValueError(f"Credential scan rejected {name}; matched environment names: {matches}")
        manifest[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    run("git", "switch", "-c", BRANCH, cwd=destination)
    run("git", "commit", "-m", "Complete structural boundaries and execution-price acceptance checks", cwd=destination)
    result = {"worktree": str(destination), "branch": BRANCH,
              "commit": run("git", "rev-parse", "HEAD", cwd=destination).strip(),
              "changed_files": manifest, "credential_scan": "passed", "pushed": False}
    (OUT / "ci_snapshot_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "ci_snapshot_diffstat.txt").write_text(run("git", "show", "--stat", "--oneline", "HEAD", cwd=destination))
    print(json.dumps({key: value for key, value in result.items() if key != "changed_files"}))
    print(json.dumps({"added_or_modified_files": len(manifest), "source_bytes": sum(v["bytes"] for v in manifest.values())}))


if __name__ == "__main__":
    main()
