"""Validate this review's evidence and preserve its working-tree identity."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, timeout=30)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


r = json.loads((OUT / "reproductions.json").read_text())
p = json.loads((OUT / "prior_probes.json").read_text())
assert len(r) == 6 and len(p) == 8
assert all("probe_error" not in value for value in [*r.values(), *p.values()])
for key in ("null", "nat"):
    assert r["invalid_artifact_dates"][key]["apply_accepted"]
    assert r["invalid_artifact_dates"][key]["gross"] == 2.0
assert not r["invalid_artifact_dates"]["valid_future"]["apply_accepted"]
assert r["legacy_artifact_mix"]["loaded_cutoff"] == "2020-12-31"
assert r["legacy_artifact_mix"]["embedded_cutoff_before_load"] == "2026-08-14"
assert r["missing_distribution_metadata"]["npy_missing"]["audit"]["status"] == "PASSED"
assert r["missing_distribution_metadata"]["npy_missing"]["gross"] == 2.0
for key in ("future", "missing"):
    assert r["missing_distribution_metadata"][key]["gross"] == 0.0
fills = r["fill_failure_silent_success"]
assert fills["decision"]["returned_normally"] and fills["decision"]["detail_calls"] == 2
assert fills["decision"]["fill_statuses"] == ["FETCH_ERROR", "FETCH_ERROR"]
assert fills["close"]["cli_return_code"] == 0 and fills["close"]["detail_calls"] == 1
assert fills["close"]["fill_statuses"] == ["FETCH_ERROR"]
assert r["initial_log_failure_skips_reconciliation"]["reconciliation_calls"] == []
assert r["inactive_staging_invalidates_var_cache"]["same_loaded_version"]
assert r["inactive_staging_invalidates_var_cache"]["cache_key_changed"]
assert p["future_h3_metadata_flats"]["case_1"]["gross"] == 0
assert p["partial_failure_skips_reconciliation"]["post_failure_collection"] == ["fills", "positions", "wallet", "journal"]
assert p["cancelled_partial_not_collected"]["result"]["fill_quantity"] == 30
assert p["all_rejected_close_cli_success"]["cli_return_code"] == 2
assert p["all_rejected_close_cli_success"]["close_execution_calls"] == 1
assert p["overlay_atomic_publish_failure"]["active_version_unchanged"]
assert p["nontrading_nan_row_loses_next_return"]["True"]["dates"] == p["nontrading_nan_row_loses_next_return"]["market_sessions_only"]["dates"]
assert not p["configured_runner"]["constructs"]

test_log = (OUT / "pytest_full.log").read_text()
test_match = re.search(r"(\d+) passed, (\d+) warnings in ([\d.]+)s", test_log)
assert test_match and "WATCHDOG: exit=0" in test_log, "Full test run not successful/complete"
assert int(test_match[1]) == 605
for name in ("ruff", "mypy", "compileall", "import_linter", "prior_probes"):
    assert "WATCHDOG: exit=0" in (OUT / (name + ".stderr")).read_text(), name
assert "All checks passed!" in (OUT / "ruff.log").read_text()
assert "120 source files" in (OUT / "mypy.log").read_text()
assert "Contracts: 4 kept, 0 broken." in (OUT / "import_linter.log").read_text()
subprocess.run(["git", "diff", "--check"], cwd=ROOT, check=True, timeout=30)

changed = set(git("-c", "core.quotePath=false", "diff", "--name-only", "HEAD").splitlines())
untracked = set(git("ls-files", "--others", "--exclude-standard").splitlines())
targets = changed | {name for name in untracked if name.startswith(("src/", "tests/", "tools/", "docs/"))}
hashes = {name: sha(ROOT / name) for name in sorted(targets) if (ROOT / name).is_file()}
previous = json.loads((ROOT / "reports/20260913_stage_abc_rereview/review_manifest.json").read_text())
previous_hashes = previous["review_target_sha256"]
changed_since_previous = [name for name, digest in hashes.items() if previous_hashes.get(name) != digest]
manifest = {
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "head": git("rev-parse", "HEAD").strip(),
    "tracked_changed_file_count": len(changed),
    "source_target_sha256": hashes,
    "changed_since_previous_review_manifest": changed_since_previous,
    "evidence_sha256": {f.name: sha(f) for f in sorted(OUT.iterdir()) if f.is_file()
                        and f.name not in {"review_manifest.json", "validation.json", "review_summary.json"}},
    "notes": "Review-only; production source, config, tests and user implementation report were not edited. Hashes identify the reviewed final working tree.",
}
(OUT / "review_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
(OUT / "validation.json").write_text("{}\n")
(OUT / "review_summary.json").write_text("{}\n")
report = (OUT / "review.md").read_text()
assert "全体テストは実行中" not in report
links = re.findall(r"\]\(([^)]+)\)", report)
for link in links:
    if link.startswith("http"):
        continue
    path_text, _, line = link.partition(":")
    path = Path(path_text) if path_text.startswith("/") else OUT / path_text
    assert path.is_file(), link
    if line:
        assert 1 <= int(line) <= len(path.read_text().splitlines()), link
assert sum(line.startswith("```") for line in report.splitlines()) % 2 == 0
validation = {
    "status": "PASS", "local_links_checked": len(links),
    "new_probe_cases": len(r), "prior_probe_cases": len(p),
    "test_passed": int(test_match[1]), "test_warnings": int(test_match[2]),
    "test_elapsed_seconds": float(test_match[3]), "git_diff_check": "PASS",
    "source_hashes_recorded": len(hashes),
}
(OUT / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
summary = {
    "status": "reviewed", "verdict": "BLOCK",
    "findings": [
        {"id":"T01", "priority":"P1", "previous":"S04", "issue":"Invalid ML train_end allows application", "probe":"invalid_artifact_dates"},
        {"id":"T02", "priority":"P2", "previous":"S04", "issue":"Legacy model/metadata mismatch accepted", "probe":"legacy_artifact_mix"},
        {"id":"T03", "priority":"P2", "previous":"S01", "issue":"Metadata-free npy path bypasses provenance validation", "probe":"missing_distribution_metadata"},
        {"id":"T04", "priority":"P2", "previous":"S02/S03", "issue":"Fill-detail failure not propagated", "probe":"fill_failure_silent_success"},
        {"id":"T05", "priority":"P2", "previous":"S02", "issue":"Initial execution-log failure skips reconciliation", "probe":"initial_log_failure_skips_reconciliation"},
    ],
    "validation": validation,
    "production_artifact_migration": "pending, existing phase2_8 rejected for missing provenance",
    "live_actions": False,
}
(OUT / "review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(validation, ensure_ascii=False, indent=2))
