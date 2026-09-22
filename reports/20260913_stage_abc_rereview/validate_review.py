"""Validate local review evidence and record the reviewed working-tree identity."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, timeout=30)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


reproductions = json.loads((OUT / "reproductions.json").read_text())
assert len(reproductions) == 8
assert all("probe_error" not in row for row in reproductions.values())
mh = reproductions["future_h3_metadata_not_audited"]
assert mh["case_1"]["audit"]["status"] == "PASSED"
assert mh["case_2"]["audit"]["status"] == "PASSED"
assert mh["max_score_change"] > 3
assert mh["case_1"]["gross"] == 2.0
for row in reproductions["atomic_reader_passes_all_horizons"].values():
    assert row["interleaved_commit"]
    assert row["mu"] == row["omega"] == 1.0
    assert row["meta"]["version"] == "old" and row["latest_mu"] == 2.0
holiday = reproductions["nontrading_nan_row_loses_next_return"]
assert not holiday["False"]["contains_20260813"]
assert "jp_gap missing" in holiday["True"]["error"]
assert holiday["market_sessions_only"]["contains_20260813"]
partial = reproductions["partial_failure_skips_reconciliation"]
assert partial["post_failure_collection"] == []
assert partial["persisted_fill_quantities"] == [None, None]
assert reproductions["cancelled_partial_not_collected"]["calls"] == 0
assert reproductions["all_rejected_close_cli_success"]["cli_return_code"] == 0
ml = reproductions["overlay_binary_and_metadata_mix"]
assert ml["loaded_marker"] == "trained-through-2026"
assert ml["loaded_train_end"] == "2020-12-31" and ml["guard_would_accept_2021"]
assert not reproductions["configured_runner"]["constructs"]

# The first CLI probe could exit at the holiday gate. The handoff rerun pins
# that gate and proves the close execution boundary is actually reached.
handoff = json.loads((OUT / "reproductions_handoff.json").read_text())
assert len(handoff) == 8
assert all("probe_error" not in row for row in handoff.values())
close_handoff = handoff["all_rejected_close_cli_success"]
assert close_handoff["holiday_skip_disabled"]
assert close_handoff["close_execution_calls"] == 1
assert close_handoff["cli_return_code"] == 0

prior_manifest_path = OUT / "review_manifest.json"
unchanged_runtime_files = 0
if prior_manifest_path.exists():
    prior_manifest = json.loads(prior_manifest_path.read_text())
    for relative, digest in prior_manifest["review_target_sha256"].items():
        if relative.startswith(("src/", "tests/", "tools/", "configs/")):
            assert sha(ROOT / relative) == digest, f"Runtime input changed: {relative}"
            unchanged_runtime_files += 1

model_probes = json.loads((OUT / "model_probes.json").read_text())
for name, row in model_probes.items():
    assert "probe_error" not in row
    for key, value in row.items():
        if "difference" in key:
            assert value == 0.0, (name, key, value)

test_log = (OUT / "pytest_full.log").read_text()
assert "591 passed, 17 warnings in 699.38s" in test_log
assert "WATCHDOG: exit=0" in test_log
for name, marker in (("ruff.log", "All checks passed!"),
                     ("mypy.log", "Success: no issues found in 120 source files"),
                     ("import_linter.log", "Contracts: 4 kept, 0 broken."),
                     ("compileall.log", "WATCHDOG: exit=0")):
    assert marker in (OUT / name).read_text(), name
subprocess.run(["git", "diff", "--check"], cwd=ROOT, check=True, timeout=30)

changed = set(git("diff", "--name-only", "HEAD").splitlines())
untracked = set(git("ls-files", "--others", "--exclude-standard").splitlines())
targets = changed | {p for p in untracked if p.startswith(("tests/", "src/", "tools/", "docs/"))}
targets.update({"reports/20260912_workspace_audit/fix_report.md",
                "reports/20260913_stage_abc_review/fix_report.md",
                "reports/20260913_stage_abc_review/review.md"})
report = (OUT / "review.md").read_text()
for finding in ("S01", "S02", "S03", "S04", "S05"):
    assert f"**{finding}の実装仕様：" in report
assert sum(line.startswith("```") for line in report.splitlines()) % 2 == 0, "Unbalanced fenced blocks"
targets.update(p.removeprefix(str(ROOT) + "/") for p in re.findall(
    r"\]\((/Users/shonen/leadlag/[^)]+?)(?::\d+)?\)", report))
source_hashes = {p: sha(ROOT / p) for p in sorted(targets) if (ROOT / p).is_file()}
previous = json.loads((ROOT / "reports/20260913_stage_abc_review/review_manifest.json").read_text())
changed_since_previous = [p for p, digest in previous["reviewed_file_sha256"].items()
                          if (ROOT / p).is_file() and sha(ROOT / p) != digest]
manifest = {
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "head": git("rev-parse", "HEAD").strip(),
    "tracked_changed_file_count": len(changed),
    "review_target_sha256": source_hashes,
    "changed_since_previous_review_manifest": changed_since_previous,
    "production_config_sha256": {
        str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "configs/production").glob("*.yaml"))
    },
    "evidence_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted(OUT.iterdir())
                        if p.is_file() and p.name not in {"review_manifest.json", "validation.json"}},
    "notes": "Review-only. Source/config/test files were not edited. Production config hashes identify inputs, not a separate review of every config.",
}
(OUT / "review_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
validation_path = OUT / "validation.json"
validation_path.write_text("{}\n")
links = []
for ref in re.findall(r"\]\(([^)]+)\)", report):
    if ref.startswith("http"):
        continue
    raw_path, _, line = ref.partition(":")
    path = Path(raw_path) if raw_path.startswith("/") else OUT / raw_path
    assert path.is_file(), ref
    if line:
        assert 1 <= int(line) <= len(path.read_text().splitlines()), ref
    links.append(ref)
validation = {
    "status": "PASS",
    "local_links_checked": len(links),
    "reproduction_cases_checked": len(reproductions),
    "model_probe_cases_checked": len(model_probes),
    "source_hashes_recorded": len(source_hashes),
    "test_result": "591 passed, 17 warnings in 699.38s",
    "git_diff_check": "PASS",
    "handoff_cli_execution_calls": close_handoff["close_execution_calls"],
    "unchanged_runtime_files_checked": unchanged_runtime_files,
    "documentation_update": "Implementation handoff added; full tests were not rerun for this document update.",
}
validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(validation, ensure_ascii=False, indent=2))
