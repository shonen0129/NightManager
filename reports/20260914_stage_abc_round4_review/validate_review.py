"""Check evidence and report links, without modifying the reviewed source."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
data = json.loads((OUT / "reproductions.json").read_text())
prior = json.loads((OUT / "prior_probes.json").read_text())
checks = {}
checks["no_probe_errors"] = all("probe_error" not in value for value in list(data.values()) + list(prior.values()))
dates = data["T01_artifact_boundaries"]
checks["T01"] = all(not value["save"]["accepted"] and not value["direct_apply"]["accepted"] for key, value in dates.items() if "save" in value)
checks["T01_normal_boundary"] = not dates["train_end_day"]["accepted"] and dates["first_oos_day"]["accepted"]
checks["T02"] = not data["T02_legacy_rejection"]["accepted"]
checks["T03"] = all(value["gross"] == 0 and value["fallback"]["audit_failure"] for value in data["T03_missing_provenance"].values())
fill = data["T04_fill_failure"]
checks["T04"] = not fill["decision"]["returned_normally"] and fill["decision"]["detail_calls"] == 2 and fill["close"]["cli_return_code"] == 2 and fill["close"]["close_incomplete"]
log = data["T05_initial_log_failure"]
checks["T05"] = log["error_type"] == "OrderExecutionIncomplete" and log["reconciliation_calls"] == ["fills", "positions", "wallet"]
checks["U01_reproduced"] = all(value["leakage"]["status"] == "PASSED" and value["gross"] > 1.9 and value["loaded_metadata"]["marker"] == "old" for key, value in data["npy_republication"].items() if key != "normal")
race = data["var_version_race"]
checks["U02_reproduced"] = race["same_cache_key"] and race["runner_version"] == race["version_a"] and race["backtest_loaded_versions"] == [race["version_b"]] and race["after_rollback_value"] != race["expected_a_marker"]
checks["U03_reproduced"] = all(value["verification_exit"] == 0 and not value["loader"]["accepted"] for key, value in data["verification_dangling_pointer"].items() if key != "healthy")
checks["U04_reproduced"] = data["migration_report"]["loader_accepts"] and data["migration_report"]["migration_exit"] == 1
test_log = (OUT / "pytest_full.log").read_text()
checks["all_tests_passed"] = "614 passed, 17 warnings" in test_log and "WATCHDOG: exit=0" in test_log
for name in ("ruff", "mypy", "compileall", "import_linter"):
    checks[name] = "WATCHDOG: exit=0" in (OUT / f"{name}.log").read_text()
checks["diff_check"] = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True).returncode == 0
checks["batch_syntax"] = subprocess.run(["bash", "-n", "scripts/batch/run_gap_distribution.sh"], cwd=ROOT, capture_output=True).returncode == 0
manifest = json.loads((OUT / "review_manifest.json").read_text())
checks["source_unchanged_during_review"] = not manifest["source_changes_during_review"]
broken_links = []
links_checked = 0
for doc in (OUT / "review.md", OUT / "README.md"):
    for target in re.findall(r"\]\(([^)]+)\)", doc.read_text()):
        if target.startswith(("http:", "https:", "#")):
            continue
        match = re.fullmatch(r"(.+?):(\d+)", target)
        name, line = (match.group(1), int(match.group(2))) if match else (target, None)
        path = Path(name)
        if not path.is_absolute():
            path = doc.parent / path
        links_checked += 1
        # validation.json is produced immediately below.
        if path.resolve() == (OUT / "validation.json").resolve():
            continue
        if not path.is_file() or (line and line > len(path.read_text().splitlines())):
            broken_links.append({"document": doc.name, "target": target})
checks["report_links"] = not broken_links
evidence_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "validation.json"}
result = {"passed": all(checks.values()), "checks": checks, "links_checked": links_checked, "broken_links": broken_links, "evidence_sha256": evidence_hashes}
(OUT / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"passed": result["passed"], "checks": checks, "broken_links": broken_links}, ensure_ascii=False, indent=2))
raise SystemExit(0 if result["passed"] else 1)
