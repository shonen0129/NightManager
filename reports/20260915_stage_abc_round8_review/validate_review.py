"""Validate Round 8 evidence and report links without modifying reviewed code."""
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read_json(name):
    return json.loads((OUT / name).read_text())


def no_probe_errors(value):
    if isinstance(value, dict):
        return "probe_error" not in value and all(no_probe_errors(v) for v in value.values())
    if isinstance(value, list):
        return all(no_probe_errors(v) for v in value)
    return True


data = read_json("reproductions.json")
contracts = read_json("contracts.json")
model = read_json("model_probes.log")
prior = read_json("prior_probes.log")
schema = read_json("schema_cache.json")
gap = read_json("var_gap_version.json")
summary = read_json("review_summary.json")
boundaries = read_json("boundaries.json")
deadline = read_json("deadline.json")
provenance = read_json("provenance.json")
checks = {"no_probe_errors": no_probe_errors([data, contracts, model, prior, schema, gap])}

bundles = data["U01_bundle_boundaries"]
checks["V01_faults_reached_and_invalid_cache_rejected"] = all(
    cases[name]["write_fault_count"] == 1
    and not cases[name]["save_returned"]
    and cases[name]["metadata"] is None
    and cases[name]["alerts"]
    and cases[name]["compute_distribution_ondemand_calls"] == 1
    and not cases[name]["compute_distribution_used_new_mu"]
    for cases in bundles.values()
    for name in ("omega_failure", "metadata_failure", "manifest_failure")
)
checks["V01_ondemand_normal_omega"] = all(cases["omega_failure"]["compute_distribution_used_old_omega"] for cases in bundles.values())
checks["V01_blend_skips_rejected_horizon"] = all(
    bundles[h]["omega_failure"]["blend_max_change"] < 1e-12
    and bundles[h]["omega_failure"]["blend_alerts"] for h in ("3", "5")
)
checks["U01_main_decide_rejects_invalid_bundle"] = all(
    case["decision"]["gross"] == 0 and case["decision"]["fallback"]["audit_failure"]
    for name, case in bundles["1"].items() if name not in ("normal", "interleaved_read")
)
checks["U01_normal_nonflat"] = bundles["1"]["normal"]["decision"]["gross"] > 1.9
checks["U01_concurrent_read_detected"] = all(
    cases["interleaved_read"]["interleaved_commit"]
    and cases["interleaved_read"]["metadata"] is None
    and cases["interleaved_read"]["alerts"] for cases in bundles.values()
)

broker = data["F01_F02_F20_brokers"]
checks["V02_resolved"] = (
    broker["expired_partial"] == "CANCELLED"
    and broker["cancel_failed_unfilled"] == "SUBMITTED"
    and broker["amend_failed_unfilled"] == "SUBMITTED"
)
checks["F01_F02_F20_normal_and_failure_paths"] = (
    broker["filled"] == "FILLED" and broker["partial"] == "PARTIALLY_FILLED"
    and broker["cancelled_partial"] == "CANCELLED" and broker["rejected"] == "FAILED"
    and not broker["all_rejected"]["returned_normally"]
    and broker["all_rejected"]["accepted"] == 0 and broker["all_rejected"]["failed"] == 2
    and broker["fill_collection"]["queries"] == 2 and broker["fill_collection"]["quantities"] == [30, 30]
    and all(broker[key]["calls"] == 3 and broker[key]["status"] == "FILLED"
            for key in ("leadlag.execution.broker_ops_poll", "leadlag.execution.close_poll"))
)
checks["V03_real_builder_resolved"] = (
    schema["rows"] == 4135 and schema["columns"] == 122
    and schema["row_hash_unchanged"] and not schema["cache_reused"]
    and not schema["has_difference"] and schema["max_abs_all_returns_difference"] == 0
)
checks["V04_gap_version_race_resolved"] = (
    gap["same_key"] and gap["cache_key_input_version"] == "A" and gap["backtest_versions"] == ["A"]
    and gap["first_marker"] == gap["after_snapshot_restore_marker"]
    and gap["after_snapshot_restore_marker"] == gap["expected_A_marker"]
)
race = data["U02_var_version_race"]
checks["U02_overlay_version_race_resolved"] = (
    race["same_cache_key"] and race["runner_version"] == race["version_a"]
    and race["backtest_loaded_versions"] == [race["version_a"]]
    and race["first_value"] == race["after_rollback_value"] == race["expected_a_marker"]
    and race["backtest_count"] == 1
)
verification = data["U03_verification"]
checks["U03_verification_resolved"] = (
    verification["healthy"]["verification_exit"] == 0 and verification["healthy"]["loader"]["accepted"]
    and all(case["verification_exit"] == 1 and not case["loader"]["accepted"]
            for name, case in verification.items() if name != "healthy")
)
checks["U04_migration_resolved"] = data["U04_migration"]["loader_accepts"] and data["U04_migration"]["migration_exit"] == 0

dates = data["T01_artifact_dates"]
checks["T01_artifact_boundaries"] = all(
    not case["save"]["accepted"] and not case["direct_apply"]["accepted"]
    for case in dates.values() if "save" in case
) and not dates["train_end_day"]["accepted"] and dates["first_oos_day"]["accepted"]
checks["T02_legacy_rejected"] = not data["T02_legacy"]["accepted"]
checks["T03_provenance_rejected"] = all(case["gross"] == 0 and case["fallback"]["audit_failure"] for case in data["T03_missing_metadata"].values())
fill = data["T04_fill_failure"]
checks["T04_fill_failure_propagates"] = (
    not fill["decision"]["returned_normally"] and fill["decision"]["detail_calls"] == 2
    and fill["close"]["cli_return_code"] == 2 and fill["close"]["close_incomplete"]
)
log = data["T05_initial_log"]
checks["T05_initial_log_failure_reconciles"] = log["error_type"] == "OrderExecutionIncomplete" and log["reconciliation_calls"] == ["fills", "positions", "wallet"]
checks["R02_reconciliation_continues"] = prior["partial_failure_skips_reconciliation"]["post_failure_collection"] == ["fills", "positions", "wallet", "journal"]
checks["S03_close_failure_exit"] = prior["all_rejected_close_cli_success"]["cli_return_code"] == 2
checks["T02_atomic_artifact_publish"] = prior["overlay_atomic_publish_failure"]["publication_failed"] and prior["overlay_atomic_publish_failure"]["active_version_unchanged"]

risk = data["F03_risk_reduction"]
checks["F03_risk_reduction_allowed"] = (
    not risk["actual_stop_flat_flow"]["blocked"] and risk["flat"]["allowed"] and risk["reduce"]["allowed"]
    and all(not risk[key]["allowed"] for key in ("increase", "reverse", "new"))
)
checks["F04_audit_flat"] = data["F04_audit_flat"]["gross"] == 0 and data["F04_audit_flat"]["fallback"]["audit_failure"]
checks["F04_future_h3_rejected"] = all(prior["future_h3_metadata_flats"][key]["gross"] == 0 for key in ("case_1", "case_2"))
inputs = data["F05_F07_F08_F09_data"]
checks["F05_stale_trade_date_rejected"] = not inputs["stale_requested_day"]["accepted"]
checks["F07_current_price"] = inputs["current_prices"] == {"calls": ["current"], "price": 1050.0}
checks["F08_holiday_mapping"] = math.isclose(inputs["holiday"]["2026-08-12"]["us_return"], 0.1) and inputs["holiday"]["2026-08-13"]["us_return"] == 0
checks["R01_S05_padding"] = all(case["contains_20260813"] for case in prior["nontrading_nan_row_loses_next_return"].values())
checks["F09_strict_clean"] = inputs["strict_clean_data"]["rows"] == 79 and inputs["strict_clean_data"]["signals_strictly_prior"]
checks["F10_correlation_cache"] = data["F10_correlation_cache"]["ratio"] == 1.0 and not data["F10_correlation_cache"]["cached_object_reused"]
checks["F10_horizon_and_future_target"] = all(value == 0 for case in model.values() for name, value in case.items() if "difference" in name)
checks["F11_numeric_cache_invalidation"] = data["F11_common_inputs_cache"]["build_calls"] == 2 and data["F11_common_inputs_cache"]["observed_last"] == 100 and not data["F11_common_inputs_cache"]["target_remains_zero"]
checks["F12_sqlite_atomic_write"] = all(
    case["fault_reached"] and case["error"] == "GapStoreError" and case["mu"] == case["omega"] == 1.0 and case["metadata"]["version"] == "old"
    for case in data["F12_atomic_sqlite_write"].values()
)
checks["R04_sqlite_atomic_read"] = all(
    case["interleaved_commit"] and case["mu"] == case["omega"] == 1.0 and case["meta"]["version"] == "old" and case["latest_mu"] == 2.0
    for case in prior["atomic_reader_passes_all_horizons"].values()
)
checks["F13_backtest_autoload"] = data["F13_default_backtest_overlay"]["ml_overlay_enabled"] and "MLOrderOverlayModel(" in data["F13_default_backtest_overlay"]["overlay_model_forwarded"]
checks["F14_inventory_slippage"] = all(case["reported_slip_bps"] == case["inventory_flow_slip_bps"] for case in data["F14_inventory_slippage"].values())
checks["F15_initial_drawdown"] = math.isclose(data["F15_initial_drawdown"]["MDD"], -0.1)
metrics = data["F16_all_days_metrics"]
checks["F16_all_days_metrics"] = metrics["n_observations"] == 4 and metrics["fallback_rate"] == 0.5 and math.isclose(metrics["total_return"], -0.082) and math.isclose(metrics["max_dd"], -0.1)
checks["F17_DSR_frequency"] = set(data["F17_DSR"]) == {"245", "252"} and all(case["reported"] == case["reference"] for case in data["F17_DSR"].values())
checks["F21_PIT_multiplier"] = data["F21_PIT_multiplier"]["actual_multiplier"] == 0.25 and data["F21_PIT_multiplier"]["fallback_flag"]
checks["production_artifact_operational_pending"] = not data["production_artifact_readiness"]["accepted"] and not prior["configured_runner"]["constructs"]
step2 = contracts["F06_step2_config"]
checks["F06_step2_inherited_config"] = (
    len(step2["config_calls"]) == 1 and step2["config_calls"][0]["strict"] and step2["config_calls"][0]["same_v2"]
    and step2["baseline_parameters"] == [5, 5, 2.0, 10.0, True, 0.8, True, True]
)
wal = contracts["R05_WAL_fingerprint"]
checks["R05_WAL_change_detected"] = wal["main_bytes_unchanged"] and wal["key_changed"] and wal["loaded_mu"] == 2.0
checks["F18_A7_guard"] = contracts["F18_A7_guard"]["blocked"] and contracts["F18_A7_guard"]["data_load_calls"] == 0
isolation = contracts["F22_R08_regression_isolation"]
checks["F22_R08_nonflat_and_no_global_pollution"] = isolation["pytest_exit"] == 0 and isolation["macro_function_restored"] and isolation["model_gross"] == 1.5 and abs(isolation["model_net"]) < 1e-12

checks["W01_previous_cases_resolved"] = all(
    not boundaries["metadata"][f"{name}_h{h}"]["loader"]["available"]
    and boundaries["metadata"][f"{name}_h{h}"]["full_validator_alerts"]
    and not boundaries["metadata"][f"{name}_h{h}"]["compute_distribution"]["accepted_cache"]
    and boundaries["metadata"][f"{name}_h{h}"]["compute_distribution"]["ondemand_calls"] == 1
    for h in (1, 3, 5) for name in ("alias_future", "trade_date_list", "trade_date_null", "horizon_inf")
)
checks["W01_blend_rejects_previous_cases"] = all(boundaries["metadata"][f"{name}_h{h}"]["blend"]["max_score_change"] == 0 and boundaries["metadata"][f"{name}_h{h}"]["blend"]["alerts"] for h in (3, 5) for name in ("alias_future", "trade_date_list", "trade_date_null", "horizon_inf"))
checks["X01_invalid_dates_allow_fallback"] = all(
    not boundaries["metadata"][f"{name}_h{h}"]["loader"]["available"]
    and boundaries["metadata"][f"{name}_h{h}"]["full_validator_alerts"]
    and not boundaries["metadata"][f"{name}_h{h}"]["compute_distribution"]["accepted_cache"]
    and boundaries["metadata"][f"{name}_h{h}"]["compute_distribution"]["ondemand_calls"] == 1
    for h in (1, 3, 5) for name in ("nat_signal", "empty_signal")
)
failed_provenance = [case for case in provenance["results"] if not case["passed"]]
checks["X01_both_backends_and_strict_contract_resolved"] = (
    provenance["execution_completed"]
    and provenance["cases"] == 72
    and not failed_provenance
)
locked = boundaries["locked_snapshot"]
checks["W02_sqlite_lock_deadline_resolved"] = not locked["still_pending_at_35_seconds"] and locked["child_exit"] == 0 and locked["after_lock_release"]["configured_timeout"] == 1 and locked["after_lock_release"]["elapsed"] < 3 and locked["after_lock_release"]["backtest_calls"] == 0 and locked["after_lock_release"]["empty"]
lifetime = boundaries["snapshot_timeout_lifetime"]
checks["W02_worker_snapshot_kept_until_finish"] = lifetime["exists_before_timeout"] and lifetime["caller_returned_empty"] and lifetime["exists_after_caller_timeout"] and lifetime["exists_when_worker_resumes"]
checks["W02_exception_and_worker_cleanup"] = all(case["snapshot_exists_after_worker_finish"] == [False] for case in deadline["cleanup"].values()) and deadline["cleanup"]["worker_late"]["snapshot_exists_after_return"] == [True]
checks["X02_expired_cache_hit_is_blocked"] = all(
    deadline[name]["injected_delay_reached"]
    and deadline[name]["returned_normally"]
    and deadline[name]["empty"]
    and deadline[name]["elapsed"] <= deadline[name]["configured_timeout"] + 0.3
    and deadline[name]["backtest_calls"] == 1
    for name in ("warm_hash", "warm_get")
)
checks["X02_cache_miss_stops_after_same_delay"] = (
    deadline["cold_hash"]["returned_normally"]
    and deadline["cold_hash"]["empty"]
    and deadline["cold_hash"]["backtest_calls"] == 0
)
checks["X02_directory_copy_checks_deadline"] = (
    deadline["directory_copy"]["configured_timeout"] == .1
    and deadline["directory_copy"]["error_type"] == "TimeoutError"
    and deadline["directory_copy"]["copytree_bypassed"]
    and deadline["directory_copy"]["deadline_checks"] >= 3
)
checks["X02_late_cache_write_is_terminated"] = (
    deadline["late_cache_write"]["timed_out"]
    and not deadline["late_cache_write"]["late_value_published"]
    and deadline["late_cache_write"]["elapsed"] < 0.5
)
test_log = (OUT / "pytest_full.log").read_text()
test_stderr = (OUT / "pytest_full.stderr").read_text()
checks["all_650_tests_passed"] = "650 passed, 17 warnings" in test_log and "WATCHDOG: exit=0" in test_stderr
for name in ("ruff", "mypy", "compileall", "import_linter", "model_probes", "prior_probes", "probe", "contracts", "schema_cache", "var_gap_version", "boundaries", "deadline", "provenance"):
    if name in {"ruff", "mypy", "compileall", "import_linter"}:
        checks[f"{name}_passed"] = {
            "ruff": "All checks passed!" in (OUT / "ruff.log").read_text(),
            "mypy": "Success: no issues found" in (OUT / "mypy.log").read_text(),
            "compileall": "WATCHDOG: exit=0" in (OUT / "compileall.stderr").read_text(),
            "import_linter": "Contracts: 4 kept, 0 broken." in (OUT / "import_linter.log").read_text(),
        }[name]
    else:
        checks[f"{name}_exit_zero"] = "WATCHDOG: exit=0" in (OUT / f"{name}.stderr").read_text()
for name in ("ruff", "mypy", "compileall", "import_linter"):
    checks[f"{name}_watchdog_exit_zero"] = "WATCHDOG: exit=0" in (OUT / f"{name}.stderr").read_text()
checks["diff_check"] = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, timeout=30).returncode == 0
checks["batch_syntax"] = subprocess.run(["bash", "-n", "scripts/batch/run_gap_distribution.sh"], cwd=ROOT, capture_output=True, timeout=10).returncode == 0
manifest = read_json("review_manifest.json")
checks["source_unchanged_during_review"] = manifest["source_changes_during_review"] == []
checks["review_only"] = summary["production_source_modified_by_review"] is False
checks["additional_acceptance"] = read_json("additional.json")["passed"] and "WATCHDOG: exit=0" in (OUT / "additional.stderr").read_text()
checks["head_unchanged"] = manifest["head"] == read_json("source_snapshot.json")["head"] == summary["head"]
checks["summary_fixes_resolved"] = (
    summary["verdict"] == "CONDITIONAL_PASS"
    and not summary["findings"]
    and {item["id"] for item in summary["resolved_fixes"]} == {"X01", "X02"}
)
checks["F01_to_F22_coverage"] = len(summary["original_findings"]) == 22 and {item["id"] for item in summary["original_findings"]} == {f"F{i:02}" for i in range(1, 23)}
checks["previous_round_coverage"] = set(summary["previous_round"]) == {"W01", "W02", "X01", "X02"}
checks["summary_evidence_references"] = all(
    (OUT / item["evidence"].split("#")[0]).is_file()
    and ("#" not in item["evidence"] or item["evidence"].split("#")[1] in read_json(item["evidence"].split("#")[0]))
    for item in summary["findings"] + summary["original_findings"]
)

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
        if path.resolve() == (OUT / "validation.json").resolve():
            continue
        if not path.is_file() or (line and line > len(path.read_text().splitlines())):
            broken_links.append({"document": doc.name, "target": target})
checks["report_links"] = not broken_links
checks = {key: bool(value) for key, value in checks.items()}
result = {
    "passed": all(checks.values()), "checks": checks,
    "links_checked": links_checked, "broken_links": broken_links,
    "evidence_sha256": {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "validation.json"
    },
}
(OUT / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"passed": result["passed"], "checks": checks, "links_checked": links_checked, "broken_links": broken_links}, ensure_ascii=False, indent=2))
raise SystemExit(0 if result["passed"] else 1)
