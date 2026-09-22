"""Build this review's summary from fresh evidence and adapt prior assertions."""
import json
from pathlib import Path
import re
import sys

OUT = Path(__file__).resolve().parent
PRIOR = OUT.parent / "20260915_stage_abc_round7_review"
summary = json.loads((PRIOR / "review_summary.json").read_text())
summary.update(
    round=8,
    scope="前回修正X01/X02の差分4ファイルと下流経路、およびF01〜F22・R〜Wの記録済み再発条件",
    production_source_modified_by_review=False,
    live_broker_called=False,
    verdict="CONDITIONAL_PASS",
    reviewed_fix_verdict="PASS",
    findings=[],
)
summary["previous_round"] = {
    "W01": "共通validatorのalias整合・null/配列/非有限値拒否を維持",
    "W02": "SQLite snapshotの待機期限・worker寿命・後始末を維持",
    "X01": "NaT文字列・空文字の拒否とstrict例外・on-demand・blend契約を独立再確認",
    "X02": "hash/read/copy/initializationの待機打切り、実SQLiteロックと保存子プロセスの終了を独立再確認",
}
summary["resolved_fixes"][1]["change"] = "呼出側の総deadlineと背景workerの所有権を実確認。保存子プロセスの正常終了・ロック時停止・異常終了rollbackを確認"
summary["verification"]["additional_acceptance"] = {
    "evidence": "additional.json",
    "passed": json.loads((OUT / "additional.json").read_text())["passed"],
    "loader_cases": 36,
    "direct_validator_cases": 20,
    "case_counts_overlap_prior_conditions": True,
    "deadline": "directory file copy / config load / cache initialization blocked until caller timeout",
    "cache_process": "3 normal writes with intervening parent reads, actual SQLite write lock, serialization error rollback; no remaining children",
}
summary["limitations"] = [
    "今回の判定は修正差分と明記した回帰条件に限定する。全ワークスペースの無欠陥や本番昇格を保証しない。",
    "VaR待機打切りはdaemon thread自体のキャンセルではない。OS内で永久停止したreadやBT workerは外側watchdogで管理する。",
    "実行環境はmacOS/Python 3.12。Windowsのspawn経路と強制終了挙動は実行していない。",
    "VaR検証は実cache・一時DBを使用し、重いBT数値ループは版識別markerまたは待機workerへ置換した。",
    "未来target摂動はmacro無効の合成入力。全外部系列の公表時刻・実約定後exposure・新規OOS成績は検証していない。",
]
# Keep prior concrete regression assertions; replace only round identity and
# evidence bookkeeping. Add independent acceptance evidence to the checks.
validator = (PRIOR / "validate_review.py").read_text().replace("Validate Round 7", "Validate Round 8")
validator = validator.replace('"compileall": not (OUT / "compileall.stderr").read_text(),', '"compileall": "WATCHDOG: exit=0" in (OUT / "compileall.stderr").read_text(),')
old = '''checks["review_changes_documented"] = set(manifest["source_changes_during_review"]) == {
    "src/leadlag/execution/var_history.py",
    "src/leadlag/utils/distribution_provenance.py",
    "tests/unit/test_gap_matrix_io.py",
    "tests/unit/test_stage_abc_followup_fixes.py",
}'''
assert old in validator
validator = validator.replace(old, '''checks["source_unchanged_during_review"] = manifest["source_changes_during_review"] == []
checks["review_only"] = summary["production_source_modified_by_review"] is False
checks["additional_acceptance"] = read_json("additional.json")["passed"] and "WATCHDOG: exit=0" in (OUT / "additional.stderr").read_text()''')
validator = validator.replace('set(summary["previous_round"]) == {"W01", "W02"}', 'set(summary["previous_round"]) == {"W01", "W02", "X01", "X02"}')
validator = validator.replace('checks["X01_NaT_and_empty_string_block_fallback"]', 'checks["X01_invalid_dates_allow_fallback"]')
# A completed collector is not itself an assertion of correctness; retain all
# prior semantic assertions and require fresh watchdog results for statics.
validator = validator.replace('checks["diff_check"] =', '''for name in ("ruff", "mypy", "compileall", "import_linter"):
    checks[f"{name}_watchdog_exit_zero"] = "WATCHDOG: exit=0" in (OUT / f"{name}.stderr").read_text()
checks["diff_check"] =''')
(OUT / "validate_review.py").write_text(validator)
if "--prepare-validator-only" in sys.argv:
    print("Prepared validator; no final summary or verdict generated")
    raise SystemExit(0)
log = (OUT / "pytest_full.log").read_text()
match = re.search(r"(\d+) passed, (\d+) warnings in ([\d.]+)s", log)
if not match or "WATCHDOG: exit=0" not in (OUT / "pytest_full.stderr").read_text():
    raise RuntimeError("Full test run is not yet complete; do not finalize")
summary["verification"]["pytest"].update(passed=int(match[1]), warnings=int(match[2]), seconds=float(match[3]), exit_code=0)
(OUT / "review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
report = OUT / "review.md"
report.write_text(report.read_text().replace("PYTEST_RESULT", f"{match[1]} passed / {match[2]} warnings / {match[3]}秒"))
print(json.dumps({"round": 8, "findings": 0, "pytest": summary["verification"]["pytest"]}, ensure_ascii=False))
