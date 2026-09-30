#!/usr/bin/env python3
"""Append target-corrected Rank IC records for the 2026-09-24 long audit.

The historical ``rank_ic`` field measured open-to-09:10 returns. The source
report and its CSV contain the corrected prediction-target Rank IC. This tool
keeps the old registry row untouched, appends a corrected current-view row,
and defaults to dry-run unless ``--write`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from leadlag.experiment_registry import ExperimentRecord, ExperimentRegistry

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "var/experiments/registry.jsonl"
RESULTS = ROOT / "var/results/20260924_sensitivity_pipeline_audit_long/rank_ic_vs_backtest_target.csv"
REPORT = "reports/20260924_sensitivity_pipeline_audit_long/report.md"
PREFIX = "20260924_sensitivity_pipeline_audit_long_"
STUDY_ID = "sensitivity-pipeline-audit-long-2026-09-24"
METRIC_SCHEMA = "sensitivity-rank-ic-target-v2"
EXPECTED_TRIAL_COUNT = 10
EXPECTED_DATES = 2779


def _load_target_rows(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_variant: dict[str, dict[str, Any]] = {}
    for row in rows:
        variant = str(row.get("variant") or "")
        if not variant or variant == "current":
            continue
        if variant in by_variant:
            raise ValueError(f"duplicate target Rank IC row for {variant}")
        confidence_interval = json.loads(str(row["ci95"]))
        if not isinstance(confidence_interval, list) or len(confidence_interval) != 2:
            raise ValueError(f"invalid Rank IC confidence interval for {variant}")
        observation_count = int(row["dates"])
        full_cross_section_count = int(row["dates_all17_target"])
        if observation_count != EXPECTED_DATES or full_cross_section_count != EXPECTED_DATES:
            raise ValueError(f"unexpected target panel coverage for {variant}")
        by_variant[variant] = {
            "target_dates": observation_count,
            "dates_all17_target": full_cross_section_count,
            "mean_rank_ic": float(row["mean_rank_ic_vs_backtest_target"]),
            "ci95": [float(value) for value in confidence_interval],
        }
    if len(by_variant) != EXPECTED_TRIAL_COUNT:
        raise ValueError(
            f"expected {EXPECTED_TRIAL_COUNT} corrected candidates; found {len(by_variant)}"
        )
    return by_variant


def _corrected_record(
    original: ExperimentRecord,
    target_metrics: dict[str, Any],
) -> ExperimentRecord:
    metrics = dict(original.metrics)
    legacy_rank_ic = metrics.pop("rank_ic", None)
    if not isinstance(legacy_rank_ic, dict):
        raise ValueError(f"legacy rank_ic metric is missing for {original.name}")
    metrics["rank_ic_open_to_0910_diagnostic"] = {
        "target_basis": "open_to_0910_returns",
        **{key: value for key, value in legacy_rank_ic.items() if key != "daily"},
    }
    metrics["rank_ic_vs_backtest_target"] = {
        **target_metrics,
        "target_basis": (
            "same-date cross-sectional prediction mu versus the 09:10-to-close "
            "backtest target; observed 09:10 proxy where available and legacy "
            "open-to-close fallback otherwise"
        ),
        "confidence_interval_method": (
            "20-session non-circular moving-block bootstrap, 5000 resamples"
        ),
        "observed_0910_cell_fraction": 0.0313,
        "observed_0910_quote_dates": 102,
        "all_17_observed_quote_dates": 8,
    }
    return ExperimentRecord(
        name=original.name,
        hypothesis=original.hypothesis,
        start_time=original.start_time,
        end_time=original.end_time,
        parameters=original.parameters,
        metrics=metrics,
        decision=original.decision,
        report_path=REPORT,
        related_records=original.related_records,
        study_id=STUDY_ID,
        metric_schema_version=METRIC_SCHEMA,
    )


def correct_registry(
    registry_path: Path,
    results_path: Path,
    *,
    write: bool = False,
) -> dict[str, Any]:
    registry = ExperimentRegistry(registry_path)
    records = list(registry)
    target_by_variant = _load_target_rows(results_path)
    original_by_variant: dict[str, ExperimentRecord] = {}
    for record in records:
        if not record.name.startswith(PREFIX) or record.correction_of is not None:
            continue
        variant = record.name.removeprefix(PREFIX)
        if variant in target_by_variant:
            if variant in original_by_variant:
                raise ValueError(f"duplicate original registry row for {variant}")
            original_by_variant[variant] = record
    if set(original_by_variant) != set(target_by_variant):
        missing = sorted(set(target_by_variant) - set(original_by_variant))
        extra = sorted(set(original_by_variant) - set(target_by_variant))
        raise ValueError(f"registry candidate mismatch: missing={missing}, extra={extra}")

    current_ids = {record.record_id for record in registry.iter_current_records()}
    planned: list[tuple[ExperimentRecord, ExperimentRecord]] = []
    already_current: list[str] = []
    for variant, original in sorted(original_by_variant.items()):
        corrected = _corrected_record(original, target_by_variant[variant])
        if original.record_id not in current_ids:
            existing = next(
                (
                    record
                    for record in registry.iter_current_records()
                    if record.correction_of == original.record_id
                ),
                None,
            )
            if existing is None or existing.metric_schema_version != METRIC_SCHEMA:
                raise ValueError(f"registry row {variant} is superseded by an unknown correction")
            actual_metric = existing.metrics.get("rank_ic_vs_backtest_target")
            expected_metric = corrected.metrics.get("rank_ic_vs_backtest_target")
            if actual_metric != expected_metric:
                raise ValueError(f"existing correction differs from source for {variant}")
            already_current.append(variant)
            continue
        planned.append((original, corrected))

    if write:
        for original, corrected in planned:
            registry.record_correction(original.record_id, corrected)
        if registry.count_trials(study_id=STUDY_ID) != EXPECTED_TRIAL_COUNT:
            raise RuntimeError("corrected sensitivity study trial count did not reconcile")
    return {
        "write": write,
        "appended_count": len(planned) if write else 0,
        "planned_count": len(planned),
        "already_current_count": len(already_current),
        "variants": [record.name.removeprefix(PREFIX) for _, record in planned],
        "study_id": STUDY_ID,
        "metric_schema_version": METRIC_SCHEMA,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--results-csv", type=Path, default=RESULTS)
    parser.add_argument("--write", action="store_true", help="append corrections to the registry")
    args = parser.parse_args()
    result = correct_registry(args.registry, args.results_csv, write=args.write)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
