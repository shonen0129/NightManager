#!/usr/bin/env python3
"""Score paired ML-overlay shadow records against verified daily outcomes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from leadlag.config.paths import project_root
from leadlag.reporting.ml_overlay_forward_evaluation import evaluate_ml_overlay_forward

ROOT = project_root()
DEFAULT_SHADOW_DIR = ROOT / "var/shadow_runs/ml_overlay_value"


def _read_jsonl(path: Path, *, optional: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        if optional:
            return []
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL row at {path}:{line_number} must be an object")
            records.append(record)
    return records


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(f"refusing to replace different evaluation artifact: {path}")
        return
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-dir", type=Path, default=DEFAULT_SHADOW_DIR)
    parser.add_argument("--outcomes", type=Path, default=None)
    parser.add_argument("--expected-trade-dates", type=Path)
    parser.add_argument("--annualization-periods", type=int, default=252)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    shadow_dir = args.shadow_dir if args.shadow_dir.is_absolute() else ROOT / args.shadow_dir
    outcome_path = args.outcomes or (shadow_dir / "outcomes.jsonl")
    if not outcome_path.is_absolute():
        outcome_path = ROOT / outcome_path
    shadow_records = _read_jsonl(shadow_dir / "daily.jsonl")
    outcome_records = _read_jsonl(outcome_path, optional=True)
    expected_dates: list[str] = []
    if args.expected_trade_dates is not None:
        dates_path = args.expected_trade_dates
        if not dates_path.is_absolute():
            dates_path = ROOT / dates_path
        expected_dates = [
            line.strip() for line in dates_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    evaluation = evaluate_ml_overlay_forward(
        shadow_records,
        outcome_records,
        annualization_periods=args.annualization_periods,
        expected_trade_dates=expected_dates,
    )
    output_dir = args.output_dir or (shadow_dir / "forward_evaluations")
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_path = output_dir / f"evaluation_{evaluation['evaluation_fingerprint'][:16]}.json"
    _write_immutable_json(output_path, evaluation)
    print(json.dumps({"output": str(output_path), **{
        key: value for key, value in evaluation.items() if key != "daily"
    }}, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
