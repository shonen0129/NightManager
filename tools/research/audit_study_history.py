"""Index historical reports and individually review stored DSRs without trading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from leadlag.experiment_registry import ExperimentRegistry
from research.study_history import build_report_index, review_registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--write-corrections",
        action="store_true",
        help="append DSR review corrections, preserving every original row",
    )
    args = parser.parse_args()
    registry = ExperimentRegistry(args.registry or args.root / "var/experiments/registry.jsonl")
    result = {
        "report_index": build_report_index(args.root),
        "dsr_review": review_registry(registry, write=args.write_corrections),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
