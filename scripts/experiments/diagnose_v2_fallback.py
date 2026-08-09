#!/usr/bin/env python3
"""Diagnose why V2 fallback remains for specific dates after Step 2 clipping."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.execution.config import load_config_from_yaml
from leadlag.models.production_v2 import generate_v2_production_portfolio


def main() -> int:
    gap_dir = ROOT / "outputs/long_period/gap_adjusted_false_clipped/20260730_173341"
    app_config = load_config_from_yaml(ROOT / "configs/production/production.yaml")

    dates = [
        "2025-10-28", "2025-10-29", "2025-10-30", "2025-10-31",
        "2025-11-04", "2025-11-05", "2025-11-06", "2025-11-07",
        "2025-11-10", "2025-11-18", "2026-01-05", "2026-01-26",
    ]

    for date_str in dates:
        res = generate_v2_production_portfolio(date_str, gap_dir, cfg=app_config.v2)
        numerical = res["numerical"]
        print(
            f"{date_str}: fallback={res['fallback']} "
            f"status={numerical['status']} "
            f"scores_finite={numerical['scores_finite']} "
            f"weights_finite={numerical['weights_finite']} "
            f"net={numerical['net_exposure_value']:.4e} "
            f"gross={numerical['gross_exposure_value']:.4f} "
            f"diag_nonneg={numerical['covariance_diag_nonneg']} "
            f"sym_ok={numerical['covariance_symmetric']} "
            f"alerts={res['alerts']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
