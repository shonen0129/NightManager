"""Compare the unified provenance contract across SQLite and .npy readers."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from leadlag.utils.gap_matrix_io import save_gap_matrices, load_gap_bundle
from leadlag.data.validation import DataValidationError
from leadlag.models.v2.distribution_source import _validate_distribution_metadata

cases = {
    "normal": ({"sig_date": "2026-08-13"}, True),
    "matching_alias": ({"sig_date": "2026-08-13T23:00:00Z", "signal_date": "2026-08-13"}, True),
    "alias_only": ({"signal_date": "2026-08-13"}, True),
    "alias_disagreement": ({"sig_date": "2026-08-13", "signal_date": "2026-08-17"}, False),
    "null_signal": ({"sig_date": None}, False),
    "nat_signal": ({"sig_date": "NaT"}, False),
    "signal_array": ({"sig_date": ["2026-08-13", "2026-08-12"]}, False),
    "null_trade": ({"sig_date": "2026-08-13", "trade_date": None}, False),
    "trade_array": ({"sig_date": "2026-08-13", "trade_date": ["2026-08-14", "2026-08-15"]}, False),
    "fractional_horizon": ({"sig_date": "2026-08-13", "horizon": 1.5}, False),
    "nonfinite_horizon": ({"sig_date": "2026-08-13", "horizon": float("inf")}, False),
    "same_day": ({"sig_date": "2026-08-14"}, False),
}
results = []
with tempfile.TemporaryDirectory(prefix="abc-r7-provenance-") as tmp:
    root = Path(tmp)
    for backend in ("npy", "sqlite"):
        for horizon in (1, 3, 5):
            patterns = {} if horizon == 1 else {"mu_pattern": "matrices/mu_gap_h{h}_{date}.npy", "omega_pattern": "matrices/omega_gap_h{h}_{date}.npy", "pattern_kwargs": {"h": horizon}}
            for name, (fields, expected) in cases.items():
                path = root / backend / str(horizon) / name
                if backend == "sqlite":
                    path = path.with_suffix(".sqlite")
                metadata = {"trade_date": "2026-08-14", "horizon": horizon, **fields}
                assert save_gap_matrices(path, "2026-08-14", np.ones(17), np.eye(17), metadata=metadata, **patterns)
                errors = {}
                mu = omega = actual_meta = normalized = None
                alerts = model_alerts = []
                try:
                    mu, omega, actual_meta, alerts = load_gap_bundle(path, "2026-08-14", require_metadata=True, **patterns)
                except Exception as exc:
                    errors["loader"] = {"type": type(exc).__name__, "message": str(exc)}
                try:
                    normalized, model_alerts = _validate_distribution_metadata(metadata, "2026-08-14", horizon)
                except Exception as exc:
                    errors["model_validator"] = {"type": type(exc).__name__, "message": str(exc)}
                strict_rejected = False
                try:
                    load_gap_bundle(path, "2026-08-14", require_metadata=True, strict=True, **patterns)
                except DataValidationError:
                    strict_rejected = True
                except Exception as exc:
                    errors["strict_loader"] = {"type": type(exc).__name__, "message": str(exc)}
                actual = mu is not None and omega is not None
                passed = not errors and actual == expected and (not model_alerts) == expected and strict_rejected == (not expected)
                if expected:
                    passed = passed and normalized == actual_meta and actual_meta["sig_date"] == "2026-08-13"
                results.append({"backend": backend, "horizon": horizon, "case": name, "expected_accepted": expected, "accepted": actual, "strict_rejected": strict_rejected, "model_accepted": not model_alerts and "model_validator" not in errors, "passed": passed, "metadata": actual_meta, "alerts": alerts, "errors": errors})
result = {"execution_completed": True, "passed": all(case["passed"] for case in results), "cases": len(results), "results": results}
(OUT / "provenance.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"passed": result["passed"], "cases": len(results), "failed": [case for case in results if not case["passed"]]}, ensure_ascii=False, indent=2))
# A completed review probe records contract failures as findings; unlike a
# regression test, its exit status indicates whether evidence was collected.
