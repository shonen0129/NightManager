"""Validate the canonical gap SQLite bundle for one trade date."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from leadlag.config.paths import gap_store_path
from leadlag.core.market_calendar import is_market_closed
from leadlag.data.gap_store import GapStore
from leadlag.domain.gap_bundle import canonical_json_bytes, sha256_bytes
from leadlag.utils.timestamps import normalize_jst_date


def check_bundle(store_path: str | Path, trade_date: str, *, horizon: int | None = None) -> dict[str, object]:
    """Return an auditable status for the date-scoped canonical bundle."""
    date_key = normalize_jst_date(trade_date).strftime("%Y-%m-%d")
    path = Path(store_path)
    result: dict[str, object] = {
        "store": str(path),
        "trade_date": date_key,
        "horizon": horizon,
        "market_closed": bool(is_market_closed(normalize_jst_date(date_key).date())),
        "status": "missing",
    }
    if not path.exists():
        result["reason"] = "store_missing"
        return result

    store = GapStore(path)
    mu, omega, metadata, manifest = store.load_horizon_bundle(date_key, horizon=horizon)
    if mu is None or omega is None or metadata is None:
        result["reason"] = "mu_omega_metadata_missing"
        return result
    if manifest is None:
        result["reason"] = "bundle_manifest_missing"
        return result
    errors = manifest.validate(
        trade_date=date_key,
        horizon=horizon,
        storage_format="sqlite",
        mu_sha256=sha256_bytes(np.ascontiguousarray(mu).tobytes()),
        omega_sha256=sha256_bytes(np.ascontiguousarray(omega).tobytes()),
        metadata_sha256=sha256_bytes(canonical_json_bytes(metadata)),
    )
    if errors:
        result["reason"] = "bundle_manifest_invalid"
        result["errors"] = errors
        return result
    result.update(
        {
            "status": "ready",
            "mu_shape": list(mu.shape),
            "omega_shape": list(omega.shape),
            "source": metadata.get("source"),
            "bundle_version": metadata.get("bundle_version", metadata.get("version")),
        }
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=gap_store_path())
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--horizon", type=int, default=None)
    args = parser.parse_args(argv)
    result = check_bundle(args.store, args.trade_date, horizon=args.horizon)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] == "ready" or result.get("market_closed") is True:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
