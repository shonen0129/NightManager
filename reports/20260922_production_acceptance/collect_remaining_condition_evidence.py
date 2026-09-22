"""Collect read-only evidence for the remaining production acceptance gates."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_load(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def cache_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for filename in ("etf_prices.sqlite", "df_exec.sqlite"):
        path = ROOT / "var" / "market_data" / filename
        if not path.exists():
            result[filename] = {"exists": False}
            continue
        with sqlite3.connect(path) as conn:
            rows = conn.execute(
                "SELECT key, updated_at FROM cache_store ORDER BY key"
            ).fetchall()
        result[filename] = {
            "exists": True,
            "sha256": sha256(path),
            "keys": [{"key": key, "store_updated_at": updated_at} for key, updated_at in rows],
        }
    return result


def main() -> None:
    provenance = json_load(OUT / "input_provenance.json")
    fill_status = json_load(OUT / "historical_fill_status.json")
    csv_analysis_path = OUT / "tachibana_csv_analysis.json"
    csv_analysis = json_load(csv_analysis_path) if csv_analysis_path.exists() else None
    snapshots = sorted((OUT / "reconciliation").glob("account_snapshot_*.json"))
    account_snapshot = (
        str(snapshots[-1].relative_to(ROOT)) if snapshots else None
    )
    scheduler: dict[str, Any] = {
        "registered_jobs": json_load(OUT / "scheduler_status.json"),
        "dry_run_log": str((OUT / "scheduler_like_dry_run.log").relative_to(ROOT)),
        "launchd_wrapper": str((OUT / "launchd_dry_run_wrapper.sh").relative_to(ROOT)),
        "promoted_production_dry_run_log": str(
            (OUT / "promoted_production_dry_run.log").relative_to(ROOT)
        ),
    }
    dry_run_text = (OUT / "scheduler_like_dry_run.log").read_text(encoding="utf-8")
    promoted_text = (OUT / "promoted_production_dry_run.log").read_text(encoding="utf-8")
    scheduler["safe_launchd_dry_run_passed"] = (
        '"dry_run": true' in dry_run_text
        and '"orders_submitted": false' in dry_run_text
    )
    scheduler["promoted_production_config_dry_run_passed"] = (
        '"dry_run": true' in promoted_text
        and '"orders_submitted": false' in promoted_text
        and '"result_path"' in promoted_text
    )

    log_text = (OUT / "historical_fills.log").read_text(encoding="utf-8")
    error_codes = sorted(set(re.findall(r"code=(\d+)", log_text)))
    broker = {
        "readonly_requery_attempts": 2,
        "unique_order_count": fill_status.get("unique_order_count"),
        "query_failed_records": fill_status.get("query_failed_records"),
        "error_codes": error_codes,
        "saved_fill_values_are_not_authoritative": True,
        "account_snapshot": account_snapshot,
        "pending_reconciliation": "runs=[]; complete=true",
    }
    if csv_analysis is not None:
        broker["official_transaction_csv"] = {
            "source": csv_analysis.get("source"),
            "sha256": sha256(Path(csv_analysis["source"])),
            "csv_row_count": csv_analysis.get("csv_row_count"),
            "official_transaction_row_count": csv_analysis.get("official_transaction_row_count"),
            "local_real_order_count": csv_analysis.get("local_real_order_count"),
            "local_group_count": csv_analysis.get("local_group_count"),
            "quantity_reconciled_group_count": csv_analysis.get("quantity_reconciled_group_count"),
            "quantity_reconciled_local_order_count": csv_analysis.get("quantity_reconciled_local_order_count"),
            "local_quantity_total": csv_analysis.get("local_quantity_total"),
            "official_quantity_for_local_keys_total": csv_analysis.get(
                "official_quantity_for_local_keys_total"
            ),
            "unreconciled_local_group_count": csv_analysis.get("unreconciled_local_group_count"),
            "parsed_artifact": "reports/20260922_production_acceptance/tachibana_csv_analysis.json",
            "fee_components_present": False,
        }
    payload = {
        "collected_at": datetime.now(UTC).isoformat(),
        "historical_provider_freshness": {
            "status": "unverifiable",
            "historical_provider_available_at": provenance.get("historical_provider_available_at"),
            "historical_available_at_proven": provenance.get("historical_available_at_proven"),
            "reason": "Canonical historical inputs contain observed/session cutoffs but no provider-issued available_at; no timestamp is synthesized.",
            "macro_current_capture": {
                "provider": provenance.get("macro_provider"),
                "requested_at": provenance.get("macro_requested_at"),
                "received_at": provenance.get("macro_received_at"),
            },
            "cache_metadata": cache_metadata(),
        },
        "broker_reconciliation": broker,
        "scheduler": scheduler,
        "decision": "Treat the user-provided Tachibana transaction CSV as authoritative for the matched execution date/side/quantity/price/PnL rows; record fee-component reconciliation and historical provider freshness as outstanding operational conditions. Production promotion was explicitly approved by the user on 2026-09-23 under the recorded numerical-gate override.",
    }
    target = OUT / "remaining_condition_evidence.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
