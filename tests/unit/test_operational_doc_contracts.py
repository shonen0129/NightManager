"""Tie maintained operating tables to resolved configuration and real behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from leadlag.broker.base import Position
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.close import close_all_positions
from leadlag.execution.config import load_config_from_yaml
from leadlag.models.v2.audit_comparator import _run_safety_audits

ROOT = Path(__file__).resolve().parents[2]


def _table(document: str, heading: str) -> dict[str, list[str]]:
    lines = (ROOT / document).read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"| {heading} |"))
    rows = {}
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        assert cells[0] not in rows, f"duplicate documented contract: {cells[0]}"
        rows[cells[0]] = cells[1:]
    assert rows, f"empty contract table: {document} {heading}"
    return rows


@pytest.mark.parametrize("document", ["docs/日次運用手順書.md", "docs/モデル技術仕様書.md"])
def test_documented_configuration_matches_resolved_production(document):
    config = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
    rows = _table(document, "Config attribute")
    # Removing the risk/carry/model contract from a table must fail too.
    required = (
        {"strategy.overnight_alpha_long", "strategy.overnight_alpha_short",
         "v2.costs.overnight_alpha_long", "v2.costs.overnight_alpha_short",
         "v2.fallback_on_audit_failure", "v2.ondemand_fallback_enabled",
         "strategy.side_leverage", "v2.costs.side_leverage",
         "risk.max_gross_exposure", "risk.max_net_exposure"}
        if "日次" in document else
        {"v2.blpx.blp_window", "v2.blpx.blp_ewma_halflife", "v2.ml_overlay_enabled"}
    )
    assert required <= rows.keys()
    for attribute, cells in rows.items():
        actual = config
        for part in attribute.split("."):
            actual = getattr(actual, part)
        expected = yaml.safe_load(cells[0])
        assert actual == expected, f"{document}: {attribute}: resolved={actual}, documented={expected}"
        assert isinstance(actual, type(expected)), f"{document}: {attribute}: wrong value type"


def test_operational_audit_cases_match_real_flattening_and_statuses():
    rows = _table("docs/日次運用手順書.md", "Audit case")
    assert rows.keys() == {"normal", "leakage_failed", "weights_invalid", "scores_invalid", "gap_missing"}
    cfg = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True).v2
    n = len(JP_TICKERS)
    for case, expected in rows.items():
        weights = np.zeros(n)
        weights[:2] = [0.5, -0.5]
        scores = np.ones(n)
        if case == "weights_invalid":
            weights[0] = 1.0
        if case == "scores_invalid":
            scores[0] = np.nan
        if case == "gap_missing":
            weights[:] = 0.0
        result = _run_safety_audits(
            weights, scores, np.zeros(n), np.eye(n), np.ones(n), None,
            "2026-10-08", "2026-10-08" if case == "leakage_failed" else "2026-10-07",
            cfg, {"gap_data_missing": case == "gap_missing", "audit_failure": False},
            {"multiplier": 1.0, "assigned_bin": "Medium", "threshold_low": 0.0, "threshold_high": 1.0},
            [], np.array(["2026-10-06"], dtype="datetime64[D]"), "test", "v2",
        )
        outcome = "flat" if np.all(result.w_final == 0.0) else "retained"
        assert [outcome, result.leakage["status"], result.numerical["status"],
                str(result.fallback["audit_failure"]).lower()] == expected, case
        if case in ("leakage_failed", "weights_invalid", "scores_invalid"):
            assert result.alerts, case


def test_documented_carry_rates_produce_rounded_planned_inventory(tmp_path):
    rows = _table("docs/日次運用手順書.md", "Config attribute")
    # 1629.T has a 10-share lot. A 330-share holding demonstrates rounding.
    positions = [
        Position(ticker="1629.T", side=side, quantity=330, price=1000.0,
                 exchange=27, execution_id=f"P-{side}", margin_trade_type=3, account_type=4)
        for side in ("BUY", "SELL")
    ]
    broker = SimpleNamespace(get_positions=lambda: positions)
    result = close_all_positions(
        broker, tmp_path, dry_run=True,
        overnight_alpha_long=float(rows["strategy.overnight_alpha_long"][0]),
        overnight_alpha_short=float(rows["strategy.overnight_alpha_short"][0]),
    )
    assert [(p["side"], p["hold_quantity"]) for p in result["held_overnight"]] == [
        ("BUY", 250), ("SELL", 170),
    ]
    assert [(p["side"], p["quantity"]) for p in result["close_results"]] == [
        ("SELL", 80), ("BUY", 160),
    ]
    assert all(p["status"] == "SIMULATED" for p in result["close_results"])
    # Planned/dry-run close is not evidence of the account's actual inventory.
    assert [p.quantity for p in positions] == [330, 330]
