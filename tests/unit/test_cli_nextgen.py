"""Unit tests for Next-Gen CLI Promotion (decision, close, daily).

Tests CLI dispatcher integration with AsyncExecutionEngine and NextGenPipeline.
"""

from __future__ import annotations

import argparse

from leadlag.cli import _handle_close, _handle_decision, setup_parser


def test_cli_parser_includes_engine_option():
    """Verify CLI parser supports --engine nextgen/v2 for decision and close."""
    parser = setup_parser()

    # decision subparser
    args_dec = parser.parse_args(["decision", "--engine", "nextgen"])
    assert args_dec.engine == "nextgen"

    args_dec_v2 = parser.parse_args(["decision", "--engine", "v2"])
    assert args_dec_v2.engine == "v2"

    # close subparser
    args_close = parser.parse_args(["close", "--engine", "nextgen"])
    assert args_close.engine == "nextgen"


def test_cli_decision_nextgen_dry_run(tmp_path):
    """Test decision command in dry-run mode using nextgen engine."""
    args = argparse.Namespace(
        engine="nextgen",
        config="configs/production/production.yaml",
        trade_date="2026-08-14",
        capital=500_000.0,
        capital_from_wallet=False,
        api_enable=False,
        api_dry_run=True,
        dry_run=True,
        output_root=str(tmp_path / "results"),
        auto_close=False,
    )

    exit_code = _handle_decision(args)
    assert exit_code == 0

    # Verify decision artifact directory was created
    decision_dir = tmp_path / "results" / "decision_20260814"
    assert decision_dir.exists()
    assert (decision_dir / "latest_weights.csv").exists()
    assert (decision_dir / "decision_summary.json").exists()


def test_cli_close_nextgen():
    """Test close command using nextgen async engine."""
    args = argparse.Namespace(
        engine="nextgen",
    )

    exit_code = _handle_close(args)
    assert exit_code == 0
