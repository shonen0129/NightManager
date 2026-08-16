"""Unit tests for Next-Gen CLI Promotion (decision, close, daily).

Tests CLI dispatcher integration with AsyncExecutionEngine and NextGenPipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from leadlag.cli import _handle_close, _handle_decision, setup_parser


def test_cli_parser_includes_engine_option() -> None:
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


def test_cli_decision_nextgen_simulation(tmp_path: Path) -> None:
    """Test decision command in simulation mode using nextgen engine (no live API)."""
    args = argparse.Namespace(
        engine="nextgen",
        config="configs/production/production.yaml",
        trade_date="2026-08-14",
        capital=500_000.0,
        capital_from_wallet=False,
        api_enable=False,
        api_dry_run=True,
        dry_run=False,
        output_root=str(tmp_path / "results"),
        live_dir=str(tmp_path / "live"),
        run_tag=None,
        gap_dir=None,
        text_output=False,
        jp_opens_csv=None,
        google_opens=False,
        api_url=None,
        api_token=None,
        auto_close=False,
    )

    exit_code = _handle_decision(args)
    assert exit_code == 0

    # Verify decision artifact directory was created
    decision_dir = tmp_path / "results" / "decision_20260814"
    assert decision_dir.exists()
    assert (decision_dir / "latest_weights.csv").exists()
    assert (decision_dir / "decision_summary.json").exists()
    assert (decision_dir / "production_audit.json").exists()

    # Verify live artifacts are mirrored to live_dir
    live_dir = tmp_path / "live"
    assert (live_dir / "latest_weights.csv").exists()


def test_cli_decision_nextgen_dry_run(tmp_path: Path) -> None:
    """Test decision command in dry-run mode (no files written except stub dir)."""
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
        live_dir=str(tmp_path / "live"),
        run_tag=None,
        gap_dir=None,
        text_output=False,
        jp_opens_csv=None,
        google_opens=False,
        api_url=None,
        api_token=None,
        auto_close=False,
    )

    exit_code = _handle_decision(args)
    assert exit_code == 0

    # In dry-run mode only the output stub directory is created; no files.
    decision_dir = tmp_path / "results" / "decision_20260814"
    assert decision_dir.exists()
    assert not (decision_dir / "latest_weights.csv").exists()


def test_cli_close_nextgen(tmp_path: Path) -> None:
    """Test close command using nextgen async engine."""
    args = argparse.Namespace(
        engine="nextgen",
        config="configs/production/production.yaml",
        output_root=str(tmp_path / "results"),
        run_tag=None,
        api_url=None,
        api_token=None,
        api_dry_run=True,
        api_enable=False,
        close_position_order=0,
    )

    exit_code = _handle_close(args)
    assert exit_code == 0

    # Verify close journal was saved
    close_dirs = [
        d for d in (tmp_path / "results").iterdir()
        if d.is_dir() and "production_close_positions" in d.name
    ]
    assert len(close_dirs) == 1
    assert (close_dirs[0] / "execution_journal.json").exists()
