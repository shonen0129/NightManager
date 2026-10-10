"""Offline smoke for the optional wallet Sharpe research command."""
import importlib.util
import json
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[2] / "tools/research/calculate_live_sharpe.py"


def load():
    spec = importlib.util.spec_from_file_location("wallet_proxy_sharpe", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_input(tmp_path, capsys):
    assert load().main(["--results-dir", str(tmp_path)]) == 0
    assert "No close wallet snapshots found" in capsys.readouterr().out


def test_observations_and_date_range(tmp_path, capsys):
    directory = tmp_path / "20260715_production_close_positions"
    directory.mkdir()
    for day, value in [("20260715", 100), ("20260716", 110), ("20260717", 105), ("20260720", 120)]:
        (directory / f"wallet_close_{day}.json").write_text(json.dumps({"ukeire_hosyoukin": value}))
    mod = load()
    assert mod.main(["--results-dir", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "Close snapshots: 4" in output
    assert "NOT realized net P&L" in output
    assert mod.main(["--results-dir", str(tmp_path), "--start-date", "2026-07-16", "--end-date", "2026-07-20"]) == 0
    assert "Close snapshots: 3" in capsys.readouterr().out
    assert mod.main(["--results-dir", str(tmp_path), "--start-date", "2026-07-20", "--end-date", "2026-07-20"]) == 0
    assert "Need at least three" in capsys.readouterr().out
