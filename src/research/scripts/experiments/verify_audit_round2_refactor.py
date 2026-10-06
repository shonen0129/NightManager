"""Compare audit round-2 changes against an independent committed baseline."""

from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from leadlag.core import blpx_math
from leadlag.core.pnl import simulate_daily_pnl
from leadlag.data.adr_producer import build_adr_features
from leadlag.data.tickers import ADR_SECTOR_MAP, ADR_TICKERS, JP_TICKERS
from leadlag.models.blpx import ProductionBLPXModel
from research.models.sector_relative_ensemble_blp_enhanced import (
    SectorRelativeEnsembleBLPEnhancedModel,
)


def _baseline(revision: str, path: str) -> str:
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], text=True)


def _load_source(source: str, name: str, directory: Path):
    path = directory / f"{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _equal(before, after):
    if isinstance(before, dict):
        assert before.keys() == after.keys()
        for key in before:
            _equal(before[key], after[key])
    elif isinstance(before, (tuple, list)):
        assert len(before) == len(after)
        for old, new in zip(before, after):
            _equal(old, new)
    elif before is None:
        assert after is None
    else:
        np.testing.assert_allclose(before, after, rtol=0, atol=0, equal_nan=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="fd82e99e")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(613)
    outcomes = {
        "baseline": args.baseline,
        "numeric_tolerance": 0,
        "blpx_cases": 0,
        "pnl_cases": 0,
        "adr_observed_case": False,
        "singular_and_nonfinite_cases": 0,
        "research_diagnostic_key_change": {"z_U": "z_U_t"},
        "nonfinite_fallback_corrections": 0,
    }
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        old = _load_source(
            _baseline(
                args.baseline, "src/research/models/sector_relative_ensemble_blp_enhanced.py"
            ),
            "audit_before_research",
            directory,
        )
        old_pnl = _load_source(
            _baseline(args.baseline, "src/leadlag/core/pnl.py"), "audit_before_pnl", directory
        )
        old_solver = _load_source(
            _baseline(args.baseline, "src/leadlag/models/blpx/blp_solver.py"),
            "audit_before_solver",
            directory,
        )
        _load_source(
            _baseline(args.baseline, "src/leadlag/models/blpx/signal_computer.py").replace(
                "leadlag.models.blpx.blp_solver", "audit_before_solver"
            ),
            "audit_before_signal",
            directory,
        )
        _load_source(
            _baseline(args.baseline, "src/leadlag/models/blpx/model_meta.py"),
            "audit_before_meta",
            directory,
        )
        old_production = _load_source(
            _baseline(args.baseline, "src/leadlag/models/blpx/model.py")
            .replace("leadlag.models.blpx.blp_solver", "audit_before_solver")
            .replace("leadlag.models.blpx.signal_computer", "audit_before_signal")
            .replace("leadlag.models.blpx.model_meta", "audit_before_meta"),
            "audit_before_production",
            directory,
        )
        for kind, before_class, after_class in [
            (
                "research",
                old.SectorRelativeEnsembleBLPEnhancedModel,
                SectorRelativeEnsembleBLPEnhancedModel,
            ),
            ("production", old_production.ProductionBLPXModel, ProductionBLPXModel),
        ]:
            for asymmetry in ["scalar", "covariance"]:
                for scaling in [False, True]:
                    cfg = {
                        "blp_window": 100,
                        "winsor_sigma": 4.0,
                        "rank": "4",
                        "lambda_pca": 0.2,
                        "lambda_sector": 0.15,
                        "beta_conf": 0.25,
                        "asymmetry_mode": asymmetry,
                        "frobenius_scale_priors": scaling,
                        "sector_eta": 0.2,
                    }
                    before = before_class(copy.deepcopy(cfg))
                    after = after_class(copy.deepcopy(cfg))
                    for invalid in [False, True]:
                        values = rng.normal(0, 0.01, (180, before.n_u + before.n_j))
                        if invalid:
                            values[50, 0] = np.nan
                            values[60, 16] = np.inf
                        basis, _ = np.linalg.qr(rng.normal(size=(32, 6)))
                        kwargs = {
                            "current_index": 140,
                            "v0_static": basis,
                            "c_full": np.eye(32),
                            "return_matrices": True,
                            "gap_override": np.zeros(17),
                            "betas_t": np.ones(17),
                            "topix_night_t": 0.0,
                        }
                        expected = before.compute_blp_signal(values.copy(), **copy.deepcopy(kwargs))
                        if kind == "research":
                            expected["z_U_t"] = expected.pop("z_U")
                        _equal(
                            expected,
                            after.compute_blp_signal(values.copy(), **copy.deepcopy(kwargs)),
                        )
                        outcomes["blpx_cases"] += 1
                        # Current JP targets and all future observations are unknown.
                        changed = values.copy()
                        changed[140:, before.n_u :] = 12345
                        changed[141:, : before.n_u] = -9999
                        baseline = after.compute_blp_signal(values.copy(), **copy.deepcopy(kwargs))
                        _equal(baseline, after.compute_blp_signal(changed, **copy.deepcopy(kwargs)))
        for matrix in [
            np.zeros((2, 2)),
            np.eye(2),
            np.full((2, 2), np.nan),
            np.full((2, 2), np.inf),
        ]:
            expected = old_solver._safe_solve_inv(matrix, np.ones((1, 2)))
            actual = blpx_math.safe_solve_inverse(matrix, np.ones((1, 2)))
            if not np.isfinite(expected[0]).all() or not np.isfinite(expected[1]).all():
                assert np.isfinite(actual[0]).all() and np.isfinite(actual[1]).all() and actual[2]
                outcomes["nonfinite_fallback_corrections"] += 1
            else:
                _equal(expected, actual)
            outcomes["singular_and_nonfinite_cases"] += 1
        for long_carry, short_carry in [(0, 0), (0.75, 0.5), (1, 1)]:
            kwargs = {
                "weights": rng.normal(size=(6, 3)),
                "target_returns": rng.normal(0, 0.01, (6, 3)),
                "gap_returns": rng.normal(0, 0.01, (6, 3)),
                "sim_dates": pd.bdate_range("2026-09-28", periods=6),
                "slip": 0.0005,
                "financing_daily": 0.03 / 365,
                "borrow_daily": 0.01 / 365,
                "reverse_daily": 0.0001,
                "alpha_long": long_carry,
                "alpha_short": short_carry,
                "side_leverage": 1.5,
            }
            _equal(
                old_pnl.simulate_daily_pnl(**copy.deepcopy(kwargs)),
                simulate_daily_pnl(**copy.deepcopy(kwargs)),
            )
            outcomes["pnl_cases"] += 1
        dates = pd.DatetimeIndex(["2026-10-01", "2026-10-02"])
        execution = pd.DataFrame(
            {"sig_date": dates}, index=pd.DatetimeIndex(["2026-10-02", "2026-10-05"])
        )
        close = pd.DataFrame(
            {ticker: [100.0, 110.0, 99.0] for ticker in ADR_TICKERS},
            index=pd.DatetimeIndex(["2026-09-30", "2026-10-01", "2026-10-02"]),
        )
        source = ast.parse(
            _baseline(args.baseline, "src/research/scripts/experiments/build_adr_features.py")
        )
        function = next(
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_adr_features"
        )
        raw = pd.concat(
            {ticker: pd.DataFrame({"Close": close[ticker]}) for ticker in close}, axis=1
        )
        context = {
            "np": np,
            "pd": pd,
            "ADR_MAP": ADR_SECTOR_MAP,
            "ALL_ADR_TICKERS": ADR_TICKERS,
            "JP_TICKERS": JP_TICKERS,
            "logger": logging.getLogger(__name__),
            "_yf_download": lambda *args: raw,
        }
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), "<baseline-adr>", "exec"), context
        )
        expected = context["build_adr_features"](execution.copy())
        actual = build_adr_features(execution.copy(), close.copy())[expected.columns]
        pd.testing.assert_frame_equal(expected, actual, check_freq=False)
        outcomes["adr_observed_case"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outcomes, indent=2) + "\n")
    print(json.dumps(outcomes))


if __name__ == "__main__":
    main()
