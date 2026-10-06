"""Exercise a locally installed wheel without exposing repository source."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path


def main() -> int:
    installation = Path(sys.argv[1]).resolve()
    runtime = tempfile.TemporaryDirectory(prefix="leadlag-wheel-runtime-")
    os.environ["LEADLAG_RUNTIME_ROOT"] = runtime.name
    # Reuse the locked dependency environment, but exclude its editable source
    # path. The leadlag package must come exclusively from the installed wheel.
    sys.path[:] = [str(installation)] + [
        item for item in sys.path
        if item and not (Path(item) / "leadlag").is_dir()
        and not (Path(item) / "research").is_dir()
    ]
    import numpy as np
    import pandas as pd
    from sklearn.dummy import DummyRegressor

    import leadlag
    import leadlag.cli
    from leadlag.config.paths import project_root, results
    from leadlag.config.schemas import AppConfig
    from leadlag.core.blpx_math import safe_solve_inverse
    from leadlag.data.adr_features import (
        DEFAULT_ADR_FEATURES_PATH,
        load_adr_features,
        publish_adr_features,
    )
    from leadlag.data.adr_producer import build_adr_features
    from leadlag.data.macro import _PERSISTED_MACRO_PATH
    from leadlag.data.tickers import ADR_TICKERS, JP_TICKERS
    from leadlag.models.ml_order_overlay import MLOrderOverlayModel
    from leadlag.models.ml_overlay_artifact import load_overlay_model, save_overlay_model
    from leadlag.models.ml_overlay_features import _predict_relative_allocation
    from leadlag.runner.model_factory import resolve_overlay_settings

    assert Path(leadlag.__file__).resolve().is_relative_to(installation)
    runtime_root = Path(runtime.name).resolve()
    assert project_root() == runtime_root
    assert results().is_relative_to(runtime_root)
    assert DEFAULT_ADR_FEATURES_PATH == runtime_root / "data/adr_features.zip"
    closes = pd.DataFrame({ticker: [100.0, 110.0] for ticker in ADR_TICKERS},
                          index=pd.to_datetime(["2026-09-30", "2026-10-01"]))
    execution = pd.DataFrame({"sig_date": [pd.Timestamp("2026-10-01")]},
                             index=pd.to_datetime(["2026-10-02"]))
    adr_frame = build_adr_features(execution, closes)
    publish_adr_features(adr_frame, required_trade_date="2026-10-02")
    loaded_adr = load_adr_features(trade_date="2026-10-02")
    pd.testing.assert_frame_equal(loaded_adr, adr_frame[[f"adr_{ticker}" for ticker in JP_TICKERS]])
    solve, inverse, fallback = safe_solve_inverse(np.full((2, 2), np.inf), np.ones((1, 2)))
    assert fallback and np.isfinite(solve).all() and np.isfinite(inverse).all()
    assert _PERSISTED_MACRO_PATH == runtime_root / "var/market_data/macro_prices_verified.pkl"
    _, relative_model = resolve_overlay_settings(AppConfig(), overlay_model_dir="models/overlay")
    assert relative_model == runtime_root / "models/overlay"
    assert importlib.util.find_spec("research") is None
    assert not any(key == "research" or key.startswith("research.") for key in sys.modules)
    try:
        leadlag.cli.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    features = pd.DataFrame({"score": [0.2, 0.8]})
    fitted = DummyRegressor(strategy="constant", constant=0.5).fit(features, [0, 1])
    model = MLOrderOverlayModel(fitted, ["score"], 1.0, False, False, False)
    expected = _predict_relative_allocation(features, model)
    with tempfile.TemporaryDirectory(prefix="leadlag-wheel-artifact-") as directory:
        artifact = Path(directory)
        save_overlay_model(model, artifact, training_metadata={
            "metadata_status": "verified", "train_start": "2015-01-05",
            "train_end": "2020-12-31", "data_hash": "synthetic-wheel-smoke",
            "config_hash": "synthetic-wheel-smoke",
        })
        restored = load_overlay_model(artifact)
        np.testing.assert_array_equal(_predict_relative_allocation(features, restored), expected)
        assert type(restored).__module__ == "leadlag.models.ml_order_overlay"
    assert not any(key == "research" or key.startswith("research.") for key in sys.modules)
    print(json.dumps({"wheel_import": str(leadlag.__file__), "cli_help": "pass",
                      "artifact_roundtrip_inference": "pass", "adr_producer_roundtrip": "pass",
                      "shared_blpx_math": "pass", "research_import": "absent"}))
    runtime.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
