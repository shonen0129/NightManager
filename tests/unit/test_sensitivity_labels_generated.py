"""Verify that generated sensitivity labels match the old hard-coded arrays."""

from __future__ import annotations

import numpy as np

from leadlag.core.correlation import get_static_sensitivity_labels
from leadlag.data.tickers import JP_TICKERS, SENSITIVITY_LABELS, US_TICKERS


def test_sensitivity_labels_generated_from_registry():
    labels = get_static_sensitivity_labels()
    for name in ("w3", "w4", "w5", "w6"):
        arr = labels[name]
        assert arr.shape == (len(US_TICKERS) + len(JP_TICKERS),)


def test_sensitivity_registry_has_all_tickers():
    for tk in US_TICKERS + JP_TICKERS:
        assert tk in SENSITIVITY_LABELS
        assert set(SENSITIVITY_LABELS[tk].keys()) == {"w3", "w4", "w5", "w6"}


def test_sensitivity_labels_against_legacy_values():
    """The legacy hard-coded arrays — this test pins them so a universe change
    cannot silently drift the prior subspace.
    """
    legacy = {
        "w3": np.array(
            [
                1.0, 0.3, 0.3, 1.0, 1.0, 0.6, -1.0, 0.3, -1.0, -1.0, 1.0, 0.0, 0.6, -0.3, -0.6,
                -1.0, 0.3, 0.6, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -0.3, -1.0, -0.3, 0.6, -0.6, 1.0, 0.6, 0.6,
            ],
            dtype=float,
        ),
        "w4": np.array(
            [
                0.3, 0.0, 0.0, 0.3, 0.6, 1.0, -0.6, -0.3, -0.6, -0.3, 0.6, 0.3, 0.0, 0.6, -0.3,
                -0.6, 0.3, 0.3, 0.6, -0.3, 1.0, 0.6, 1.0, 1.0, -0.3, -1.0, -0.3, 1.0, -0.6, 0.3, 0.0, -1.0,
            ],
            dtype=float,
        ),
        "w5": np.array(
            [
                0.3, 0.0, 1.0, 0.0, 0.3, 0.0, -0.3, 0.0, -1.0, 0.0, -0.3, 0.0, 0.3, 0.0, -0.3,
                -0.3, 1.0, 0.0, 0.3, 0.0, -0.3, 0.3, 0.0, 0.0, 0.0, -1.0, 0.0, 0.6, -0.3, 0.0, 0.0, 0.0,
            ],
            dtype=float,
        ),
        "w6": np.array(
            [
                1.0, -0.3, 1.0, 0.3, 0.3, -0.6, -0.3, 0.3, -0.6, -0.3, -0.3, 0.0, 0.6, -0.3, -0.3,
                -0.3, 1.0, 0.3, 0.6, -0.3, 0.0, 0.6, 0.3, -0.3, -0.3, -1.0, -0.3, 1.0, -0.6, 0.3, 0.0, 0.3,
            ],
            dtype=float,
        ),
    }
    generated = get_static_sensitivity_labels()
    for name in legacy:
        np.testing.assert_allclose(
            generated[name], legacy[name], rtol=1e-12, atol=1e-12, err_msg=f"{name} mismatch"
        )
