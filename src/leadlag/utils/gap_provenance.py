"""Stable provenance identities for gap-distribution bundles.

The gap matrices are derived artifacts.  Their sidecar metadata must identify
the historical input available at the trade date, the effective V2 config, the
model family, and the JP ticker order used to lay out the arrays.  This module
keeps the producer and cache consumer on the same canonical representation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from leadlag.data.tickers import JP_TICKERS
from leadlag.utils.dataframe_fingerprint import dataframe_fingerprint
from leadlag.utils.timestamps import normalize_jst_date

MODEL_VERSION = "production_residual_blpx_v2"


def _jsonable(value: Any) -> Any:
    """Return a JSON-compatible representation for a config object."""
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except TypeError:
            return value.model_dump()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def config_version(config: Any) -> str:
    """Return a deterministic SHA-256 identity for an effective config."""
    payload = _jsonable(config)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def model_version(model: Any | None = None) -> str:
    """Return the stable model family identifier used by bundle producers."""
    # The model class is deliberately not included: importing a refactored
    # implementation must not make a valid cache unverifiable when the model
    # family and effective config are unchanged.
    return MODEL_VERSION


def input_version(df_exec: pd.DataFrame, trade_date: Any) -> str:
    """Fingerprint the execution inputs available at the 09:10 decision cut."""
    date = normalize_jst_date(trade_date)
    if date not in df_exec.index:
        raise ValueError(f"trade_date {date.date()} is not present in df_exec")
    asof = df_exec.loc[:date].copy()
    # Match HistoricalInputs.calculation_frame(09:10): current-day close
    # labels are unavailable when the cache is consumed, even though a
    # research producer may hold the completed frame for target evaluation.
    label_prefixes = ("jp_oc_", "jp_cc_", "topix_oc", "topix_cc", "target_", "y_jp_")
    label_columns = [column for column in asof.columns if str(column).startswith(label_prefixes)]
    if label_columns:
        asof.loc[date, label_columns] = float("nan")
    return dataframe_fingerprint(asof)


def open_910_version(open_910_returns: pd.DataFrame, trade_date: Any) -> str:
    """Fingerprint run-owned 09:10 inputs available through ``trade_date``.

    Gap bundles depend on the 5-minute 09:10 observation even when the
    execution frame itself is unchanged.  Restricting the fingerprint to the
    as-of slice keeps future corrections from invalidating a historical bundle
    while still rejecting a stale bundle for the current decision date.
    """
    if not isinstance(open_910_returns, pd.DataFrame):
        raise TypeError("open_910_returns must be a pandas.DataFrame")
    date = normalize_jst_date(trade_date)
    if date not in open_910_returns.index:
        raise ValueError(f"trade_date {date.date()} is not present in open_910_returns")
    return dataframe_fingerprint(open_910_returns.loc[:date].copy())


def gap_inputs_version(
    trade_date: Any,
    gap_returns: Any,
    betas: Any,
    topix_night_return: Any,
    *,
    horizon: int = 1,
) -> str:
    """Fingerprint the point-in-time inputs used by gap adjustment.

    The execution frame and 09:10 return frame do not contain live snapshot
    values in all production paths.  Include those values explicitly so a
    cached matrix cannot be reused after the observed gap or beta changes.
    """
    digest = hashlib.sha256()
    header = json.dumps(
        {
            "trade_date": normalize_jst_date(trade_date).isoformat(),
            "horizon": int(horizon),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(header)
    for label, values in (
        ("gap_returns", gap_returns),
        ("betas", betas),
        ("topix_night_return", np.asarray([topix_night_return])),
    ):
        array = np.asarray(values, dtype=np.float64)
        digest.update(label.encode("utf-8"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def bundle_identity(
    df_exec: pd.DataFrame,
    trade_date: Any,
    *,
    config: Any,
    model: Any | None = None,
    open_910_returns: pd.DataFrame | None = None,
    gap_inputs: tuple[Any, Any, Any] | None = None,
    horizon: int = 1,
) -> dict[str, Any]:
    """Build the identity fields stored in one gap bundle sidecar."""
    identity = {
        "input_version": input_version(df_exec, trade_date),
        "model_version": model_version(model),
        "config_version": config_version(config),
        "ticker_order": list(JP_TICKERS),
    }
    if open_910_returns is not None:
        identity["open_910_version"] = open_910_version(open_910_returns, trade_date)
    if gap_inputs is not None:
        identity["gap_inputs_version"] = gap_inputs_version(
            trade_date, *gap_inputs, horizon=horizon
        )
    return identity


def validate_bundle_identity(
    metadata: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None = None,
    *,
    require: bool = False,
) -> list[str]:
    """Validate required identity fields and, when supplied, their values."""
    if not isinstance(metadata, Mapping):
        return ["Gap bundle identity metadata is missing"] if (require or expected) else []

    errors: list[str] = []
    fields = ["input_version", "model_version", "config_version", "ticker_order"]
    if expected is not None and "open_910_version" in expected:
        fields.append("open_910_version")
    if expected is not None and "gap_inputs_version" in expected:
        fields.append("gap_inputs_version")
    if require or expected is not None:
        for field in fields:
            value = metadata.get(field)
            if field == "ticker_order":
                if not isinstance(value, (list, tuple)) or not value:
                    errors.append("Gap bundle ticker_order is missing or empty")
            elif value is None or not str(value).strip():
                errors.append(f"Gap bundle {field} is missing")

    if expected is not None:
        for field in fields:
            actual = metadata.get(field)
            wanted = expected.get(field)
            actual_value: Any
            wanted_value: Any
            if field == "ticker_order":
                actual_value = tuple(actual) if isinstance(actual, (list, tuple)) else ()
                wanted_value = tuple(wanted) if isinstance(wanted, (list, tuple)) else ()
            else:
                actual_value = None if actual is None else str(actual)
                wanted_value = None if wanted is None else str(wanted)
            if actual_value != wanted_value:
                errors.append(f"Gap bundle {field} mismatch")
    return errors


__all__ = [
    "MODEL_VERSION",
    "bundle_identity",
    "config_version",
    "gap_inputs_version",
    "input_version",
    "open_910_version",
    "model_version",
    "validate_bundle_identity",
]
