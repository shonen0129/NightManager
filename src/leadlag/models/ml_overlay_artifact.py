"""Versioned persistence and provenance validation for ML overlay artifacts.

The artifact layout is intentionally small and immutable: ``CURRENT`` points
to one directory under ``versions/`` containing both the pickle and its JSON
metadata.  Readers never combine files from different versions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from leadlag.data.tickers import JP_TICKERS
from leadlag.utils.timestamps import normalize_jst_date

logger = logging.getLogger("leadlag.models.ml_order_overlay")

_OVERLAY_METADATA_VERSION = 2
_OVERLAY_REQUIRED_PROVENANCE_FIELDS = (
    "metadata_version",
    "metadata_status",
    "train_start",
    "train_end",
    "data_hash",
    "config_hash",
)
_OVERLAY_EMBEDDED_MATCH_FIELDS = (
    "artifact_version",
    "metadata_version",
    "metadata_status",
    "train_start",
    "train_end",
    "label_asof_end",
    "data_hash",
    "config_hash",
    "cont_cols",
    "target_std",
    "use_ticker",
    "use_classification",
    "per_ticker_interactions",
    "n_tickers",
    "p_trade_scale",
)


def datetime_now_for_artifact() -> str:
    """Return a filesystem-safe UTC timestamp for an artifact version."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync after an atomic publication."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        logger.debug("Directory fsync unavailable for %s", path)


def _normalize_overlay_date(value: Any, field_name: str, context: str) -> str:
    """Return a date-only provenance value or reject an ambiguous one."""
    if isinstance(value, (list, tuple, set, dict, pd.Series, pd.Index, np.ndarray)):
        raise ValueError(f"{context} {field_name} must be one scalar date")
    if value is None:
        raise ValueError(f"{context} {field_name} is required")
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{context} {field_name} is not a valid date: {value!r}") from exc
    if pd.isna(parsed):
        raise ValueError(f"{context} {field_name} must not be null or NaT")
    return cast(str, normalize_jst_date(parsed).strftime("%Y-%m-%d"))


def _validate_overlay_provenance(metadata: dict[str, Any], context: str) -> dict[str, Any]:
    """Validate and normalize the provenance required for safe application."""
    if not isinstance(metadata, dict):
        raise ValueError(f"{context} metadata must be a JSON object")
    missing = [field for field in _OVERLAY_REQUIRED_PROVENANCE_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"{context} metadata is missing required provenance: {', '.join(missing)}")
    if metadata.get("metadata_version") != _OVERLAY_METADATA_VERSION:
        raise ValueError(
            f"{context} metadata_version must be {_OVERLAY_METADATA_VERSION}, "
            f"got {metadata.get('metadata_version')!r}"
        )
    if metadata.get("metadata_status") != "verified":
        raise ValueError(f"{context} metadata_status must be 'verified'")

    normalized = dict(metadata)
    normalized["train_start"] = _normalize_overlay_date(
        metadata["train_start"], "train_start", context
    )
    normalized["train_end"] = _normalize_overlay_date(metadata["train_end"], "train_end", context)
    if normalized["train_start"] > normalized["train_end"]:
        raise ValueError(f"{context} train_start must not be after train_end")
    if "label_asof_end" in metadata:
        normalized["label_asof_end"] = _normalize_overlay_date(
            metadata["label_asof_end"], "label_asof_end", context
        )
        if normalized["label_asof_end"] > normalized["train_end"]:
            raise ValueError(f"{context} label_asof_end must not be after train_end")
    for field_name in ("data_hash", "config_hash"):
        value = metadata[field_name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{context} {field_name} must be a non-empty string")
        normalized[field_name] = value.strip()
    return normalized


def save_overlay_model(
    model: Any,
    output_dir: Path,
    training_metadata: dict[str, Any] | None = None,
) -> None:
    """Publish one immutable model/provenance pair under ``CURRENT``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "cont_cols": model.cont_cols,
        "target_std": float(model.target_std),
        "use_ticker": model.use_ticker,
        "use_classification": model.use_classification,
        "per_ticker_interactions": model.per_ticker_interactions,
        "n_tickers": len(JP_TICKERS),
        "p_trade_scale": float(getattr(model, "p_trade_scale", 1.0)),
        "metadata_version": _OVERLAY_METADATA_VERSION,
    }
    if training_metadata:
        structural_fields = set(metadata)
        for key, value in training_metadata.items():
            if key in structural_fields and value != metadata[key]:
                raise ValueError(f"Training metadata conflicts with fitted overlay field: {key}")
            metadata[key] = value
    metadata = _validate_overlay_provenance(metadata, "Overlay artifact publication")

    version_id = f"{datetime_now_for_artifact()}-{uuid.uuid4().hex[:12]}"
    metadata["artifact_version"] = version_id
    object.__setattr__(model, "metadata", dict(metadata))
    staging_dir = output_dir / f".staging-{version_id}"
    version_dir = output_dir / "versions" / version_id
    staging_dir.mkdir(parents=True, exist_ok=False)
    version_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        model_bytes = pickle.dumps(model)
        metadata["model_sha256"] = hashlib.sha256(model_bytes).hexdigest()
        model_tmp = staging_dir / "model.pkl"
        metadata_tmp = staging_dir / "metadata.json"
        with model_tmp.open("wb") as handle:
            handle.write(model_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        with metadata_tmp.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        staging_dir.rename(version_dir)
        current_tmp = output_dir / f".CURRENT-{version_id}"
        with current_tmp.open("w", encoding="utf-8") as handle:
            handle.write(version_id + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(current_tmp, output_dir / "CURRENT")
        _fsync_directory(output_dir)
    except Exception:
        logger.exception("Failed to publish overlay artifact version %s", version_id)
        raise
    logger.info("Overlay model saved to %s", output_dir)


def load_overlay_model(model_dir: Path) -> Any:
    """Load and fully validate the active overlay artifact."""
    model_dir = Path(model_dir)
    current_path = model_dir / "CURRENT"
    if not current_path.exists():
        if (model_dir / "model.pkl").exists() or (model_dir / "metadata.json").exists():
            raise ValueError(
                f"Legacy root overlay artifact is not accepted: {model_dir}. "
                "Retrain and publish a versioned artifact with CURRENT."
            )
        raise FileNotFoundError(f"Overlay artifact CURRENT pointer not found: {current_path}")

    active_version = current_path.read_text(encoding="utf-8").strip()
    if (
        not active_version
        or active_version in {".", ".."}
        or Path(active_version).name != active_version
    ):
        raise ValueError(f"Invalid active overlay artifact version pointer: {current_path}")
    versions_dir = model_dir / "versions"
    artifact_dir = versions_dir / active_version
    try:
        if artifact_dir.resolve().parent != versions_dir.resolve():
            raise ValueError(
                f"Active overlay artifact version escapes versions directory: {current_path}"
            )
    except OSError as exc:
        raise ValueError(f"Cannot resolve active overlay artifact: {current_path}: {exc}") from exc
    if not artifact_dir.is_dir():
        raise ValueError(f"Active overlay artifact version is missing: {artifact_dir}")

    model_path = artifact_dir / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Overlay model not found: {model_path}")
    model_bytes = model_path.read_bytes()
    model = pickle.loads(model_bytes)

    # Import lazily so importing production inference never imports research or
    # the compatibility facade while a pickle is being read.
    from leadlag.models.ml_order_overlay import MLOrderOverlayModel

    if not isinstance(model, MLOrderOverlayModel):
        raise TypeError(f"Loaded model is not an MLOrderOverlayModel: {type(model)}")
    embedded_metadata = getattr(model, "metadata", {})
    if not isinstance(embedded_metadata, dict):
        embedded_metadata = {}
    metadata_path = artifact_dir / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"Overlay artifact metadata is missing: {metadata_path}")
    try:
        loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata = _validate_overlay_provenance(
            loaded_metadata, f"Overlay artifact metadata {metadata_path}"
        )
    except Exception as exc:
        raise ValueError(f"Invalid overlay metadata: {metadata_path}: {exc}") from exc

    if metadata.get("artifact_version") != active_version:
        raise ValueError(
            f"Overlay metadata version mismatch for {artifact_dir}: "
            f"expected {active_version}, got {metadata.get('artifact_version')}"
        )
    expected_hash = metadata.get("model_sha256")
    actual_hash = hashlib.sha256(model_bytes).hexdigest()
    if expected_hash != actual_hash:
        raise ValueError(
            f"Overlay model digest mismatch for {artifact_dir}: "
            f"expected {expected_hash}, got {actual_hash}"
        )
    try:
        embedded_metadata = _validate_overlay_provenance(
            embedded_metadata, f"Overlay model metadata {model_path}"
        )
    except Exception as exc:
        raise ValueError(f"Invalid embedded overlay metadata: {model_path}: {exc}") from exc
    for key in _OVERLAY_EMBEDDED_MATCH_FIELDS:
        if embedded_metadata.get(key) != metadata.get(key):
            raise ValueError(f"Overlay model/metadata field mismatch for {artifact_dir}: {key}")
    object.__setattr__(model, "metadata", metadata)
    logger.info("Overlay model loaded from %s", model_dir)
    return model


__all__ = [
    "_normalize_overlay_date",
    "_validate_overlay_provenance",
    "datetime_now_for_artifact",
    "load_overlay_model",
    "save_overlay_model",
]
