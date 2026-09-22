"""Gap matrix I/O helpers shared by production and signal-enhancement modules."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leadlag.data.gap_store import GapStore, is_gap_store_path
from leadlag.data.tickers import JP_TICKERS
from leadlag.data.validation import DataValidationError, validate_gap_matrices
from leadlag.domain.gap_bundle import (
    GapBundleRef,
    canonical_json_bytes,
)
from leadlag.utils.distribution_provenance import validate_distribution_provenance
from leadlag.utils.gap_provenance import validate_bundle_identity

logger = logging.getLogger(__name__)


def _npy_manifest_ref(raw: dict[str, Any], metadata_bytes: bytes | None) -> GapBundleRef:
    """Recover a legacy horizon only from digest-verified provenance metadata."""
    if raw.get("format_version", 1) == 1 and "schema_version" not in raw and "horizon" not in raw:
        if metadata_bytes is not None:
            if hashlib.sha256(metadata_bytes).hexdigest() != raw.get("metadata_sha256"):
                raise ValueError("metadata digest mismatch")
            metadata = json.loads(metadata_bytes)
            if not isinstance(metadata, dict):
                raise ValueError("metadata root must be a JSON object")
            horizon = metadata.get("horizon")
            # The default h=1 filenames historically used horizon=None.
            raw = {**raw, "horizon": None if horizon == 1 else horizon}
    return GapBundleRef.from_dict(raw)


def _format_gap_date(date_str: str) -> str:
    """Convert any parseable date string to the YYYYMMDD gap file suffix."""
    return str(pd.to_datetime(date_str).strftime("%Y%m%d"))


def _npy_bundle_metadata_path(
    gap_input_dir: Path,
    date_str: str,
    mu_pattern: str,
    pattern_kwargs: dict | None,
) -> Path:
    """Return the sidecar path paired with one ``mu_gap`` file.

    ``gap_metadata[_hN]_YYYYMMDD.json`` is published only after its μ/Ω pair
    is complete, so a reader that sees an interrupted ``.npy`` publication
    cannot treat it as a provenanced distribution.
    """
    formatted = Path(mu_pattern.format(date=_format_gap_date(date_str), **(pattern_kwargs or {})))
    mu_path = gap_input_dir / formatted
    name = mu_path.name
    if name.startswith("mu_gap_"):
        return mu_path.with_name("gap_metadata_" + name.removeprefix("mu_gap_")).with_suffix(
            ".json"
        )
    return mu_path.with_suffix(".metadata.json")


def _npy_bundle_manifest_path(
    gap_input_dir: Path,
    date_str: str,
    mu_pattern: str,
    pattern_kwargs: dict | None,
) -> Path:
    """Return the commit marker for one μ/Ω/metadata bundle."""
    formatted = Path(mu_pattern.format(date=_format_gap_date(date_str), **(pattern_kwargs or {})))
    mu_path = gap_input_dir / formatted
    return mu_path.with_name(f".{mu_path.stem}.bundle.json")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_array_bytes(data: np.ndarray) -> bytes:
    """Return stable array bytes for SQLite bundle identity."""
    return np.ascontiguousarray(np.asarray(data)).tobytes()


def _bundle_provenance_error(
    metadata: dict[str, Any] | None,
    date_str: str,
    horizon: int | None,
) -> str | None:
    """Return a provenance error for a bundle used by a decision path."""
    _normalized, error = validate_distribution_provenance(
        metadata,
        date_str,
        1 if horizon is None else horizon,
        label="Gap bundle provenance",
    )
    return error


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write one file atomically and fsync its contents before publication."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _parse_pattern_to_matrix_type(
    file_pattern: str,
    pattern_kwargs: dict | None,
) -> tuple[str | None, int | None]:
    """Map a filename pattern like ``matrices/mu_gap_h{h}_{date}.npy`` to a
    (matrix_type, horizon) pair for the SQLite store.
    """
    pattern_kwargs = pattern_kwargs or {}
    basename = Path(file_pattern).name
    if "rank_reversal" in basename:
        matrix_type = "rank_reversal"
    elif "mu_gap" in basename:
        matrix_type = "mu"
    elif "omega_gap" in basename:
        matrix_type = "omega"
    else:
        return None, None

    horizon = pattern_kwargs.get("h")
    if horizon is not None and "h{h}" not in basename and "{h}" not in basename:
        horizon = None
    if horizon is not None:
        horizon = int(horizon)
    return matrix_type, horizon


def _try_load_gap_from_store(
    gap_input_dir: Path,
    date_str: str,
    file_pattern: str,
    pattern_kwargs: dict | None,
    required: bool = False,
) -> tuple[np.ndarray | None, list[str]]:
    """Try to load a gap matrix from a SQLite ``GapStore``.

    Returns (array, alerts).  If *gap_input_dir* is not a store path or the
    matrix is not found, returns (None, [alert]) so the caller can fall back.
    """
    if not is_gap_store_path(gap_input_dir):
        return None, []

    try:
        store = GapStore(gap_input_dir)
    except Exception as e:
        return None, [f"GapStore open failed for {gap_input_dir}: {e}"]

    matrix_type, horizon = _parse_pattern_to_matrix_type(file_pattern, pattern_kwargs)
    if matrix_type is None:
        return None, [f"Cannot map pattern {file_pattern!r} to a GapStore matrix type"]

    arr = store.get(date_str, matrix_type, horizon=horizon)
    if arr is None:
        alert = f"GapStore missing {matrix_type} (h={horizon}) for {date_str}"
        if required:
            logger.warning(alert)
        else:
            logger.debug(alert)
        return None, [alert]
    return arr, []


def _try_save_gap_to_store(
    gap_output_dir: Path,
    date_str: str,
    file_pattern: str,
    pattern_kwargs: dict | None,
    data: np.ndarray,
) -> bool:
    """Write a single matrix to a SQLite ``GapStore`` if *gap_output_dir* is one.

    Returns True if the store path was used, False otherwise (caller should
    fall back to ``.npy``).
    """
    if not is_gap_store_path(gap_output_dir):
        return False

    try:
        store = GapStore(gap_output_dir)
    except Exception as e:
        logger.warning("GapStore open failed for %s: %s", gap_output_dir, e)
        return False

    matrix_type, horizon = _parse_pattern_to_matrix_type(file_pattern, pattern_kwargs)
    if matrix_type is None:
        logger.warning(
            "Cannot map pattern %r to a GapStore matrix type; skipping store write",
            file_pattern,
        )
        return False

    try:
        store.put(date_str, matrix_type, data, horizon=horizon)
    except Exception as e:
        logger.warning("Failed to write %s to GapStore: %s", file_pattern, e)
        return False
    return True


def load_gap_npy(
    gap_input_dir: Path,
    date_str: str,
    file_pattern: str,
    pattern_kwargs: dict | None = None,
    *,
    required: bool = False,
) -> tuple[np.ndarray | None, list[str]]:
    """Load a single gap matrix from a ``.npy`` file or a SQLite ``GapStore``.

    Args:
        gap_input_dir: Root directory containing the file, or a ``.sqlite``
            gap store file.
        date_str: Trade date in any format parseable by ``pd.to_datetime``.
        file_pattern: Path template with ``{date}`` placeholder and optional
            additional named placeholders (e.g. ``{h}`` for horizon).
        pattern_kwargs: Optional extra format arguments for *file_pattern*.
        required: If True, a missing file is logged at WARNING; otherwise DEBUG.

    Returns:
        Tuple of (array, alerts).  Array is ``None`` when the matrix is missing
        or cannot be loaded.
    """
    pattern_kwargs = pattern_kwargs or {}

    # 1. Try SQLite gap store first (canonical source).
    arr, alerts = _try_load_gap_from_store(
        gap_input_dir, date_str, file_pattern, pattern_kwargs, required=required
    )
    if arr is not None:
        return arr, []
    if alerts:
        return None, alerts

    # 2. Fall back to per-date .npy files (test / legacy compatibility).
    date_numeric = _format_gap_date(date_str)
    file_path = gap_input_dir / file_pattern.format(date=date_numeric, **pattern_kwargs)

    if not file_path.exists():
        alert = f"Gap file missing: {file_path}"
        if required:
            logger.warning(alert)
        else:
            logger.debug(alert)
        return None, [alert]

    try:
        arr = np.load(file_path)
    except Exception as e:
        alert = f"Failed to load {file_path}: {e}"
        logger.warning(alert)
        return None, [alert]

    return arr, []


def save_gap_npy(
    gap_output_dir: Path,
    date_str: str,
    data: np.ndarray,
    file_pattern: str,
    pattern_kwargs: dict | None = None,
) -> bool:
    """Save a single gap matrix to a ``.npy`` file or SQLite ``GapStore``.

    Args:
        gap_output_dir: Root directory for the output file, or a ``.sqlite``
            gap store file.
        date_str: Trade date in any format parseable by ``pd.to_datetime``.
        data: Numpy array to store.
        file_pattern: Path template with ``{date}`` placeholder and optional
            additional named placeholders.
        pattern_kwargs: Optional extra format arguments for *file_pattern*.

    Returns:
        True if the matrix was written successfully.
    """
    pattern_kwargs = pattern_kwargs or {}

    # 1. Try SQLite gap store first (canonical sink).
    if _try_save_gap_to_store(gap_output_dir, date_str, file_pattern, pattern_kwargs, data):
        return True

    # 2. Fall back to per-date .npy file (test / legacy compatibility).
    date_numeric = _format_gap_date(date_str)
    file_path = gap_output_dir / file_pattern.format(date=date_numeric, **pattern_kwargs)
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(file_path, data)
    except Exception as e:
        logger.warning("Failed to save %s: %s", file_path, e)
        return False
    return True


def load_gap_bundle(
    gap_input_dir: Path,
    date_str: str,
    mu_pattern: str = "matrices/mu_gap_{date}.npy",
    omega_pattern: str = "matrices/omega_gap_{date}.npy",
    pattern_kwargs: dict | None = None,
    *,
    strict: bool = False,
    require_metadata: bool = False,
    expected_identity: Mapping[str, Any] | None = None,
    require_identity: bool = False,
    n_j: int = len(JP_TICKERS),
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any] | None, list[str]]:
    """Load a μ/Ω/metadata bundle from one consistent source snapshot.

    Args:
        gap_input_dir: Root directory containing the gap matrix files, or a
            ``.sqlite`` gap store file.
        date_str: Trade date in any format parseable by ``pd.to_datetime``.
        mu_pattern: File pattern for the mu matrix.  Must contain ``{date}``.
        omega_pattern: File pattern for the Omega matrix.  Must contain ``{date}``.
        pattern_kwargs: Optional extra format arguments for the patterns
            (e.g. ``{"h": 3}`` for horizon-aware patterns like
            ``matrices/mu_gap_h{h}_{date}.npy``).
        strict: If True, raise ``DataValidationError`` when the matrices are
            missing, have the wrong shape, or fail basic invariants. If False,
            return a tuple with ``None`` arrays and alerts for unusable input
            so the caller can choose its fallback flow.
        require_metadata: If True, a committed bundle without a metadata
            object is considered unusable.  This is required by decision and
            research paths that need point-in-time provenance; the default
            remains permissive for low-level matrix inspection.
        expected_identity: Optional input/config/model/ticker identity to
            compare with the loaded bundle metadata.
        require_identity: If True, require all four identity fields even when
            no expected values are supplied.  This is used for newly written
            production artifacts while preserving low-level legacy readers.
        n_j: Expected number of JP assets for shape validation.

    Returns:
        Tuple of (mu_gap, Omega_gap, metadata, alerts).  Both arrays are ``None`` when
        either file is missing or cannot be loaded (non-strict mode).

    Raises:
        DataValidationError: When ``strict=True`` and validation fails.
    """
    pattern_kwargs = pattern_kwargs or {}
    horizon = pattern_kwargs.get("h")
    # Do not coerce untrusted metadata-like values here.  Horizon validation
    # below must turn NaN/Inf/array values into a normal rejection alert.
    is_h1_required = horizon is None or (
        isinstance(horizon, (int, np.integer))
        and not isinstance(horizon, (bool, np.bool_))
        and int(horizon) == 1
    )
    alerts: list[str] = []
    metadata: dict[str, Any] | None = None
    manifest_ref: GapBundleRef | None = None

    # A SQLite store must be read through the bundled API.  Calling
    # load_gap_npy twice opens two connections and can mix μ from one commit
    # with Ω from the next commit.
    if is_gap_store_path(gap_input_dir):
        mu_type, mu_horizon = _parse_pattern_to_matrix_type(mu_pattern, pattern_kwargs)
        omega_type, omega_horizon = _parse_pattern_to_matrix_type(omega_pattern, pattern_kwargs)
        if mu_type == "mu" and omega_type == "omega" and mu_horizon == omega_horizon:
            try:
                store = GapStore(gap_input_dir)
                mu_gap, Omega_gap, metadata, manifest_ref = store.load_horizon_bundle(
                    date_str, horizon=mu_horizon
                )
                if mu_gap is None:
                    alerts.append(
                        f"GapStore missing mu/omega bundle (h={mu_horizon}) for {date_str}"
                    )
            except Exception as exc:
                mu_gap, Omega_gap = None, None
                alerts.append(f"GapStore bundle load failed for {date_str}: {exc}")
        else:
            mu_gap, mu_alerts = load_gap_npy(
                gap_input_dir, date_str, mu_pattern, pattern_kwargs, required=is_h1_required
            )
            Omega_gap, omega_alerts = load_gap_npy(
                gap_input_dir, date_str, omega_pattern, pattern_kwargs, required=is_h1_required
            )
            alerts.extend(mu_alerts + omega_alerts)
    else:
        mu_gap, mu_alerts = load_gap_npy(
            gap_input_dir, date_str, mu_pattern, pattern_kwargs, required=is_h1_required
        )
        Omega_gap, omega_alerts = load_gap_npy(
            gap_input_dir, date_str, omega_pattern, pattern_kwargs, required=is_h1_required
        )
        alerts.extend(mu_alerts + omega_alerts)

    if strict and (mu_gap is None or Omega_gap is None):
        raise DataValidationError("; ".join(alerts) if alerts else "Gap matrices unavailable")

    if mu_gap is not None and Omega_gap is not None:
        if manifest_ref is not None:
            actual_metadata_bytes = (
                canonical_json_bytes(metadata) if metadata is not None else None
            )
            manifest_errors = manifest_ref.validate(
                trade_date=date_str,
                horizon=(None if horizon is None else int(horizon)),
                storage_format="sqlite",
                mu_sha256=_sha256_bytes(_canonical_array_bytes(mu_gap)),
                omega_sha256=_sha256_bytes(_canonical_array_bytes(Omega_gap)),
                metadata_sha256=(
                    _sha256_bytes(actual_metadata_bytes)
                    if actual_metadata_bytes is not None
                    else None
                ),
            )
            if manifest_errors:
                alerts.append(
                    "[FATAL] Gap bundle consistency check failed: "
                    + "; ".join(manifest_errors)
                )
                if strict:
                    raise DataValidationError("; ".join(alerts))
                return None, None, None, alerts
        if not is_gap_store_path(gap_input_dir):
            mu_path = gap_input_dir / mu_pattern.format(
                date=_format_gap_date(date_str), **pattern_kwargs
            )
            omega_path = gap_input_dir / omega_pattern.format(
                date=_format_gap_date(date_str), **pattern_kwargs
            )
            metadata_path = _npy_bundle_metadata_path(
                gap_input_dir, date_str, mu_pattern, pattern_kwargs
            )
            manifest_path = _npy_bundle_manifest_path(
                gap_input_dir, date_str, mu_pattern, pattern_kwargs
            )
            bundle_valid = False
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    mu_bytes = mu_path.read_bytes()
                    omega_bytes = omega_path.read_bytes()
                    metadata_bytes = metadata_path.read_bytes() if metadata_path.exists() else None
                    manifest_ref = _npy_manifest_ref(manifest, metadata_bytes)
                    manifest_errors = manifest_ref.validate(
                        trade_date=date_str,
                        horizon=(None if horizon is None else int(horizon)),
                        storage_format="npy",
                        mu_sha256=_sha256_bytes(mu_bytes),
                        omega_sha256=_sha256_bytes(omega_bytes),
                        metadata_sha256=(
                            _sha256_bytes(metadata_bytes) if metadata_bytes is not None else None
                        ),
                    )
                    if manifest_errors:
                        raise ValueError("; ".join(manifest_errors))
                    # Decode the same bytes whose digests were verified.  This
                    # avoids validating one generation and returning another
                    # when a writer publishes concurrently.
                    loaded_mu = np.load(io.BytesIO(mu_bytes))
                    loaded_omega = np.load(io.BytesIO(omega_bytes))
                    if metadata_bytes is not None:
                        loaded_metadata = json.loads(metadata_bytes.decode("utf-8"))
                        if not isinstance(loaded_metadata, dict):
                            raise TypeError("metadata root must be a JSON object")
                        metadata = loaded_metadata
                    else:
                        metadata = None
                    mu_gap = loaded_mu
                    Omega_gap = loaded_omega
                    bundle_valid = True
                except Exception as exc:  # noqa: BLE001
                    metadata = None
                    alerts.append(
                        f"[FATAL] Gap bundle consistency check failed: {manifest_path}: {exc}"
                    )
            else:
                alerts.append(
                    f"[FATAL] Gap bundle provenance missing; commit manifest missing: {manifest_path}"
                )

            if strict and not bundle_valid:
                raise DataValidationError("; ".join(alerts))
            if not bundle_valid:
                # A pair whose commit marker or byte digests do not validate is
                # not a distribution.  Returning either array would let
                # compatibility callers pair a new μ with an old Ω (or use a
                # provenance-free generation) while only retaining a warning.
                return None, None, None, alerts
        if require_metadata:
            normalized_metadata, provenance_error = validate_distribution_provenance(
                metadata,
                date_str,
                1 if horizon is None else horizon,
                label="Gap bundle provenance",
            )
            if provenance_error:
                alert = f"[FATAL] {provenance_error}"
                alerts.append(alert)
                if strict:
                    raise DataValidationError("; ".join(alerts))
                return None, None, None, alerts
            metadata = normalized_metadata
        identity_errors = validate_bundle_identity(
            metadata,
            expected_identity,
            require=require_identity or expected_identity is not None,
        )
        if identity_errors:
            alerts.extend(f"[FATAL] {error}" for error in identity_errors)
            if strict:
                raise DataValidationError("; ".join(alerts))
            return None, None, None, alerts
        v_alerts = validate_gap_matrices(mu_gap, Omega_gap, n_j=n_j)
        if v_alerts:
            if strict:
                raise DataValidationError("; ".join(v_alerts))
            alerts.extend(v_alerts)
            if any(alert.startswith("[FATAL]") for alert in v_alerts):
                return None, None, None, alerts
        return mu_gap, Omega_gap, metadata, alerts
    return None, None, metadata, alerts


def load_gap_matrices(
    gap_input_dir: Path,
    date_str: str,
    mu_pattern: str = "matrices/mu_gap_{date}.npy",
    omega_pattern: str = "matrices/omega_gap_{date}.npy",
    pattern_kwargs: dict | None = None,
    *,
    strict: bool = False,
    require_metadata: bool = True,
    expected_identity: Mapping[str, Any] | None = None,
    require_identity: bool = False,
    n_j: int = len(JP_TICKERS),
) -> tuple[np.ndarray | None, np.ndarray | None, list[str]]:
    """Compatibility μ/Ω loader backed by :func:`load_gap_bundle`.

    Provenance is required by default because this three-value API cannot
    return metadata to a caller.  Set ``require_metadata=False`` only for
    explicitly diagnostic, low-level matrix inspection.  ``expected_identity``
    and ``require_identity`` apply the same bundle identity checks as
    :func:`load_gap_bundle`.
    """
    mu_gap, Omega_gap, _metadata, alerts = load_gap_bundle(
        gap_input_dir,
        date_str,
        mu_pattern=mu_pattern,
        omega_pattern=omega_pattern,
        pattern_kwargs=pattern_kwargs,
        strict=strict,
        require_metadata=require_metadata,
        expected_identity=expected_identity,
        require_identity=require_identity,
        n_j=n_j,
    )
    return mu_gap, Omega_gap, alerts


def load_gap_bundle_manifest(
    gap_input_dir: Path,
    date_str: str,
    mu_pattern: str = "matrices/mu_gap_{date}.npy",
    omega_pattern: str = "matrices/omega_gap_{date}.npy",
    pattern_kwargs: dict | None = None,
) -> tuple[GapBundleRef | None, list[str]]:
    """Read the publication reference without loading matrix payloads.

    This is useful for cache identity and run manifests.  It intentionally
    does not turn a manifest into a usable distribution; callers that need
    arrays must still use :func:`load_gap_bundle`, which verifies the payload
    digests and shape/provenance contracts.
    """
    pattern_kwargs = pattern_kwargs or {}
    horizon = pattern_kwargs.get("h")
    if horizon is not None:
        try:
            horizon = int(horizon)
        except (TypeError, ValueError) as exc:
            return None, [f"invalid horizon in gap bundle pattern: {exc}"]

    if is_gap_store_path(gap_input_dir):
        mu_type, mu_horizon = _parse_pattern_to_matrix_type(mu_pattern, pattern_kwargs)
        omega_type, omega_horizon = _parse_pattern_to_matrix_type(omega_pattern, pattern_kwargs)
        if mu_type != "mu" or omega_type != "omega" or mu_horizon != omega_horizon:
            return None, ["gap bundle patterns do not describe one matching horizon"]
        try:
            _mu, _omega, _metadata, sqlite_manifest = GapStore(gap_input_dir).load_horizon_bundle(
                date_str,
                horizon=mu_horizon,
            )
        except Exception as exc:  # noqa: BLE001
            return None, [f"GapStore manifest load failed for {date_str}: {exc}"]
        if sqlite_manifest is None:
            return None, [f"GapStore manifest missing for {date_str}"]
        return sqlite_manifest, []

    manifest_path = _npy_bundle_manifest_path(
        gap_input_dir,
        date_str,
        mu_pattern,
        pattern_kwargs,
    )
    if not manifest_path.exists():
        return None, [f"Gap bundle manifest missing: {manifest_path}"]
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata_path = _npy_bundle_metadata_path(gap_input_dir, date_str, mu_pattern, pattern_kwargs)
        manifest_ref = _npy_manifest_ref(raw, metadata_path.read_bytes() if metadata_path.exists() else None)
        errors = manifest_ref.validate(
            trade_date=date_str,
            horizon=horizon,
            storage_format="npy",
            mu_sha256=manifest_ref.mu_sha256,
            omega_sha256=manifest_ref.omega_sha256,
            metadata_sha256=manifest_ref.metadata_sha256,
        )
        return manifest_ref, errors
    except Exception as exc:  # noqa: BLE001
        return None, [f"Gap bundle manifest invalid: {exc}"]


def save_gap_matrices(
    gap_output_dir: Path,
    date_str: str,
    mu_gap: np.ndarray,
    omega_gap: np.ndarray,
    mu_pattern: str = "matrices/mu_gap_{date}.npy",
    omega_pattern: str = "matrices/omega_gap_{date}.npy",
    pattern_kwargs: dict | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Save a pair of mu_gap / Omega_gap matrices.

    If *gap_output_dir* is a ``.sqlite`` file the matrices are written through
    :class:`GapStore`; otherwise per-date ``.npy`` files are written under the
    supplied directory.  The ``latest/`` directory of ``.npy`` files is
    preserved for backward compatibility.

    Args:
        gap_output_dir: Directory or SQLite gap store path.
        date_str: Trade date in any format parseable by ``pd.to_datetime``.
        mu_gap: Expected-return vector.
        omega_gap: Covariance matrix.
        mu_pattern: File pattern for the mu matrix.
        omega_pattern: File pattern for the Omega matrix.
        pattern_kwargs: Optional extra format arguments for horizon-aware
            patterns (e.g. ``{"h": 3}``).
        metadata: Optional dict of metadata (sig_date, etc.) stored with the
            pair, including an adjacent JSON sidecar for ``.npy`` output.

    Returns:
        True if both matrices were written successfully.
    """
    pattern_kwargs = pattern_kwargs or {}

    # 1. Try SQLite gap store first (canonical sink).
    if is_gap_store_path(gap_output_dir):
        try:
            store = GapStore(gap_output_dir)
        except Exception as e:
            logger.warning("GapStore open failed for %s: %s", gap_output_dir, e)
            return False

        mu_type, mu_horizon = _parse_pattern_to_matrix_type(mu_pattern, pattern_kwargs)
        omega_type, omega_horizon = _parse_pattern_to_matrix_type(
            omega_pattern, pattern_kwargs
        )

        if mu_type is None or omega_type is None:
            logger.warning(
                "Cannot map mu/omega patterns to GapStore types; skipping store write"
            )
            return False

        # If both patterns are the default (no horizon) pair, use the bundled
        # ``save`` API so metadata is stored together with mu/omega.
        if mu_horizon is None and omega_horizon is None:
            try:
                stored_metadata = metadata if metadata is not None else {}
                manifest = GapBundleRef.from_payload(
                    trade_date=date_str,
                    horizon=None,
                    storage_format="sqlite",
                    mu_bytes=_canonical_array_bytes(mu_gap),
                    omega_bytes=_canonical_array_bytes(omega_gap),
                    metadata_bytes=canonical_json_bytes(stored_metadata),
                    metadata=stored_metadata,
                )
                store.save(
                    date_str,
                    mu_gap,
                    omega_gap,
                    metadata=metadata,
                    manifest=manifest,
                )
            except Exception as e:
                logger.warning("Failed to save gap pair to GapStore: %s", e)
                return False
            return True

        # Horizon-aware: store each matrix with its horizon.  Metadata, if any,
        # is stored under the 'meta' type with the same horizon.
        try:
            if mu_type == "mu" and omega_type == "omega" and mu_horizon == omega_horizon:
                stored_metadata = metadata if metadata is not None else {}
                store.save_horizon(
                    date_str,
                    mu_gap,
                    omega_gap,
                    metadata=metadata,
                    horizon=mu_horizon,
                    manifest=GapBundleRef.from_payload(
                        trade_date=date_str,
                        horizon=mu_horizon,
                        storage_format="sqlite",
                        mu_bytes=_canonical_array_bytes(mu_gap),
                        omega_bytes=_canonical_array_bytes(omega_gap),
                        metadata_bytes=canonical_json_bytes(stored_metadata),
                        metadata=stored_metadata,
                    ),
                )
            else:
                store.put(date_str, mu_type, mu_gap, horizon=mu_horizon)
                store.put(date_str, omega_type, omega_gap, horizon=omega_horizon)
                if metadata is not None:
                    store.put(date_str, "meta", metadata, horizon=mu_horizon)
        except Exception as e:
            logger.warning("Failed to save horizon gap pair to GapStore: %s", e)
            return False
        return True

    # 2. Fall back to per-date .npy files (test / legacy compatibility).
    date_numeric = _format_gap_date(date_str)
    mu_path = gap_output_dir / mu_pattern.format(date=date_numeric, **pattern_kwargs)
    omega_path = gap_output_dir / omega_pattern.format(date=date_numeric, **pattern_kwargs)
    metadata_path = _npy_bundle_metadata_path(
        gap_output_dir, date_str, mu_pattern, pattern_kwargs
    )
    manifest_path = _npy_bundle_manifest_path(
        gap_output_dir, date_str, mu_pattern, pattern_kwargs
    )
    try:
        # The manifest is the commit marker.  Readers accept a pair only when
        # the exact bytes they loaded match the last published marker.  A
        # writer may therefore replace μ, Ω, or metadata in any order without
        # exposing a mixed generation as a valid bundle.
        mu_buffer = io.BytesIO()
        np.save(mu_buffer, mu_gap)
        omega_buffer = io.BytesIO()
        np.save(omega_buffer, omega_gap)
        mu_bytes = mu_buffer.getvalue()
        omega_bytes = omega_buffer.getvalue()
        metadata_bytes: bytes | None = None
        if metadata is not None:
            metadata_bytes = json.dumps(
                metadata, ensure_ascii=False, indent=2, default=str
            ).encode("utf-8")

        _atomic_write_bytes(mu_path, mu_bytes)
        _atomic_write_bytes(omega_path, omega_bytes)
        if metadata_bytes is None:
            try:
                metadata_path.unlink()
            except FileNotFoundError:
                pass
        else:
            _atomic_write_bytes(metadata_path, metadata_bytes)

        manifest_payload = GapBundleRef.from_payload(
            trade_date=date_str,
            horizon=(
                None
                if pattern_kwargs.get("h") is None
                else int(pattern_kwargs["h"])
            ),
            storage_format="npy",
            mu_bytes=mu_bytes,
            omega_bytes=omega_bytes,
            metadata_bytes=metadata_bytes,
            metadata=metadata,
        ).to_dict()
        _atomic_write_bytes(
            manifest_path,
            json.dumps(manifest_payload, ensure_ascii=False, sort_keys=True).encode("utf-8"),
        )
    except Exception as e:
        logger.warning("Failed to save gap matrices to %s / %s: %s", mu_path, omega_path, e)
        return False
    return True
