"""Content-addressed references for one gap-distribution bundle.

The μ vector, Ω matrix, provenance metadata, and the horizon they describe
form one publication unit.  This type is deliberately independent of the
storage implementation so SQLite and the compatibility ``.npy`` store can
share the same version vocabulary.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

_DATE_RE = re.compile(r"^(\d{4})-?(\d{2})-?(\d{2})$")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON metadata deterministically for a content digest."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return the full SHA-256 digest for *value*."""

    return hashlib.sha256(value).hexdigest()


def _normalise_date_key(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a date string")
    match = _DATE_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"{field} must use YYYY-MM-DD or YYYYMMDD")
    year, month, day = (int(part) for part in match.groups())
    try:
        date(year, month, day)
    except ValueError as exc:
        raise ValueError(f"{field} is not a valid calendar date") from exc
    return f"{year:04d}{month:02d}{day:02d}"


def _optional_version(metadata: Mapping[str, Any] | None, *keys: str) -> str | None:
    if not isinstance(metadata, Mapping):
        return None
    for key in keys:
        value = metadata.get(key)
        if value is not None and not isinstance(value, (dict, list, tuple)):
            text = str(value).strip()
            if text:
                return text
    return None


def _ticker_order(metadata: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(metadata, Mapping):
        return ()
    value = metadata.get("ticker_order")
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise ValueError("ticker_order must be a sequence of strings")
    try:
        order = tuple(str(item) for item in value)
    except TypeError as exc:
        raise ValueError("ticker_order must be a sequence of strings") from exc
    if len(order) != len(set(order)):
        raise ValueError("ticker_order must contain unique tickers")
    return order


@dataclass(frozen=True)
class GapBundleRef:
    """Immutable manifest for one μ/Ω/metadata publication.

    ``format_version=1`` is retained for compatibility with manifests already
    written by the workspace.  ``schema_version`` and the version fields are
    additive, so an older reader can still use the three original digests.
    ``horizon=None`` denotes the default h=1 bundle; an explicit integer is
    used for horizon-aware files.
    """

    trade_date: str
    horizon: int | None
    storage_format: str
    mu_sha256: str
    omega_sha256: str
    metadata_sha256: str | None
    schema_version: str = "gap-bundle-v1"
    input_version: str | None = None
    model_version: str | None = None
    config_version: str | None = None
    ticker_order: tuple[str, ...] = ()
    format_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "trade_date", _normalise_date_key(self.trade_date, "trade_date"))
        if self.horizon is not None:
            if isinstance(self.horizon, bool) or not isinstance(self.horizon, int):
                raise ValueError("horizon must be an integer or None")
            if self.horizon <= 0:
                raise ValueError("horizon must be positive")
        if self.storage_format not in {"npy", "sqlite"}:
            raise ValueError("storage_format must be 'npy' or 'sqlite'")
        if self.format_version not in {1, 2}:
            raise ValueError("unsupported bundle manifest")
        for field_name in ("mu_sha256", "omega_sha256"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{field_name} must be a SHA-256 hex digest")
        if self.metadata_sha256 is not None and (
            not isinstance(self.metadata_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", self.metadata_sha256)
        ):
            raise ValueError("metadata_sha256 must be a SHA-256 hex digest or None")
        if not isinstance(self.ticker_order, tuple):
            object.__setattr__(self, "ticker_order", tuple(self.ticker_order))

    @classmethod
    def from_payload(
        cls,
        *,
        trade_date: str,
        horizon: int | None,
        storage_format: str,
        mu_bytes: bytes,
        omega_bytes: bytes,
        metadata_bytes: bytes | None,
        metadata: Mapping[str, Any] | None = None,
    ) -> GapBundleRef:
        """Build a reference from the exact bytes that will be published."""

        return cls(
            trade_date=trade_date,
            horizon=horizon,
            storage_format=storage_format,
            mu_sha256=sha256_bytes(mu_bytes),
            omega_sha256=sha256_bytes(omega_bytes),
            metadata_sha256=(
                sha256_bytes(metadata_bytes) if metadata_bytes is not None else None
            ),
            input_version=_optional_version(
                metadata, "input_version", "input_fingerprint", "data_hash"
            ),
            model_version=_optional_version(
                metadata, "model_version", "model_revision", "code_revision"
            ),
            config_version=_optional_version(
                metadata, "config_version", "config_hash", "run_config_hash"
            ),
            ticker_order=_ticker_order(metadata),
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> GapBundleRef:
        """Parse a manifest, accepting the pre-S3c three-digest format."""

        if not isinstance(raw, Mapping):
            raise ValueError("unsupported bundle manifest")
        format_version = raw.get("format_version", 1)
        if isinstance(format_version, bool) or not isinstance(format_version, int):
            raise ValueError("unsupported bundle manifest")
        horizon = raw.get("horizon")
        if horizon in (None, -1):
            horizon = None
        elif isinstance(horizon, int) and not isinstance(horizon, bool):
            pass
        elif isinstance(horizon, float) and math.isfinite(horizon) and horizon.is_integer():
            horizon = int(horizon)
        else:
            raise ValueError("horizon must be an integer or None")
        ticker_order = raw.get("ticker_order", ())
        return cls(
            trade_date=str(raw.get("trade_date", "")),
            horizon=horizon,
            storage_format=str(raw.get("storage_format", raw.get("storage", "npy"))),
            mu_sha256=str(raw.get("mu_sha256", "")),
            omega_sha256=str(raw.get("omega_sha256", "")),
            metadata_sha256=(
                None if raw.get("metadata_sha256") is None else str(raw["metadata_sha256"])
            ),
            schema_version=str(raw.get("schema_version", "gap-bundle-v1")),
            input_version=(None if raw.get("input_version") is None else str(raw["input_version"])),
            model_version=(None if raw.get("model_version") is None else str(raw["model_version"])),
            config_version=(None if raw.get("config_version") is None else str(raw["config_version"])),
            ticker_order=tuple(str(item) for item in ticker_order),
            format_version=format_version,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready manifest mapping."""

        return {
            "format_version": self.format_version,
            "schema_version": self.schema_version,
            "trade_date": self.trade_date,
            "horizon": self.horizon,
            "storage_format": self.storage_format,
            "mu_sha256": self.mu_sha256,
            "omega_sha256": self.omega_sha256,
            "metadata_sha256": self.metadata_sha256,
            "input_version": self.input_version,
            "model_version": self.model_version,
            "config_version": self.config_version,
            "ticker_order": list(self.ticker_order),
        }

    def validate(
        self,
        *,
        trade_date: str,
        horizon: int | None,
        storage_format: str,
        mu_sha256: str,
        omega_sha256: str,
        metadata_sha256: str | None,
    ) -> list[str]:
        """Compare a loaded publication against its manifest."""

        expected_date = _normalise_date_key(trade_date, "trade_date")
        errors: list[str] = []
        if self.trade_date != expected_date:
            errors.append("trade date mismatch")
        if self.horizon != horizon:
            errors.append("horizon mismatch")
        if self.storage_format != storage_format:
            errors.append("storage format mismatch")
        if self.mu_sha256 != mu_sha256:
            errors.append("mu digest mismatch")
        if self.omega_sha256 != omega_sha256:
            errors.append("omega digest mismatch")
        if self.metadata_sha256 != metadata_sha256:
            errors.append("metadata digest mismatch")
        return errors


__all__ = ["GapBundleRef", "canonical_json_bytes", "sha256_bytes"]
