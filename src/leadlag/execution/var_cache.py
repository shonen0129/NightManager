"""Small, pure contracts used by the VaR/ES return cache.

The backtest and snapshot lifecycle stay in ``var_history`` because they own
process and filesystem cleanup.  Cache identity and the monotonic deadline
are value objects, so they can be tested without running a backtest.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeadlineBudget:
    """An absolute monotonic deadline shared by preparation and cache steps."""

    deadline: float

    @classmethod
    def from_timeout(cls, timeout: float) -> DeadlineBudget:
        return cls(time.monotonic() + max(0.01, float(timeout)))

    def remaining(self, *, label: str = "operation") -> float:
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError(f"{label} deadline exceeded")
        return left


@dataclass(frozen=True)
class VaRCacheIdentity:
    """All version inputs that can change a VaR/ES historical return series."""

    effective_config: Any
    start_date: str
    slippage_bps: float | None
    df_exec_hash: str
    code_hash: str
    overlay_identity: str
    gap_input_hash: str
    input_snapshot_hash: str = "none"

    def payload(self) -> dict[str, Any]:
        return {
            "config": self.effective_config,
            "start_date": self.start_date,
            "slippage_bps": self.slippage_bps,
            "df_exec_hash": self.df_exec_hash,
            "code_hash": self.code_hash,
            "overlay_artifact": self.overlay_identity,
            "gap_input": self.gap_input_hash,
            "input_snapshot": self.input_snapshot_hash,
        }

    @property
    def digest(self) -> str:
        """Return the stable full digest for this identity."""
        return hashlib.sha256(
            json.dumps(self.payload(), sort_keys=True, default=str).encode()
        ).hexdigest()

    @property
    def cache_key(self) -> str:
        """Return the existing compact SQLite key format."""
        return f"daily_returns:{self.digest[:16]}"


def build_var_cache_key(
    *,
    effective_config: Any,
    start_date: str,
    slippage_bps: float | None,
    df_exec_hash: str,
    code_hash: str,
    overlay_identity: str,
    gap_input_hash: str,
    input_snapshot_hash: str = "none",
) -> str:
    """Build a cache key from the exact inputs used by the VaR backtest."""

    return VaRCacheIdentity(
        effective_config=effective_config,
        start_date=start_date,
        slippage_bps=slippage_bps,
        df_exec_hash=df_exec_hash,
        code_hash=code_hash,
        overlay_identity=overlay_identity,
        gap_input_hash=gap_input_hash,
        input_snapshot_hash=input_snapshot_hash,
    ).cache_key


__all__ = ["DeadlineBudget", "VaRCacheIdentity", "build_var_cache_key"]
