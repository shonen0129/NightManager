"""Tachibana API session cache for connection persistence across pipeline steps.

Saves decrypted virtual URLs and the p_no counter so that a subsequent process
can resume the session without re-performing PKI login. If the resumed session
is rejected by the API, the client falls back to normal login.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SESSION_MAX_AGE = timedelta(minutes=15)
_REQUIRED_URL_KEYS = {"sUrlRequest", "sUrlMaster", "sUrlPrice", "sUrlEvent"}


def _cache_root() -> Path:
    """Project root is 4 levels above this file (src/leadlag/broker/tachibana/...)."""
    return Path(__file__).resolve().parents[4]


def _session_cache_path() -> Path:
    path = _cache_root() / "live" / "pipeline_data" / "cache" / "tachibana" / "session.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _opens_cache_path(trade_date: str) -> Path:
    path = _cache_root() / "live" / "pipeline_data" / "cache" / "tachibana" / "opens"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{trade_date}.csv"


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as f:
        temp_path = Path(f.name)
        os.chmod(temp_path, 0o600)
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)
    os.chmod(path, 0o600)


def _atomic_csv_write(path: Path, df: pd.DataFrame) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as f:
        temp_path = Path(f.name)
        df.to_csv(f, index=False)
    os.replace(temp_path, path)


def save_session_cache(state: dict[str, Any]) -> None:
    """Persist TachibanaClient session state to disk."""
    path = _session_cache_path()
    try:
        _atomic_json_write(
            path,
            {
                "decrypted_urls": state.get("decrypted_urls", {}),
                "p_no": state.get("p_no", 1),
                "logged_in": state.get("logged_in", False),
                "saved_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        logger.info("[TACHIBANA-CACHE] Session saved to %s", path)
    except Exception as e:
        logger.warning("[TACHIBANA-CACHE] Failed to save session cache: %s", e)


def load_session_cache() -> dict[str, Any] | None:
    """Load a recent, structurally valid TachibanaClient session state."""
    path = _session_cache_path()
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        saved_at = datetime.fromisoformat(state["saved_at"])
        age = datetime.now(timezone.utc) - saved_at.astimezone(timezone.utc)
        urls = state.get("decrypted_urls", {})
        if (
            not state.get("logged_in")
            or not _REQUIRED_URL_KEYS.issubset(urls)
            or age < timedelta(0)
            or age > _SESSION_MAX_AGE
        ):
            logger.info("[TACHIBANA-CACHE] Session cache is stale or invalid; ignoring")
            clear_session_cache()
            return None
        logger.info("[TACHIBANA-CACHE] Loaded session from %s", path)
        return state
    except Exception as e:
        logger.warning("[TACHIBANA-CACHE] Failed to load session cache: %s", e)
        clear_session_cache()
        return None


def save_open_prices_cache(
    opens: dict[str, float],
    topix_open: float | None,
    trade_date: str,
) -> None:
    """Persist fetched open prices to a trade-date-specific CSV."""
    path = _opens_cache_path(trade_date)
    try:
        records = [{"ticker": tk, "open_price": price} for tk, price in opens.items()]
        if topix_open is not None:
            records.append({"ticker": "TOPIX", "open_price": topix_open})
        _atomic_csv_write(path, pd.DataFrame(records))
        logger.info("[TACHIBANA-CACHE] Open prices saved to %s", path)
    except Exception as e:
        logger.warning("[TACHIBANA-CACHE] Failed to save open prices cache: %s", e)


def load_open_prices_cache(
    trade_date: str,
) -> tuple[dict[str, float], float | None] | None:
    """Load cached open prices for exactly the requested trade date."""
    path = _opens_cache_path(trade_date)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if (
            set(df.columns) != {"ticker", "open_price"}
            or df.empty
            or df["ticker"].duplicated().any()
        ):
            return None
        prices = pd.to_numeric(df["open_price"], errors="coerce")
        if not np.isfinite(prices).all() or (prices <= 0).any():
            return None
        manual_opens: dict[str, float] = {}
        topix_open = None
        for ticker, price in zip(df["ticker"].astype(str), prices):
            if ticker == "TOPIX":
                topix_open = float(price)
            else:
                manual_opens[ticker] = float(price)
        return manual_opens, topix_open
    except Exception as e:
        logger.warning("[TACHIBANA-CACHE] Failed to load open prices cache: %s", e)
        return None


def clear_session_cache() -> None:
    """Remove session cache file, e.g. on explicit logout."""
    path = _session_cache_path()
    try:
        if path.exists():
            path.unlink()
    except Exception as e:
        logger.warning("[TACHIBANA-CACHE] Failed to clear session cache: %s", e)
