"""Versioned input contracts for the production decision boundary.

The decision engine historically accepted a dataframe, a point-in-time lake,
an optional snapshot, and a separate price mapping.  Those values could
silently disagree.  This module gives the engine one explicit contract while
keeping the old arguments available at the outer compatibility boundary.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# JP session close is the earliest time at which close-derived labels are
# available to a later decision.  The contract intentionally keeps this
# schedule in one place so h=1/3/5 labels use the same PIT rule.
LABEL_AVAILABLE_CLOSE_TIME = pd.Timedelta(hours=15, minutes=30)


def _timestamp(value: Any, name: str) -> pd.Timestamp:
    """Coerce timestamps to the strategy's naive JST clock."""
    if value is None:
        raise ValueError(f"{name} is required")
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError(f"{name} must be a valid timestamp")
    if result.tzinfo is not None:
        result = result.tz_convert("Asia/Tokyo").tz_localize(None)
    return result


def _date(value: Any, name: str) -> pd.Timestamp:
    return _timestamp(value, name).normalize()


def _readonly_array(value: Any, name: str, *, dtype: Any = float) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    # A bytes-backed buffer cannot be made writable again via setflags.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _readonly_datetime_array(value: Any, name: str) -> np.ndarray:
    """Copy a one-dimensional date array into an immutable JST buffer.

    ``pd.to_datetime`` on a timezone-aware collection converts the values to
    UTC before the timezone is discarded when NumPy receives the result.  PIT
    history dates are calendar dates in the strategy's JST clock, so normalize
    each scalar through ``_timestamp`` before building the immutable array.
    """
    if isinstance(value, (str, bytes, pd.Timestamp, np.datetime64)) or np.isscalar(value):
        raise ValueError(f"{name} must be a one-dimensional array")
    try:
        items = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional array") from exc
    try:
        normalized = [_timestamp(item, f"{name}[{index}]") for index, item in enumerate(items)]
        array = np.array(normalized, dtype="datetime64[ns]", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain valid timestamps") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _readonly_mapping(value: Mapping[str, Any] | None, *, numeric: bool) -> Mapping[str, Any]:
    if value is None:
        value = {}
    converted: dict[str, Any]
    if numeric:
        converted = {str(key): float(item) for key, item in value.items()}
    else:
        converted = {str(key): str(item) for key, item in value.items()}
    return _FrozenMapping(converted)


class _FrozenMapping(Mapping[str, Any]):
    """Small picklable read-only mapping for cross-worker input snapshots."""

    __slots__ = ("_data",)

    def __init__(self, value: Mapping[str, Any]) -> None:
        self._data = dict(value)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        return (type(self), (self._data,))


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _array_digest(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "sha256": sha256(contiguous.tobytes()).hexdigest(),
    }


def _frame_digest(frame: pd.DataFrame) -> str:
    """Return a stable digest for the schema, index, and values of a frame."""
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype="uint64")
    return _digest(
        {
            "columns": [(str(column), str(frame[column].dtype)) for column in frame.columns],
            "index_dtype": str(frame.index.dtype),
            "rows": len(frame),
            "values": _array_digest(values),
        }
    )


@dataclass(frozen=True)
class InputVersion:
    """Content-addressed version of the inputs used for one decision."""

    schema_version: str
    source: str
    known_fingerprint: str
    historical_fingerprint: str
    evaluation_fingerprint: str | None = None

    @property
    def digest(self) -> str:
        return _digest(
            {
                "schema_version": self.schema_version,
                "source": self.source,
                "known": self.known_fingerprint,
                "historical": self.historical_fingerprint,
                "evaluation": self.evaluation_fingerprint,
            }
        )


@dataclass(frozen=True)
class KnownMarketInputs:
    """Values known at the decision timestamp for one execution date."""

    trade_date: pd.Timestamp
    as_of: pd.Timestamp
    ticker_order: tuple[str, ...]
    us_returns: np.ndarray
    jp_gap_returns: np.ndarray
    jp_betas: np.ndarray
    topix_night_return: float
    current_prices: Mapping[str, float] = field(default_factory=dict)
    prev_closes: Mapping[str, float] = field(default_factory=dict)
    macro_features: Mapping[str, float] = field(default_factory=dict)
    adr_features: Mapping[str, float] = field(default_factory=dict)
    sig_date: pd.Timestamp | None = None
    observed_at: Mapping[str, str] = field(default_factory=dict)
    source: str = "unknown"
    price_sources: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        trade_date = _date(self.trade_date, "trade_date")
        as_of = _timestamp(self.as_of, "as_of")
        if as_of.normalize() != trade_date:
            raise ValueError("as_of date must match trade_date in JST")
        decision_cutoff = trade_date + pd.Timedelta(hours=9, minutes=10)
        if as_of < decision_cutoff:
            raise ValueError("as_of must be at or after 09:10 JST")
        for name, value in self.observed_at.items():
            if _timestamp(value, f"observed_at[{name}]") > as_of:
                raise ValueError(f"observed_at[{name}] is later than as_of")
        sig_date = None if self.sig_date is None else _date(self.sig_date, "sig_date")
        if sig_date is not None and sig_date >= trade_date:
            raise ValueError("sig_date must be strictly earlier than trade_date")
        tickers = tuple(str(item) for item in self.ticker_order)
        if not tickers or len(set(tickers)) != len(tickers):
            raise ValueError("ticker_order must contain unique tickers")

        us_returns = _readonly_array(self.us_returns, "us_returns")
        jp_gap_returns = _readonly_array(self.jp_gap_returns, "jp_gap_returns")
        jp_betas = _readonly_array(self.jp_betas, "jp_betas")
        if len(jp_gap_returns) != len(tickers) or len(jp_betas) != len(tickers):
            raise ValueError("JP arrays must match ticker_order length")

        object.__setattr__(self, "trade_date", trade_date)
        object.__setattr__(self, "as_of", as_of)
        object.__setattr__(self, "sig_date", sig_date)
        object.__setattr__(self, "ticker_order", tickers)
        object.__setattr__(self, "us_returns", us_returns)
        object.__setattr__(self, "jp_gap_returns", jp_gap_returns)
        object.__setattr__(self, "jp_betas", jp_betas)
        object.__setattr__(self, "topix_night_return", float(self.topix_night_return))
        object.__setattr__(self, "current_prices", _readonly_mapping(self.current_prices, numeric=True))
        object.__setattr__(self, "prev_closes", _readonly_mapping(self.prev_closes, numeric=True))
        object.__setattr__(self, "macro_features", _readonly_mapping(self.macro_features, numeric=True))
        object.__setattr__(self, "adr_features", _readonly_mapping(self.adr_features, numeric=True))
        object.__setattr__(self, "observed_at", _readonly_mapping(self.observed_at, numeric=False))
        object.__setattr__(self, "price_sources", _readonly_mapping(self.price_sources, numeric=False))


    @property
    def fingerprint(self) -> str:
        return _digest(
            {
                "trade_date": self.trade_date.isoformat(),
                "as_of": self.as_of.isoformat(),
                "sig_date": None if self.sig_date is None else self.sig_date.isoformat(),
                "ticker_order": self.ticker_order,
                "us_returns": _array_digest(self.us_returns),
                "jp_gap_returns": _array_digest(self.jp_gap_returns),
                "jp_betas": _array_digest(self.jp_betas),
                "topix_night_return": self.topix_night_return,
                "current_prices": dict(self.current_prices),
                "prev_closes": dict(self.prev_closes),
                "macro_features": dict(self.macro_features),
                "adr_features": dict(self.adr_features),
                "observed_at": dict(self.observed_at),
                "source": self.source,
                "price_sources": dict(self.price_sources),
            }
        )


@dataclass(frozen=True, init=False)
class HistoricalInputs:
    """Historical frame and PIT training metadata owned by a decision run.

    The constructor takes deep copies.  Callers receive copies through
    ``to_frame``, ``calculation_frame``, and the optional feature-frame
    properties; none exposes the owned frames.
    """

    _frame: pd.DataFrame = field(repr=False)
    horizon: int
    baseline_start: pd.Timestamp
    baseline_end: pd.Timestamp
    pit_ir_history: np.ndarray | Mapping[str, np.ndarray] | None
    pit_history_trade_dates: np.ndarray | Mapping[str, np.ndarray] | None
    label_available_at: Mapping[str, str]
    observed_at: Mapping[str, str]
    observed_at_by_date: Mapping[str, Mapping[str, str]]
    _open_910_returns: pd.DataFrame | None
    _macro_prices: pd.DataFrame | None
    _adr_features: pd.DataFrame | None
    _rank_reversal_signals: pd.DataFrame | None
    source: str

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        horizon: int = 1,
        baseline_start: Any = "2010-01-01",
        baseline_end: Any = "2014-12-31",
        pit_ir_history: Any | None = None,
        pit_history_trade_dates: Any | None = None,
        label_available_at: Mapping[str, str] | None = None,
        observed_at: Mapping[str, str] | None = None,
        observed_at_by_date: Mapping[str, Mapping[str, str]] | None = None,
        open_910_returns: pd.DataFrame | None = None,
        macro_prices: pd.DataFrame | None = None,
        adr_features_frame: pd.DataFrame | None = None,
        rank_reversal_signals: pd.DataFrame | None = None,
        source: str = "unknown",
    ) -> None:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("historical frame must be a non-empty DataFrame")
        if isinstance(horizon, bool) or int(horizon) != horizon or int(horizon) < 1:
            raise ValueError("horizon must be a positive integer")
        start = _date(baseline_start, "baseline_start")
        end = _date(baseline_end, "baseline_end")
        if start > end:
            raise ValueError("baseline_start must be <= baseline_end")
        if start != pd.Timestamp("2010-01-01") or end != pd.Timestamp("2014-12-31"):
            raise ValueError("HistoricalInputs baseline must remain fixed at 2010-01-01..2014-12-31")
        owned = frame.copy(deep=True)
        if not isinstance(owned.index, pd.DatetimeIndex):
            owned.index = pd.to_datetime(owned.index)
        if owned.index.tz is not None:
            owned.index = owned.index.tz_convert("Asia/Tokyo").tz_localize(None)
        owned.index = owned.index.normalize()
        if owned.index.has_duplicates or owned.index.hasnans:
            raise ValueError("historical index must contain unique, valid trade dates")
        owned = owned.sort_index()
        if isinstance(pit_ir_history, Mapping):
            history: np.ndarray | Mapping[str, np.ndarray] | None = _FrozenMapping(
                {
                    _date(key, "pit_ir_history key").strftime("%Y-%m-%d"): _readonly_array(
                        value, "pit_ir_history"
                    )
                    for key, value in pit_ir_history.items()
                }
            )
        else:
            history = None if pit_ir_history is None else _readonly_array(pit_ir_history, "pit_ir_history")
        if isinstance(pit_history_trade_dates, Mapping):
            history_dates: np.ndarray | Mapping[str, np.ndarray] | None = _FrozenMapping(
                {
                    _date(key, "pit_history_trade_dates key").strftime("%Y-%m-%d"): _readonly_datetime_array(
                        value, "pit_history_trade_dates"
                    )
                    for key, value in pit_history_trade_dates.items()
                }
            )
        else:
            history_dates = (
                None
                if pit_history_trade_dates is None
                else _readonly_datetime_array(pit_history_trade_dates, "pit_history_trade_dates")
            )
        def _owned_frame(value: pd.DataFrame | None, name: str) -> pd.DataFrame | None:
            if value is None:
                return None
            if not isinstance(value, pd.DataFrame):
                raise ValueError(f"{name} must be a DataFrame when supplied")
            result = value.copy(deep=True)
            if not isinstance(result.index, pd.DatetimeIndex):
                result.index = pd.to_datetime(result.index)
            if result.index.tz is not None:
                result.index = result.index.tz_convert("Asia/Tokyo").tz_localize(None)
            result.index = result.index.normalize()
            if result.index.has_duplicates or result.index.hasnans:
                raise ValueError(f"{name} index must contain unique, valid timestamps")
            return result.sort_index()

        open_910 = _owned_frame(open_910_returns, "open_910_returns")
        macro = _owned_frame(macro_prices, "macro_prices")
        adr = _owned_frame(adr_features_frame, "adr_features")
        rank_reversal = _owned_frame(rank_reversal_signals, "rank_reversal_signals")
        object.__setattr__(self, "_frame", owned)
        object.__setattr__(self, "horizon", int(horizon))
        object.__setattr__(self, "baseline_start", start)
        object.__setattr__(self, "baseline_end", end)
        object.__setattr__(self, "pit_ir_history", history)
        object.__setattr__(self, "pit_history_trade_dates", history_dates)
        object.__setattr__(self, "label_available_at", _readonly_mapping(label_available_at, numeric=False))
        object.__setattr__(self, "observed_at", _readonly_mapping(observed_at, numeric=False))
        per_date: dict[str, Mapping[str, str]] = {}
        for key, values in (observed_at_by_date or {}).items():
            if not isinstance(values, Mapping):
                raise ValueError("observed_at_by_date values must be mappings")
            date_key = _date(key, "observed_at_by_date key").strftime("%Y-%m-%d")
            per_date[date_key] = _readonly_mapping(values, numeric=False)
        object.__setattr__(self, "observed_at_by_date", _FrozenMapping(per_date))
        # Keep owned frames private.  Public accessors below return defensive
        # copies so a caller cannot mutate a versioned run snapshot in place.
        object.__setattr__(self, "_open_910_returns", open_910)
        object.__setattr__(self, "_macro_prices", macro)
        object.__setattr__(self, "_adr_features", adr)
        object.__setattr__(self, "_rank_reversal_signals", rank_reversal)
        object.__setattr__(self, "source", str(source))

    @property
    def frame(self) -> pd.DataFrame:
        return self.to_frame()

    def to_frame(self) -> pd.DataFrame:
        """Return an isolated frame for a caller that may mutate it."""
        return self._frame.copy(deep=True)

    @staticmethod
    def _copy_optional_frame(frame: pd.DataFrame | None) -> pd.DataFrame | None:
        return None if frame is None else frame.copy(deep=True)

    @property
    def open_910_returns(self) -> pd.DataFrame | None:
        """Return an isolated copy of the owned 09:10 input frame."""
        return self._copy_optional_frame(self._open_910_returns)

    @property
    def macro_prices(self) -> pd.DataFrame | None:
        """Return an isolated copy of the owned macro price frame."""
        return self._copy_optional_frame(self._macro_prices)

    @property
    def adr_features(self) -> pd.DataFrame | None:
        """Return an isolated copy of the owned ADR feature frame."""
        return self._copy_optional_frame(self._adr_features)

    @property
    def rank_reversal_signals(self) -> pd.DataFrame | None:
        """Return the run-owned cross-sectional rank-reversal signal frame."""
        return self._copy_optional_frame(self._rank_reversal_signals)

    def pit_ir_history_for(self, as_of: Any) -> np.ndarray | None:
        """Return the PIT history applicable to one execution date.

        Backtests may provide a date-keyed mapping so every decision uses the
        history that was available before its own date.  Live and compatibility
        callers continue to use one array for the supplied run snapshot.
        """
        if self.pit_ir_history is None:
            return None
        if isinstance(self.pit_ir_history, Mapping):
            return self.pit_ir_history.get(_date(as_of, "as_of").strftime("%Y-%m-%d"))
        return self.pit_ir_history

    def pit_history_trade_dates_for(self, as_of: Any) -> np.ndarray | None:
        """Return the trade dates corresponding to the selected PIT IR rows."""
        if self.pit_history_trade_dates is None:
            return None
        if isinstance(self.pit_history_trade_dates, Mapping):
            return self.pit_history_trade_dates.get(_date(as_of, "as_of").strftime("%Y-%m-%d"))
        return self.pit_history_trade_dates

    def observed_at_for(self, as_of: Any) -> Mapping[str, str]:
        """Return feature observation timestamps for one historical date."""
        if not self.observed_at_by_date:
            return self.observed_at
        date_key = _date(as_of, "as_of").strftime("%Y-%m-%d")
        scoped = self.observed_at_by_date.get(date_key)
        if scoped is None:
            return self.observed_at
        # A date-scoped mapping may refine one feature without dropping the
        # common observation timestamps for all other features.  This merged
        # view is the one used by validation and is kept immutable like the
        # constructor-owned mappings.
        merged = dict(self.observed_at)
        merged.update(scoped)
        return _FrozenMapping(merged)

    def validate_observed_at(self, as_of: Any) -> None:
        """Reject auxiliary inputs whose recorded observation is in the future.

        ``KnownMarketInputs`` already applies this check to the primary
        snapshot.  Historical auxiliary frames (macro, ADR, rank reversal,
        PIT inputs, and the explicit 09:10 frame) use the same decision
        boundary, but historically their timestamps were only recorded and
        fingerprinted.  Validate the date-selected mapping before the model
        can consume those frames.
        """
        cutoff = _timestamp(as_of, "as_of")
        for name, value in self.observed_at_for(cutoff).items():
            observed = _timestamp(value, f"historical observed_at[{name}]")
            if observed > cutoff:
                raise ValueError(
                    f"historical observed_at[{name}] is later than as_of"
                )

    @staticmethod
    def _is_label_column(column: object) -> bool:
        """Return whether *column* contains a close/target label.

        Execution-time fields (US close returns, gaps, opens, and betas) stay
        visible on the current row.  Close-derived JP and TOPIX columns are
        masked until the corresponding session has finished.
        """
        name = str(column)
        return name.startswith(("jp_oc_", "jp_cc_", "topix_oc", "topix_cc", "target_", "y_jp_"))

    def calculation_frame(self, as_of: Any | None = None) -> pd.DataFrame:
        """Return a calculation view cut at ``as_of``.

        With no cutoff this retains the compatibility view used by offline
        research callers.  Production decisions pass the known timestamp;
        rows after that trade date are removed and close-derived labels on the
        current row are masked.  The masking uses pandas copy-on-write on
        pandas 3 and an owned copy on older versions.
        """
        # pandas 3 guarantees Copy-on-Write, so a view is isolated on mutation
        # without copying the full dataset on every backtest day. Older pandas
        # remains supported using an owned copy.
        if as_of is None:
            return self._frame.copy(deep=int(pd.__version__.split(".")[0]) < 3)
        cutoff = _date(as_of, "as_of")
        if cutoff not in self._frame.index:
            raise ValueError(f"as_of {cutoff.date()} is not present in historical frame")
        cutoff_ts = _timestamp(as_of, "as_of")
        frame = self._frame.loc[:cutoff].copy(deep=int(pd.__version__.split(".")[0]) < 3)
        # Explicit availability metadata overrides the standard JP-close
        # cutoff. Applying the default mask first would make an earlier
        # explicit availability impossible to express.
        for column in frame.columns:
            availability = self.label_available_at.get(str(column))
            if availability is None:
                for key, value in self.label_available_at.items():
                    if str(key).endswith("*") and str(column).startswith(str(key)[:-1]):
                        availability = value
                        break
            if availability is None and not self._is_label_column(column):
                continue
            available_at = self._label_available_timestamp(
                availability,
                cutoff,
                f"label_available_at[{column}]",
            )
            # The map describes when the current row's label becomes usable.
            # Historical rows remain available; only the current row is
            # hidden until its supplied timestamp, matching the default close
            # label rule above.
            if cutoff_ts < available_at:
                frame.loc[cutoff, column] = np.nan
        return frame

    @staticmethod
    def _label_available_timestamp(
        availability: Any | None,
        trade_date: pd.Timestamp,
        name: str,
    ) -> pd.Timestamp:
        """Resolve an absolute or session-relative label availability value.

        A full historical frame cannot store one absolute timestamp for every
        trade date.  ``trade_date+15:30`` (and the ``session_close`` alias)
        keeps the availability rule explicit while preserving the existing
        absolute timestamp form used by live snapshots and tests.
        """
        if availability is None:
            return trade_date + LABEL_AVAILABLE_CLOSE_TIME
        if isinstance(availability, str):
            value = availability.strip().lower()
            if value in {"session_close", "trade_date+15:30", "trade_date+15:30:00"}:
                return trade_date + LABEL_AVAILABLE_CLOSE_TIME
            if value.startswith("trade_date+"):
                offset = value.removeprefix("trade_date+")
                try:
                    hours, minutes, *seconds = (int(part) for part in offset.split(":"))
                    return trade_date + pd.Timedelta(
                        hours=hours,
                        minutes=minutes,
                        seconds=seconds[0] if seconds else 0,
                    )
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"{name} must contain a valid trade_date offset") from exc
        return _timestamp(availability, name)

    @property
    def fingerprint(self) -> str:
        return _digest(
            {
                "frame": _frame_digest(self._frame),
                "horizon": self.horizon,
                "baseline_start": self.baseline_start.isoformat(),
                "baseline_end": self.baseline_end.isoformat(),
                "pit_ir_history": self._pit_history_digest(),
                "pit_history_trade_dates": self._pit_history_dates_digest(),
                "label_available_at": dict(self.label_available_at),
                "observed_at": dict(self.observed_at),
                "observed_at_by_date": {
                    key: dict(value) for key, value in self.observed_at_by_date.items()
                },
                "open_910_returns": None if self._open_910_returns is None else _frame_digest(self._open_910_returns),
                "macro_prices": None if self._macro_prices is None else _frame_digest(self._macro_prices),
                "adr_features": None if self._adr_features is None else _frame_digest(self._adr_features),
                "rank_reversal_signals": None
                if self._rank_reversal_signals is None
                else _frame_digest(self._rank_reversal_signals),
                "source": self.source,
            }
        )

    def _pit_history_digest(self) -> Any:
        if self.pit_ir_history is None:
            return None
        if isinstance(self.pit_ir_history, Mapping):
            return {
                key: _array_digest(self.pit_ir_history[key])
                for key in sorted(self.pit_ir_history)
            }
        return _array_digest(self.pit_ir_history)

    def _pit_history_dates_digest(self) -> Any:
        if self.pit_history_trade_dates is None:
            return None
        if isinstance(self.pit_history_trade_dates, Mapping):
            return {
                key: _array_digest(self.pit_history_trade_dates[key])
                for key in sorted(self.pit_history_trade_dates)
            }
        return _array_digest(self.pit_history_trade_dates)


@dataclass(frozen=True)
class EvaluationInputs:
    """Post-decision values used only for realized evaluation."""

    trade_date: pd.Timestamp
    target_returns: np.ndarray | None = None
    evaluation_prices: Mapping[str, float] = field(default_factory=dict)
    source: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "trade_date", _date(self.trade_date, "trade_date"))
        target = None if self.target_returns is None else _readonly_array(self.target_returns, "target_returns")
        object.__setattr__(self, "target_returns", target)
        object.__setattr__(self, "evaluation_prices", _readonly_mapping(self.evaluation_prices, numeric=True))

    @property
    def fingerprint(self) -> str:
        return _digest(
            {
                "trade_date": self.trade_date.isoformat(),
                "target_returns": None if self.target_returns is None else _array_digest(self.target_returns),
                "evaluation_prices": dict(self.evaluation_prices),
                "source": self.source,
            }
        )


@dataclass(frozen=True)
class DecisionInputs:
    """Single input object passed from the runner to the decision engine."""

    known: KnownMarketInputs
    historical: HistoricalInputs
    gap_input_dir: Path | None = None
    use_file_cache: bool = True
    schema_version: str = "decision-inputs-v1"

    def __post_init__(self) -> None:
        if self.gap_input_dir is not None:
            object.__setattr__(self, "gap_input_dir", Path(self.gap_input_dir))
        self.historical.validate_observed_at(self.known.as_of)

    @property
    def trade_date(self) -> pd.Timestamp:
        return self.known.trade_date

    @property
    def version(self) -> InputVersion:
        return InputVersion(
            schema_version=self.schema_version,
            source=self.known.source,
            known_fingerprint=self.known.fingerprint,
            historical_fingerprint=self.historical.fingerprint,
        )

    @classmethod
    def from_parts(
        cls,
        frame: pd.DataFrame,
        *,
        trade_date: Any,
        as_of: Any,
        ticker_order: tuple[str, ...],
        us_returns: Any,
        jp_gap_returns: Any,
        jp_betas: Any,
        topix_night_return: float,
        current_prices: Mapping[str, float],
        prev_closes: Mapping[str, float],
        macro_features: Mapping[str, float] | None = None,
        adr_features: Mapping[str, float] | None = None,
        sig_date: Any | None = None,
        observed_at: Mapping[str, str] | None = None,
        gap_input_dir: Path | None = None,
        use_file_cache: bool = True,
        source: str = "unknown",
        price_sources: Mapping[str, str] | None = None,
        historical: HistoricalInputs | None = None,
        label_available_at: Mapping[str, str] | None = None,
        historical_observed_at: Mapping[str, str] | None = None,
        historical_observed_at_by_date: Mapping[str, Mapping[str, str]] | None = None,
        open_910_returns: pd.DataFrame | None = None,
        macro_prices: pd.DataFrame | None = None,
        adr_features_frame: pd.DataFrame | None = None,
        rank_reversal_signals: pd.DataFrame | None = None,
    ) -> DecisionInputs:
        known = KnownMarketInputs(
            trade_date=_date(trade_date, "trade_date"),
            as_of=_timestamp(as_of, "as_of"),
            ticker_order=ticker_order,
            us_returns=us_returns,
            jp_gap_returns=jp_gap_returns,
            jp_betas=jp_betas,
            topix_night_return=topix_night_return,
            current_prices=current_prices,
            prev_closes=prev_closes,
            macro_features=macro_features or {},
            adr_features=adr_features or {},
            sig_date=sig_date,
            observed_at=observed_at or {},
            source=source,
            price_sources=price_sources or {},
        )
        return cls(
            known=known,
            historical=historical or HistoricalInputs(
                frame,
                source=source,
                label_available_at=label_available_at,
                observed_at=historical_observed_at,
                observed_at_by_date=historical_observed_at_by_date,
                open_910_returns=open_910_returns,
                macro_prices=macro_prices,
                adr_features_frame=adr_features_frame,
                rank_reversal_signals=rank_reversal_signals,
            ),
            gap_input_dir=gap_input_dir,
            use_file_cache=use_file_cache,
        )


__all__ = [
    "DecisionInputs",
    "EvaluationInputs",
    "HistoricalInputs",
    "InputVersion",
    "KnownMarketInputs",
]
