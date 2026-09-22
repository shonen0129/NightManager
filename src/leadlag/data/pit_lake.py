"""Point-in-Time (PIT) Data Lake & Snapshot Engine.

Provides date-aligned market snapshots and owned historical input contracts.
Historical training windows are constrained by the model's as-of calculation
view and audits; ``history_frame`` remains a full-history compatibility copy
for evaluation and offline inspection.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from leadlag.data.intraday_inputs import resolve_execution_prices
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from leadlag.domain.inputs import DecisionInputs, HistoricalInputs, KnownMarketInputs

logger = logging.getLogger(__name__)


def validate_production_decision_inputs(inputs: DecisionInputs) -> None:
    """Reject incomplete 09:10 prices before the production model runs.

    Generic input fixtures may represent a smaller universe, while the
    production contract is fixed at ``JP_TICKERS``.  Keeping this check at the
    data/production boundary prevents a missing quote from becoming a zero gap.
    """
    known = inputs.known
    if tuple(known.ticker_order) != tuple(JP_TICKERS):
        return
    missing = [ticker for ticker in JP_TICKERS if ticker not in known.current_prices]
    invalid = [
        ticker
        for ticker in JP_TICKERS
        if ticker in known.current_prices
        and (
            not np.isfinite(float(known.current_prices[ticker]))
            or float(known.current_prices[ticker]) <= 0.0
        )
    ]
    if missing or invalid:
        details: list[str] = []
        if missing:
            details.append(f"missing current_prices: {missing}")
        if invalid:
            details.append(f"invalid current_prices: {invalid}")
        raise ValueError(
            "Production DecisionInputs require finite positive current_prices "
            "for every JP ticker (" + "; ".join(details) + ")"
        )


class PITLookaheadError(Exception):
    """Raised when an attempt is made to access data from the future."""
    pass


@dataclass(frozen=True)
class MarketSnapshot:
    """Immutable Point-in-Time Market Snapshot at a specific timestamp.

    Contains only data that was finalized and known at or before ``as_of``.
    """
    as_of: pd.Timestamp
    trade_date: str  # YYYY-MM-DD
    us_returns: np.ndarray  # Shape (15,) US Close-to-Close returns (or frac-diff)
    jp_gap_returns: np.ndarray  # Shape (17,) JP 9:10 gap returns: (open / prev_close - 1)
    jp_betas: np.ndarray  # Shape (17,) Rolling TOPIX betas
    topix_night_return: float  # TOPIX overnight return
    current_prices: Mapping[str, float]  # JP tickers -> 9:10 execution prices
    prev_closes: Mapping[str, float]  # JP tickers -> previous day close prices
    price_sources: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Own and freeze arrays/mappings at the PIT boundary.

        ``frozen=True`` protects attribute replacement, but NumPy arrays and
        dictionaries remain mutable unless they are explicitly copied here.
        A snapshot is therefore safe to pass between the data, model, and
        execution layers without one layer changing another layer's inputs.
        """
        arrays = ("us_returns", "jp_gap_returns", "jp_betas")
        for name in arrays:
            value = np.array(getattr(self, name), dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "current_prices", MappingProxyType({str(k): float(v) for k, v in self.current_prices.items()}))
        object.__setattr__(self, "prev_closes", MappingProxyType({str(k): float(v) for k, v in self.prev_closes.items()}))
        object.__setattr__(self, "topix_night_return", float(self.topix_night_return))
        object.__setattr__(self, "price_sources", MappingProxyType(dict(self.price_sources)))

    def to_known_inputs(
        self,
        *,
        sig_date: pd.Timestamp | str | None = None,
        observed_at: Mapping[str, str] | None = None,
        source: str = "pit_lake",
    ) -> KnownMarketInputs:
        """Convert this snapshot into the versioned known-input contract."""
        return KnownMarketInputs(
            trade_date=pd.Timestamp(self.trade_date),
            as_of=self.as_of,
            ticker_order=tuple(JP_TICKERS),
            us_returns=self.us_returns,
            jp_gap_returns=self.jp_gap_returns,
            jp_betas=self.jp_betas,
            topix_night_return=self.topix_night_return,
            current_prices=self.current_prices,
            prev_closes=self.prev_closes,
            sig_date=sig_date,
            observed_at=observed_at or {},
            source=source,
            price_sources=self.price_sources,
        )

    def validate(self, max_abs_return: float = 0.20) -> tuple[bool, list[str]]:
        """Run sanity checks on snapshot data (prices, returns, NaN/Inf).

        Returns:
            (is_valid, list_of_error_messages)
        """
        errors: list[str] = []

        # 1. Finite and bounded US returns
        if not np.all(np.isfinite(self.us_returns)):
            errors.append("Non-finite values detected in us_returns")
        elif np.any(np.abs(self.us_returns) > max_abs_return):
            errors.append(f"US returns exceed max_abs_return threshold ({max_abs_return:.1%})")

        # 2. Finite and bounded JP gap returns
        if not np.all(np.isfinite(self.jp_gap_returns)):
            errors.append("Non-finite values detected in jp_gap_returns")
        elif np.any(np.abs(self.jp_gap_returns) > max_abs_return):
            errors.append(f"JP gap returns exceed max_abs_return threshold ({max_abs_return:.1%})")

        # 3. Finite Betas
        if not np.all(np.isfinite(self.jp_betas)):
            errors.append("Non-finite values detected in jp_betas")

        # 4. Valid current prices (positive and finite)
        for tk in JP_TICKERS:
            price = self.current_prices.get(tk, 0.0)
            if not np.isfinite(price) or price <= 0.0:
                errors.append(f"Invalid or missing execution price for {tk}: {price}")

        return (len(errors) == 0, errors)

    def is_valid(self, max_abs_return: float = 0.20) -> bool:
        """Return True if the snapshot passes all sanity checks."""
        valid, _ = self.validate(max_abs_return=max_abs_return)
        return valid


class PITDataLake:
    """Point-in-Time Data Lake for Lead-Lag Quantitative Engine.

    Guarantees that queries for date T cannot observe any data generated after T 09:10:00 JST.
    """

    def __init__(self, df_exec: pd.DataFrame) -> None:
        """Initialize with an execution dataframe.

        Args:
            df_exec: DataFrame indexed by trade_date (Timestamp or string).
        """
        self._df = df_exec.copy()
        if not isinstance(self._df.index, pd.DatetimeIndex):
            self._df.index = pd.to_datetime(self._df.index)
        if self._df.index.tz is not None:
            self._df.index = self._df.index.tz_convert("Asia/Tokyo").tz_localize(None)
        self._df.index = self._df.index.normalize()
        if self._df.empty:
            raise ValueError("df_exec must contain at least one trade date")
        if self._df.index.hasnans:
            raise ValueError("df_exec index must contain valid trade dates")
        if self._df.index.has_duplicates:
            raise ValueError("df_exec index must contain unique trade dates")
        self._df = self._df.sort_index()

        self._trade_dates = self._df.index
        logger.info(
            "Initialized PITDataLake with %d dates from %s to %s",
            len(self._trade_dates),
            self._trade_dates[0].strftime("%Y-%m-%d"),
            self._trade_dates[-1].strftime("%Y-%m-%d"),
        )

    @property
    def start_date(self) -> pd.Timestamp:
        return self._trade_dates[0]

    @property
    def end_date(self) -> pd.Timestamp:
        return self._trade_dates[-1]

    @property
    def df_exec(self) -> pd.DataFrame:
        """Return an isolated compatibility copy of the execution DataFrame.

        New decision code uses ``build_decision_inputs``.  Returning a copy
        here prevents legacy callers from mutating the lake's owned history.
        """
        return self._df.copy(deep=True)

    def history_frame(self) -> pd.DataFrame:
        """Return an isolated copy for callers outside the decision boundary."""
        return self._df.copy(deep=True)

    def available_dates_up_to(self, as_of: pd.Timestamp | str) -> pd.DatetimeIndex:
        """Return trade dates that occurred on or before as_of."""
        as_of_ts = pd.to_datetime(as_of)
        if as_of_ts.tzinfo is not None:
            as_of_ts = as_of_ts.tz_convert("Asia/Tokyo").tz_localize(None)
        return self._trade_dates[self._trade_dates <= as_of_ts]

    def get_snapshot(self, as_of: pd.Timestamp | str) -> MarketSnapshot:
        """Extract known-market fields for the JST trade date of as_of.

        The cache's date alignment is checked here. This is not evidence that
        every source observation was available at the supplied intraday time;
        that requires observation timestamps at the input adapter.

        Args:
            as_of: Point-in-time timestamp (or date string YYYY-MM-DD).

        Returns:
            MarketSnapshot containing the selected known-market columns.
        """
        as_of_ts = pd.to_datetime(as_of)
        if as_of_ts.tzinfo is not None:
            as_of_ts = as_of_ts.tz_convert("Asia/Tokyo").tz_localize(None)
        target_ts = as_of_ts.normalize()

        if target_ts not in self._df.index:
            if target_ts > self._trade_dates[-1]:
                raise PITLookaheadError(
                    f"as_of {as_of} is beyond the latest available trade date "
                    f"{self._trade_dates[-1].date()}. Future data is not allowed."
                )
            if target_ts < self._trade_dates[0]:
                raise PITLookaheadError(
                    f"as_of {as_of} is before the earliest available trade date "
                    f"{self._trade_dates[0].date()}."
                )
            # as_of is within the lake range but not a trade date in the index.
            raise PITLookaheadError(
                f"as_of {as_of} is not a trade date in the lake. "
                f"Use available_dates_up_to({as_of}) to find the nearest prior date."
            )

        row = self._df.loc[target_ts]
        trade_date_str = target_ts.strftime("%Y-%m-%d")

        # 1. US Returns (US close on previous US day, finalized overnight)
        # Missing or non-finite US returns are kept as NaN so that
        # MarketSnapshot.validate() rejects the snapshot instead of silently
        # feeding 0.0 into the BLPX signal.
        us_cols = [f"us_cc_{tk}" for tk in US_TICKERS]
        us_returns = np.array(
            [float(row[col]) if col in row and np.isfinite(float(row[col])) else np.nan for col in us_cols],
            dtype=float,
        )

        # 2. JP 9:10 Execution Prices & Previous Closes
        current_prices: dict[str, float] = {}
        prev_closes: dict[str, float] = {}
        jp_gap_returns = np.zeros(len(JP_TICKERS), dtype=float)
        jp_betas = np.zeros(len(JP_TICKERS), dtype=float)

        for j, tk in enumerate(JP_TICKERS):
            open_col = f"jp_open_trade_{tk}"
            close_col = f"jp_close_sig_{tk}"
            beta_col = f"jp_beta_{tk}"
            gap_col = f"jp_gap_{tk}"

            p_open = float(row.get(open_col, np.nan))
            p_prev_close = float(row.get(close_col, np.nan))
            beta_val = float(row.get(beta_col, 0.0))

            if np.isfinite(p_open) and p_open > 0.0:
                current_prices[tk] = p_open
            if np.isfinite(p_prev_close) and p_prev_close > 0.0:
                prev_closes[tk] = p_prev_close

            # Beta
            jp_betas[j] = beta_val if np.isfinite(beta_val) else 0.0

            # Gap return (use precalculated or compute on the fly)
            if gap_col in row and np.isfinite(float(row[gap_col])):
                jp_gap_returns[j] = float(row[gap_col])
            elif tk in current_prices and tk in prev_closes and prev_closes[tk] > 0.0:
                jp_gap_returns[j] = (current_prices[tk] / prev_closes[tk]) - 1.0
            else:
                jp_gap_returns[j] = 0.0

        # 3. TOPIX overnight return
        topix_night_val = float(row.get("topix_night_return", 0.0))
        topix_night_return = topix_night_val if np.isfinite(topix_night_val) else 0.0

        return MarketSnapshot(
            as_of=as_of_ts,
            trade_date=trade_date_str,
            us_returns=us_returns,
            jp_gap_returns=jp_gap_returns,
            jp_betas=jp_betas,
            topix_night_return=topix_night_return,
            current_prices=current_prices,
            prev_closes=prev_closes,
        )

    def get_execution_snapshot(
        self,
        as_of: pd.Timestamp | str,
        open_910_returns: pd.DataFrame,
    ) -> MarketSnapshot:
        """Use the run's 09:10 observation, with identified open-price fallback."""
        snapshot = self.get_snapshot(as_of)
        date = pd.Timestamp(snapshot.trade_date)
        prices, sources = resolve_execution_prices(
            self._df, open_910_returns, JP_TICKERS, required_index=[date]
        )
        previous = np.array([snapshot.prev_closes.get(ticker, np.nan) for ticker in JP_TICKERS])
        if not (np.isfinite(previous) & (previous > 0)).all():
            raise ValueError("execution snapshot requires finite positive previous closes")
        current = prices.loc[date]
        return replace(
            snapshot, current_prices=current.to_dict(),
            jp_gap_returns=current.to_numpy(dtype=float) / previous - 1.0,
            price_sources=sources.loc[date].to_dict(),
        )

    def build_decision_inputs(
        self,
        as_of: pd.Timestamp | str,
        *,
        snapshot: MarketSnapshot | None = None,
        current_prices: Mapping[str, float] | None = None,
        jp_gap_returns: np.ndarray | None = None,
        gap_input_dir: Path | None = None,
        use_file_cache: bool = True,
        source: str = "pit_lake",
        historical: HistoricalInputs | None = None,
        open_910_returns: pd.DataFrame | None = None,
        macro_prices: pd.DataFrame | None = None,
        adr_features_frame: pd.DataFrame | None = None,
        rank_reversal_signals: pd.DataFrame | None = None,
        observed_at: Mapping[str, str] | None = None,
        historical_observed_at: Mapping[str, str] | None = None,
        historical_observed_at_by_date: Mapping[str, Mapping[str, str]] | None = None,
    ) -> DecisionInputs:
        """Build one versioned contract from a PIT snapshot and owned history.

        Explicit prices/gaps are accepted only at this adapter boundary.  The
        model receives the resulting ``DecisionInputs`` object and no longer
        has to reconcile parallel legacy arguments.
        """
        requested = pd.Timestamp(as_of)
        if requested.tzinfo is not None:
            requested = requested.tz_convert("Asia/Tokyo").tz_localize(None)
        requested_date = requested.normalize()
        date_only_request = requested == requested_date
        # A date-only compatibility call still represents the standard 09:10
        # decision. Building its snapshot at midnight would mark 09:10
        # prices/gaps as known before they were observed.
        requested_cutoff = requested_date + pd.Timedelta(hours=9, minutes=10)
        if not date_only_request and requested < requested_cutoff:
            raise ValueError("decision snapshot must be at or after 09:10 JST")
        snap = snapshot or self.get_snapshot(requested_cutoff if date_only_request else requested)
        trade_date = pd.Timestamp(snap.trade_date)
        snapshot_as_of = pd.Timestamp(snap.as_of)
        if snapshot_as_of.tzinfo is not None:
            snapshot_as_of = snapshot_as_of.tz_convert("Asia/Tokyo").tz_localize(None)
        snapshot_date_ok = snapshot_as_of.normalize() == trade_date
        if requested_date != trade_date or not snapshot_date_ok:
            raise ValueError("snapshot date must match requested decision date")
        # A date-only request is a compatibility form whose latest permitted
        # decision cutoff is the standard 09:10 JST boundary.  Once a caller
        # supplies an intraday timestamp, the snapshot must be from that exact
        # instant; accepting a same-date post-close snapshot here would expose
        # realized labels through HistoricalInputs.calculation_frame().
        if date_only_request:
            timestamp_ok = snapshot_as_of == requested_cutoff
        else:
            timestamp_ok = snapshot_as_of == requested
        if not timestamp_ok:
            raise ValueError("snapshot timestamp must match the requested decision cutoff")
        row = self._df.loc[trade_date]
        raw_sig_date = row.get("sig_date") if hasattr(row, "get") else None
        if raw_sig_date is None and hasattr(row, "get"):
            raw_sig_date = row.get("signal_date")
        if raw_sig_date is not None:
            try:
                if pd.isna(raw_sig_date):
                    raw_sig_date = None
            except (TypeError, ValueError):
                pass
        prices = snap.current_prices if current_prices is None else current_prices
        if jp_gap_returns is not None:
            gaps = jp_gap_returns
        elif current_prices is None:
            gaps = snap.jp_gap_returns
        else:
            # Legacy callers may provide live 09:10 prices without a rebuilt
            # snapshot. Recompute the gap at this adapter boundary so the
            # contract cannot combine new prices with stale row gaps.
            gaps = np.zeros(len(JP_TICKERS), dtype=float)
            for index, ticker in enumerate(JP_TICKERS):
                price = prices.get(ticker)
                previous = snap.prev_closes.get(ticker)
                if price is not None and previous is not None and previous > 0.0:
                    value = float(price) / float(previous) - 1.0
                    gaps[index] = value if np.isfinite(value) else 0.0
        return DecisionInputs.from_parts(
            self._df,
            trade_date=trade_date,
            as_of=snap.as_of,
            ticker_order=tuple(JP_TICKERS),
            us_returns=snap.us_returns,
            jp_gap_returns=gaps,
            jp_betas=snap.jp_betas,
            topix_night_return=snap.topix_night_return,
            current_prices=prices,
            prev_closes=snap.prev_closes,
            sig_date=raw_sig_date,
            observed_at=observed_at,
            historical_observed_at=historical_observed_at,
            historical_observed_at_by_date=historical_observed_at_by_date,
            gap_input_dir=gap_input_dir,
            use_file_cache=use_file_cache,
            source=source,
            price_sources=snap.price_sources if current_prices is None else {
                ticker: "explicit_current_price" for ticker in current_prices
            },
            historical=historical,
            open_910_returns=open_910_returns,
            macro_prices=macro_prices,
            adr_features_frame=adr_features_frame,
            rank_reversal_signals=rank_reversal_signals,
        )

    def validate_no_lookahead(self, test_date: pd.Timestamp | str) -> bool:
        """Audit method: verify that fetching a snapshot does not leak same-day close returns."""
        snap = self.get_snapshot(test_date)
        # Only pre-approved snapshot attributes may be present. Any same-day
        # close return/price attributes (e.g. jp_oc_returns, close_prices)
        # would indicate a look-ahead leak.
        allowed = {
            "as_of",
            "trade_date",
            "us_returns",
            "jp_gap_returns",
            "jp_betas",
            "topix_night_return",
            "current_prices",
            "prev_closes",
            "price_sources",
        }
        extra = set(vars(snap).keys()) - allowed
        if extra:
            raise PITLookaheadError(
                f"MarketSnapshot contains forbidden attributes: {extra}"
            )
        return True
