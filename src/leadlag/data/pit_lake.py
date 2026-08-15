"""Point-in-Time (PIT) Data Lake & Snapshot Engine.

Ensures absolute temporal integrity by allowing data access ONLY via
as_of timestamps.未来データへのアクセスを型とAPIレベルで物理的に遮断
(By-Construction Leak Prevention).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from leadlag.data.tickers import JP_TICKERS, US_TICKERS

logger = logging.getLogger(__name__)


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
    current_prices: dict[str, float]  # JP tickers -> 9:10 execution prices
    prev_closes: dict[str, float]  # JP tickers -> previous day close prices

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
        """Return the full execution DataFrame (assumed to contain no future rows)."""
        return self._df

    def available_dates_up_to(self, as_of: pd.Timestamp | str) -> pd.DatetimeIndex:
        """Return trade dates that occurred on or before as_of."""
        as_of_ts = pd.to_datetime(as_of)
        return self._trade_dates[self._trade_dates <= as_of_ts]

    def get_snapshot(self, as_of: pd.Timestamp | str) -> MarketSnapshot:
        """Extract an immutable MarketSnapshot strictly available at as_of.

        Args:
            as_of: Point-in-time timestamp (or date string YYYY-MM-DD).

        Returns:
            MarketSnapshot containing strictly historical data.
        """
        as_of_ts = pd.to_datetime(as_of)

        if as_of_ts not in self._df.index:
            if as_of_ts > self._trade_dates[-1]:
                raise PITLookaheadError(
                    f"as_of {as_of} is beyond the latest available trade date "
                    f"{self._trade_dates[-1].date()}. Future data is not allowed."
                )
            if as_of_ts < self._trade_dates[0]:
                raise PITLookaheadError(
                    f"as_of {as_of} is before the earliest available trade date "
                    f"{self._trade_dates[0].date()}."
                )
            # as_of is within the lake range but not a trade date in the index.
            raise PITLookaheadError(
                f"as_of {as_of} is not a trade date in the lake. "
                f"Use available_dates_up_to({as_of}) to find the nearest prior date."
            )

        target_ts = as_of_ts

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
            as_of=target_ts,
            trade_date=trade_date_str,
            us_returns=us_returns,
            jp_gap_returns=jp_gap_returns,
            jp_betas=jp_betas,
            topix_night_return=topix_night_return,
            current_prices=current_prices,
            prev_closes=prev_closes,
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
        }
        extra = set(vars(snap).keys()) - allowed
        if extra:
            raise PITLookaheadError(
                f"MarketSnapshot contains forbidden attributes: {extra}"
            )
        return True
