"""Session state model for execution rows.

Tracks where in the daily cycle a given df_exec row sits.  This is used by
preprocessors and the decision pipeline to enforce point-in-time rules.
"""

from __future__ import annotations

from enum import Enum


class SessionState(Enum):
    """Point-in-time state of a single trading session."""

    PRE_OPEN = "pre_open"          # before JP market open (no 09:10 prices)
    POST_OPEN = "post_open"        # after 09:10 prices available
    POST_CLOSE = "post_close"      # after JP market close
    SETTLED = "settled"            # all prices confirmed and reconciled
