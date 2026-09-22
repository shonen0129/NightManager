"""Summarize input coverage without exposing account information."""
import json
import logging
from collections import Counter

import numpy as np

from leadlag.data.pit_lake import PITDataLake
from leadlag.data.tickers import JP_TICKERS
from research.scripts.experiments.structural_artifact_acceptance import (
    REPORT, WORK, load, resources, write_json,
)

logging.basicConfig(level=logging.WARNING)
frame, history, config = resources()
chunks = [load(WORK / "collected" / f"chunk_{i}.pkl") for i in range(4)]
decisions = {k: v for chunk in chunks for k, v in chunk["decisions"].items()}
dates = frame.index[frame.index >= "2015-01-05"]
output = {"missing_dates": [], "last_rows": [], "fallbacks": {}}
for date in dates.difference(list(decisions)):
    reason = None
    try:
        PITDataLake(frame).get_execution_snapshot(str(date.date()) + " 09:10", history.open_910_returns)
    except Exception as exc:
        reason = str(exc)
    output["missing_dates"].append({"date": str(date.date()), "reason": reason,
                                   "is_provisional": bool(frame.loc[date].get("is_provisional", False))})
for date in dates[-3:]:
    opens = frame.loc[date, [f"jp_open_trade_{t}" for t in JP_TICKERS]].to_numpy(dtype=float)
    output["last_rows"].append({"date": str(date.date()), "is_provisional": bool(frame.loc[date].get("is_provisional", False)),
                               "invalid_opens": int((~np.isfinite(opens) | (opens <= 0)).sum())})
output["fallbacks"] = dict(Counter(str(v.fallback) for v in decisions.values()))
output["audit_failure_details"] = [{"date": str(d.date()), "numerical": v.numerical,
                                     "leakage": v.leakage, "alerts": v.alerts}
                                    for d, v in decisions.items() if v.fallback.get("audit_failure")]
write_json(REPORT / "evaluation_coverage_result.json", output)
print(json.dumps(output, indent=2))
