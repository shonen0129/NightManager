"""The migrated research cache writer publishes a frame exactly once."""

import pandas as pd

from research.scripts.experiments import save_new_df_exec_to_cache as writer


def test_save_new_df_exec_writes_once(monkeypatch):
    frame = pd.DataFrame({"value": [1]}, index=pd.to_datetime(["2025-01-20"]))
    calls = []
    monkeypatch.setattr(writer.pd, "read_pickle", lambda path: frame)
    monkeypatch.setattr(writer, "save_df_exec_to_local_cache", calls.append)
    assert writer.main() == 0
    assert len(calls) == 1
    assert calls[0] is frame
