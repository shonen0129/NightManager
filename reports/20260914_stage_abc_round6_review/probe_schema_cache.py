"""Verify column-identity cache behavior with the real numerical builder."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from leadlag.execution.config import load_config_from_yaml
from leadlag.models.blpx import ProductionBLPXModel
from leadlag.data.tickers import US_TICKERS, JP_TICKERS

cfg = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True)
frame = pd.read_csv(ROOT / "tests/regression/baselines/df_exec_20260814.csv.gz", index_col=0, parse_dates=True)
first, second = [f"us_cc_{ticker}" for ticker in US_TICKERS[:2]]
renamed = frame.rename(columns={first: second, second: first})
targets = frame[[f"jp_oc_{ticker}" for ticker in JP_TICKERS]].to_numpy()
shared = ProductionBLPXModel(cfg.v2.blpx)
old = shared._prepare_common_inputs(frame, y_jp_target=targets)
cached = shared._prepare_common_inputs(renamed, y_jp_target=targets)
fresh = ProductionBLPXModel(cfg.v2.blpx)._prepare_common_inputs(renamed, y_jp_target=targets)
result = {
    "rows": len(frame), "columns": len(frame.columns),
    "renamed_columns": [first, second],
    "row_hash_unchanged": bool(np.array_equal(pd.util.hash_pandas_object(frame, index=True), pd.util.hash_pandas_object(renamed, index=True))),
    "cache_reused": old is cached,
    "max_abs_all_returns_difference": float(np.max(np.abs(cached["all_returns_raw"] - fresh["all_returns_raw"]))),
    "has_difference": not np.allclose(cached["all_returns_raw"], fresh["all_returns_raw"], equal_nan=True),
}
(OUT / "schema_cache.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
