from pathlib import Path
import pandas as pd

for path in (
    Path("var/results/beta_shift_comparison/df_exec_old.pkl"),
    Path("var/live/pipeline_data/cache/preprocessed/20260728/preprocessed_data.pkl"),
):
    value = pd.read_pickle(path)
    if isinstance(value, dict):
        print(path, "dict", sorted(value), {key: getattr(item, "shape", None) for key, item in value.items()})
    else:
        print(path, len(value), value.index.min(), value.index.max(), len(value.columns))
