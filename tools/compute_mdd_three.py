import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(r"c:\Users\cydr\日米ラグ")
csv_path = ROOT / "results" / "fine_grid_local" / "compare_three_series.csv"
out_csv = ROOT / "results" / "fine_grid_local" / "mdd_summary.csv"

df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date")
res = {}
for col in df.columns:
    series = df[col].astype(float)
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max
    mdd = float(drawdown.min())
    res[col] = mdd

out = pd.DataFrame.from_dict(res, orient="index", columns=["MDD"]).sort_values("MDD")
out.to_csv(out_csv)
print(out.to_string())
