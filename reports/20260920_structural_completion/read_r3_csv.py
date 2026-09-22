from pathlib import Path

import pandas as pd


def main() -> None:
    p = Path("var/results/20260920_structural_completion/r3_regenerated_gap/20260920_063702/gap_adjusted_distribution_long.csv")
    df = pd.read_csv(p)
    print(df[(df.trade_date == "2026-08-14") & (df.ticker == "1617.T")].to_string(index=False))


if __name__ == "__main__":
    main()
