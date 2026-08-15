#!/usr/bin/env python
"""Print the date range of the local df_exec cache."""

from leadlag.data.cache import load_df_exec_from_local_cache


def main() -> None:
    df = load_df_exec_from_local_cache()
    print("start", df.index[0], "end", df.index[-1], "rows", len(df))


if __name__ == "__main__":
    main()
