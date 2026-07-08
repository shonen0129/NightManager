"""src/features/feature_selection_fdr.py — Sprint 3-A FDR-based Feature Selection.

Implements Benjamini-Hochberg FDR correction applied to hinge feature selection.
All selection is performed ONLY within train windows — validation/test data
never influences which features are selected.

Algorithm per walk-forward window:
1. Compute daily cross-sectional Rank IC of each hinge feature vs. target.
2. Aggregate: mean Rank IC, ICIR, p-value (t-test on IC series).
3. Apply Benjamini-Hochberg FDR correction.
4. Filter by: q <= fdr_q AND |mean_rank_ic| >= min_abs_rank_ic.
5. Sign consistency filter: split train into first/second half,
   require sign(mean_ic_first_half) == sign(mean_ic_second_half) with
   sign_consistency_ratio >= min_sign_consistency.
6. Keep top max_features_after_fdr features by |mean_rank_ic|.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.stats as stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rank IC computation
# ---------------------------------------------------------------------------


def compute_daily_rank_ic(
    features: pd.DataFrame,
    target: pd.Series,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute daily cross-sectional Rank IC (Spearman) for each feature vs. target.

    Parameters
    ----------
    features:
        MultiIndex DataFrame (date, ticker) or wide panel (date x ticker).
        We expect a wide panel: dates as index, columns as asset identifiers.
        Each row = one cross-section.
    target:
        Same shape panel (dates x tickers).
    dates:
        Date subset to compute IC for (train window only).

    Returns
    -------
    pd.DataFrame
        IC timeseries, shape (n_dates, n_features). Columns are feature names.
    """
    raise NotImplementedError(
        "compute_daily_rank_ic requires panel-shaped data; use "
        "compute_rank_ic_long_format instead."
    )


def compute_rank_ic_long_format(
    features_long: pd.DataFrame,
    target_long: pd.Series,
    feature_cols: list[str],
    date_col: str = "date",
) -> pd.DataFrame:
    """Compute daily cross-sectional Rank IC (Spearman) in long format.

    Parameters
    ----------
    features_long:
        Long-format DataFrame with at least [date_col] + feature_cols.
    target_long:
        Series with same index as features_long (aligned daily cross-sections).
    feature_cols:
        List of feature column names to evaluate.
    date_col:
        Name of date column in features_long.

    Returns
    -------
    pd.DataFrame
        IC timeseries (date → feature → rank_ic).
        Shape: (n_unique_dates, n_features).
    """
    if features_long.empty or len(feature_cols) == 0:
        return pd.DataFrame(columns=feature_cols).set_index("date")

    # 1. Align features and target
    df = features_long[[date_col] + feature_cols].copy()
    df["_target"] = target_long.values

    # 2. Count valid pairs per date and column
    not_na = df[feature_cols].notna().multiply(df["_target"].notna(), axis=0)
    valid_counts = not_na.groupby(df[date_col]).sum()

    # 3. Rank features and target within each date
    f_ranked = df.groupby(date_col)[feature_cols].rank()
    y_ranked = df.groupby(date_col)["_target"].rank()

    # 4. Demean within each date to compute Pearson on ranks
    f_mean = f_ranked.groupby(df[date_col]).transform("mean")
    y_mean = y_ranked.groupby(df[date_col]).transform("mean")

    f_demeaned = f_ranked - f_mean
    y_demeaned = y_ranked - y_mean

    # Mask NaNs where input features or target were NaN
    f_demeaned = f_demeaned.where(df[feature_cols].notna())
    y_demeaned = y_demeaned.where(df["_target"].notna())

    # 5. Compute Pearson correlation per date: cov(X, Y) / sqrt(var(X) * var(Y))
    cov = f_demeaned.multiply(y_demeaned, axis=0).groupby(df[date_col]).sum()
    var_f = (f_demeaned ** 2).groupby(df[date_col]).sum()
    var_y = (y_demeaned ** 2).groupby(df[date_col]).sum()

    denom = np.sqrt(var_f.multiply(var_y, axis=0))
    denom = denom.replace(0, np.nan)
    corrs = cov / denom

    # Enforce minimum count >= 3
    corrs[valid_counts < 3] = np.nan

    return corrs


# ---------------------------------------------------------------------------
# BH FDR correction
# ---------------------------------------------------------------------------


def benjamini_hochberg(p_values: np.ndarray, q: float) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values:
        Array of p-values (one per feature).
    q:
        FDR control level.

    Returns
    -------
    np.ndarray
        Boolean array: True where feature is selected (q_value <= q).
    """
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool)

    # Sort by p-value
    order = np.argsort(p_values)
    sorted_p = p_values[order]

    # BH threshold: p_i <= (i/n) * q
    thresholds = (np.arange(1, n + 1) / n) * q
    below = sorted_p <= thresholds

    # All indices up to and including the largest k with p_k <= threshold
    if not below.any():
        selected_sorted = np.zeros(n, dtype=bool)
    else:
        max_k = np.max(np.where(below)[0])
        selected_sorted = np.arange(n) <= max_k

    # Restore original order
    result = np.empty(n, dtype=bool)
    result[order] = selected_sorted
    return result


# ---------------------------------------------------------------------------
# Feature statistics
# ---------------------------------------------------------------------------


def compute_feature_stats(ic_ts: pd.DataFrame) -> pd.DataFrame:
    """Compute mean IC, ICIR, p-value, and sign consistency from IC timeseries.

    Parameters
    ----------
    ic_ts:
        DataFrame of daily Rank IC values (index=date, columns=features).

    Returns
    -------
    pd.DataFrame
        Summary stats per feature: mean_rank_ic, std_rank_ic, rank_icir, p_value,
        n_obs, hit_rate, sign_consistency_first_half, sign_consistency_second_half.
    """
    records = []
    n_dates = len(ic_ts)
    mid = n_dates // 2

    for col in ic_ts.columns:
        series = ic_ts[col].dropna()
        n = len(series)
        if n < 5:
            records.append({
                "feature": col,
                "mean_rank_ic": np.nan,
                "std_rank_ic": np.nan,
                "rank_icir": np.nan,
                "p_value": 1.0,
                "n_obs": n,
                "hit_rate": np.nan,
                "sign_consistency": np.nan,
            })
            continue

        mean_ic = series.mean()
        std_ic = series.std()
        icir = (mean_ic / std_ic * np.sqrt(252)) if std_ic > 1e-8 else 0.0

        # t-test: H0: mean_ic = 0
        t_stat = mean_ic / (std_ic / np.sqrt(n))
        p_val = float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))

        hit_rate = float((series > 0).mean())

        # Sign consistency across first/second half of train
        first_half = ic_ts[col].iloc[:mid].dropna()
        second_half = ic_ts[col].iloc[mid:].dropna()

        if len(first_half) >= 3 and len(second_half) >= 3:
            sign_h1 = float(np.sign(first_half.mean()))
            sign_h2 = float(np.sign(second_half.mean()))
            sign_consistency = 1.0 if sign_h1 == sign_h2 else 0.0
        else:
            sign_consistency = np.nan

        records.append({
            "feature": col,
            "mean_rank_ic": mean_ic,
            "std_rank_ic": std_ic,
            "rank_icir": icir,
            "p_value": p_val,
            "n_obs": n,
            "hit_rate": hit_rate,
            "sign_consistency": sign_consistency,
        })

    return pd.DataFrame(records).set_index("feature")


# ---------------------------------------------------------------------------
# Main FDR feature selector
# ---------------------------------------------------------------------------


class FDRFeatureSelector:
    """Walk-forward FDR-based feature selector.

    Usage::

        selector = FDRFeatureSelector(fdr_q=0.10, min_abs_rank_ic=0.02,
                                       min_sign_consistency=0.60,
                                       max_features=20)
        selected = selector.select(ic_timeseries)
        stats_df = selector.stats_

    Attributes
    ----------
    stats_ : pd.DataFrame
        Feature statistics for last fit call.
    q_values_ : np.ndarray
        BH q-values for last fit call.
    """

    def __init__(
        self,
        fdr_q: float = 0.10,
        min_abs_rank_ic: float = 0.02,
        min_sign_consistency: float = 0.60,
        max_features: int = 20,
    ) -> None:
        self.fdr_q = fdr_q
        self.min_abs_rank_ic = min_abs_rank_ic
        self.min_sign_consistency = min_sign_consistency
        self.max_features = max_features

        # Results from last select()
        self.stats_: pd.DataFrame | None = None
        self.q_values_: np.ndarray | None = None

    def select(self, ic_timeseries: pd.DataFrame) -> list[str]:
        """Select features using BH FDR on the given IC timeseries.

        All computation is local to the provided ic_timeseries
        (assumed to be from the train window only).

        Parameters
        ----------
        ic_timeseries:
            DataFrame of daily Rank IC (index=date, columns=feature names).

        Returns
        -------
        list[str]
            Selected feature names (ordered by |mean_rank_ic| descending).
        """
        if ic_timeseries.empty or ic_timeseries.shape[1] == 0:
            logger.warning("FDR selector: empty IC timeseries. No features selected.")
            self.stats_ = pd.DataFrame()
            self.q_values_ = np.array([])
            return []

        stats_df = compute_feature_stats(ic_timeseries)
        self.stats_ = stats_df

        # Extract p-values
        p_vals = stats_df["p_value"].fillna(1.0).values
        feature_names = stats_df.index.tolist()

        # BH FDR correction
        bh_selected = benjamini_hochberg(p_vals, q=self.fdr_q)

        # Compute BH q-values for reporting
        n = len(p_vals)
        order = np.argsort(p_vals)
        q_values = np.empty(n)
        q_values[order] = p_vals[order] * n / (np.arange(1, n + 1))
        # Monotone correction: enforce q[k] <= q[k+1]
        for i in range(n - 2, -1, -1):
            q_values[order[i]] = min(q_values[order[i]], q_values[order[i + 1]])
        self.q_values_ = q_values

        # Build selection mask
        candidates: list[str] = []
        for i, feat in enumerate(feature_names):
            if not bh_selected[i]:
                continue

            row = stats_df.loc[feat]
            mean_ic = row["mean_rank_ic"]
            sign_cons = row["sign_consistency"]

            if np.isnan(mean_ic):
                continue
            if abs(mean_ic) < self.min_abs_rank_ic:
                continue
            if not np.isnan(sign_cons) and sign_cons < self.min_sign_consistency:
                continue

            candidates.append(feat)

        # Sort by |mean_rank_ic| descending, then cap
        candidates_sorted = sorted(
            candidates,
            key=lambda f: abs(stats_df.loc[f, "mean_rank_ic"]),
            reverse=True,
        )
        selected = candidates_sorted[: self.max_features]

        logger.info(
            "FDR selector: %d / %d features passed BH FDR + IC + sign filters "
            "(q=%.2f, min_ic=%.3f, min_cons=%.2f). Final selection: %d.",
            len(candidates),
            len(feature_names),
            self.fdr_q,
            self.min_abs_rank_ic,
            self.min_sign_consistency,
            len(selected),
        )
        return selected


# ---------------------------------------------------------------------------
# Walk-forward selection runner
# ---------------------------------------------------------------------------


def run_walk_forward_fdr_selection(
    ic_timeseries_by_window: dict[int, pd.DataFrame],
    window_metadata: list[dict],
    fdr_q: float = 0.10,
    min_abs_rank_ic: float = 0.02,
    min_sign_consistency: float = 0.60,
    max_features: int = 20,
) -> pd.DataFrame:
    """Run FDR feature selection across all walk-forward windows.

    Parameters
    ----------
    ic_timeseries_by_window:
        Dict mapping window_id → IC timeseries DataFrame (train dates × features).
    window_metadata:
        List of dicts with keys: window_id, train_start, train_end.
    fdr_q, min_abs_rank_ic, min_sign_consistency, max_features:
        Selector hyperparameters.

    Returns
    -------
    pd.DataFrame
        Long-format selection results with columns:
        window_id, train_start, train_end, feature, mean_rank_ic,
        rank_icir, p_value, q_value, sign_consistency, selected.
    """
    selector = FDRFeatureSelector(
        fdr_q=fdr_q,
        min_abs_rank_ic=min_abs_rank_ic,
        min_sign_consistency=min_sign_consistency,
        max_features=max_features,
    )

    all_records = []
    meta_by_id = {m["window_id"]: m for m in window_metadata}

    for window_id, ic_ts in ic_timeseries_by_window.items():
        meta = meta_by_id.get(window_id, {})
        train_start = meta.get("train_start", "")
        train_end = meta.get("train_end", "")

        selected = selector.select(ic_ts)
        selected_set = set(selected)

        stats_df = selector.stats_ if selector.stats_ is not None else pd.DataFrame()
        q_values = selector.q_values_

        for i, feat in enumerate(stats_df.index if stats_df is not None else []):
            row = stats_df.loc[feat]
            all_records.append({
                "window_id": window_id,
                "train_start": train_start,
                "train_end": train_end,
                "feature": feat,
                "mean_rank_ic": row.get("mean_rank_ic", np.nan),
                "rank_icir": row.get("rank_icir", np.nan),
                "p_value": row.get("p_value", np.nan),
                "q_value": q_values[i] if q_values is not None and i < len(q_values) else np.nan,
                "sign_consistency": row.get("sign_consistency", np.nan),
                "selected": feat in selected_set,
            })

    return pd.DataFrame(all_records)


# ---------------------------------------------------------------------------
# Feature stability summary
# ---------------------------------------------------------------------------


def compute_feature_stability(selection_df: pd.DataFrame) -> pd.DataFrame:
    """Compute how often each feature was selected across walk-forward windows.

    Parameters
    ----------
    selection_df:
        Output of run_walk_forward_fdr_selection.

    Returns
    -------
    pd.DataFrame
        Per-feature stability stats: n_windows, n_selected, selection_freq,
        mean_rank_ic, mean_sign_consistency.
    """
    if selection_df.empty:
        return pd.DataFrame()

    grouped = selection_df.groupby("feature")
    n_windows = selection_df["window_id"].nunique()

    records = []
    for feat, grp in grouped:
        n_selected = grp["selected"].sum()
        records.append({
            "feature": feat,
            "n_windows": n_windows,
            "n_selected": int(n_selected),
            "selection_freq": float(n_selected) / n_windows if n_windows > 0 else 0.0,
            "mean_rank_ic": grp["mean_rank_ic"].mean(),
            "mean_sign_consistency": grp["sign_consistency"].mean(),
        })

    return pd.DataFrame(records).sort_values("selection_freq", ascending=False)
