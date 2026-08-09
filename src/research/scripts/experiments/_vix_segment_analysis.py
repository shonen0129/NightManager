#!/usr/bin/env python3
"""Segment baseline net returns by US/JP VIX regime and analyze large gain/loss days."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
RESULTS_DIR = ROOT / "results" / "vix_regime_overlay"
REPORT_DIR = ROOT / "reports" / "vix_regime_overlay"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_returns(name: str) -> pd.Series:
    path = RESULTS_DIR / f"daily_returns_{name}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.iloc[:, 0]


def load_vix() -> pd.DataFrame:
    return pd.read_csv(
        ROOT / "market_data" / "vix_regime_overlay" / "vix_cache.csv",
        index_col=0,
        parse_dates=True,
    )


def compute_z_scores(series: pd.Series, window: int = 60, min_periods: int = 20) -> pd.Series:
    log = np.log(series)
    roll = log.rolling(window=window, min_periods=min_periods)
    return (log - roll.mean()) / roll.std()


def make_segments(us_z: pd.Series, jp_z: pd.Series, threshold: float = 0.5) -> pd.Series:
    """Four VIX regimes based on z-scores."""
    conditions = [
        (us_z > threshold) & (jp_z <= threshold),   # US high, JP low
        (us_z <= threshold) & (jp_z > threshold),   # US low, JP high
        (us_z > threshold) & (jp_z > threshold),    # both high
        (us_z <= threshold) & (jp_z <= threshold),  # both low
    ]
    choices = ["US_high_JP_low", "US_low_JP_high", "both_high", "both_low"]
    return pd.Series(np.select(conditions, choices, default="unknown"), index=us_z.index)


def regime_stats(rets: pd.Series, segments: pd.Series) -> pd.DataFrame:
    records = []
    for seg in ["US_high_JP_low", "US_low_JP_high", "both_high", "both_low"]:
        mask = segments == seg
        sub = rets[mask].dropna()
        if len(sub) == 0:
            continue
        records.append({
            "regime": seg,
            "n_days": int(len(sub)),
            "pct_days": float(mask.mean() * 100),
            "mean_ret_bps": float(sub.mean() * 10000),
            "std_ret_bps": float(sub.std() * 10000),
            "sharpe": float(sub.mean() / sub.std() * np.sqrt(252)) if sub.std() > 0 else 0.0,
            "total_ret_pct": float(((1 + sub).prod() - 1) * 100),
            "median_ret_bps": float(sub.median() * 10000),
            "max_ret_bps": float(sub.max() * 10000),
            "min_ret_bps": float(sub.min() * 10000),
        })
    return pd.DataFrame(records)


def tail_day_analysis(rets: pd.Series, us_z: pd.Series, jp_z: pd.Series, segments: pd.Series, q: float = 0.10) -> pd.DataFrame:
    """For top/bottom q return days, show average US/JP VIX z and regime distribution."""
    rets_clean = rets.dropna()
    len(rets_clean)
    top_q = rets_clean.quantile(1 - q)
    bot_q = rets_clean.quantile(q)

    top_mask = rets_clean >= top_q
    bot_mask = rets_clean <= bot_q

    records = []
    for label, mask in [("top", top_mask), ("bottom", bot_mask)]:
        idx = rets_clean.index[mask]
        records.append({
            "tail": label,
            "n_days": int(mask.sum()),
            "mean_ret_bps": float(rets_clean[mask].mean() * 10000),
            "mean_us_vix_z": float(us_z.loc[idx].mean()),
            "mean_jp_vix_z": float(jp_z.loc[idx].mean()),
            "mean_spread_z": float((jp_z - us_z).loc[idx].mean()),
            "pct_US_high_JP_low": float((segments.loc[idx] == "US_high_JP_low").mean() * 100),
            "pct_US_low_JP_high": float((segments.loc[idx] == "US_low_JP_high").mean() * 100),
            "pct_both_high": float((segments.loc[idx] == "both_high").mean() * 100),
            "pct_both_low": float((segments.loc[idx] == "both_low").mean() * 100),
        })
    return pd.DataFrame(records)


def overlay_impact_by_regime(
    base_rets: pd.Series,
    overlay_rets: pd.Series,
    segments: pd.Series,
) -> pd.DataFrame:
    """Compare baseline vs overlay within each VIX regime."""
    records = []
    for seg in ["US_high_JP_low", "US_low_JP_high", "both_high", "both_low"]:
        mask = segments == seg
        b = base_rets[mask].dropna()
        o = overlay_rets[mask].dropna()
        if len(b) == 0 or len(o) == 0:
            continue
        common = b.index.intersection(o.index)
        diff = (o.loc[common] - b.loc[common]) * 10000
        records.append({
            "regime": seg,
            "n_days": int(len(common)),
            "base_mean_bps": float(b.loc[common].mean() * 10000),
            "overlay_mean_bps": float(o.loc[common].mean() * 10000),
            "mean_diff_bps": float(diff.mean()),
            "win_rate_pct": float((diff > 0).mean() * 100),
            "tail_bottom10_avg_diff_bps": float(diff.nsmallest(int(np.ceil(0.10 * len(diff)))).mean()),
            "tail_top10_avg_diff_bps": float(diff.nlargest(int(np.ceil(0.10 * len(diff)))).mean()),
        })
    return pd.DataFrame(records)


def main():
    base = load_returns("baseline")
    overlay_jp = load_returns("discrete_jp_led_w60")
    overlay_global = load_returns("discrete_global_stress_w60")
    overlay_cont = load_returns("continuous_spread_w60")

    vix = load_vix()
    common = base.index.intersection(vix.index)
    base = base.loc[common]
    overlay_jp = overlay_jp.loc[common]
    overlay_global = overlay_global.loc[common]
    overlay_cont = overlay_cont.loc[common]

    us_z = compute_z_scores(vix["us_vix"].reindex(common).ffill().bfill())
    jp_z = compute_z_scores(vix["jp_vix"].reindex(common).ffill().bfill())
    jp_z - us_z
    segments = make_segments(us_z, jp_z, threshold=0.5)

    # Regime stats for baseline
    stats = regime_stats(base, segments)
    stats.to_csv(RESULTS_DIR / "baseline_regime_stats.csv", index=False)

    # Tail day analysis
    tail = tail_day_analysis(base, us_z, jp_z, segments, q=0.10)
    tail.to_csv(RESULTS_DIR / "baseline_tail_analysis.csv", index=False)

    # Overlay impact by regime
    impacts = {
        "discrete_jp_led_w60": overlay_jp,
        "discrete_global_stress_w60": overlay_global,
        "continuous_spread_w60": overlay_cont,
    }
    all_impact = []
    for name, o_rets in impacts.items():
        df = overlay_impact_by_regime(base, o_rets, segments)
        df["overlay"] = name
        all_impact.append(df)
    impact_df = pd.concat(all_impact, ignore_index=True)
    impact_df.to_csv(RESULTS_DIR / "overlay_impact_by_regime.csv", index=False)

    # Build report
    lines = [
        "# VIX Regime Segment Analysis: Baseline Large Return/Loss Days",
        "",
        "- Period: 2018-04-01 ~ 2024-12-31",
        "- VIX z-score: 60-day rolling on log VIX",
        "- Regime threshold: z > 0.5 = high",
        "- Tail definition: top/bottom 10% of daily net returns",
        "",
        "## Baseline Returns by VIX Regime",
        "",
    ]
    lines.extend(_df_to_md(stats))
    lines.append("")

    lines.append("## Tail Day Characteristics (top/bottom 10%)")
    lines.append("")
    lines.extend(_df_to_md(tail))
    lines.append("")

    lines.append("## Overlay Impact by Regime (overlay - baseline, bps)")
    lines.append("")
    for name in ["discrete_jp_led_w60", "discrete_global_stress_w60", "continuous_spread_w60"]:
        lines.append(f"### {name}")
        lines.append("")
        sub = impact_df[impact_df["overlay"] == name].drop(columns=["overlay"])
        lines.extend(_df_to_md(sub))
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    # Auto-generate key bullet points
    if not tail.empty:
        top = tail[tail["tail"] == "top"].iloc[0]
        bot = tail[tail["tail"] == "bottom"].iloc[0]
        lines.append(f"- **Top 10% return days** ({int(top['n_days'])} days) have mean US VIX z = {top['mean_us_vix_z']:.2f}, JP VIX z = {top['mean_jp_vix_z']:.2f}.")
        lines.append(f"- **Bottom 10% return days** ({int(bot['n_days'])} days) have mean US VIX z = {bot['mean_us_vix_z']:.2f}, JP VIX z = {bot['mean_jp_vix_z']:.2f}.")
        if top["mean_jp_vix_z"] > bot["mean_jp_vix_z"]:
            lines.append("- **JP VIX is actually higher on large *gain* days than on large *loss* days.** This is the opposite of the JP-led-shock hypothesis and explains why cutting gross in JP-high days hurt alpha.")
        else:
            lines.append("- JP VIX is higher on large loss days, consistent with the hypothesis.")

    if not stats.empty:
        best = stats.loc[stats["total_ret_pct"].idxmax()]
        worst = stats.loc[stats["total_ret_pct"].idxmin()]
        lines.append(f"- **Best regime for baseline**: {best['regime']} (total return {best['total_ret_pct']:.1f}%, Sharpe {best['sharpe']:.2f}).")
        lines.append(f"- **Worst regime for baseline**: {worst['regime']} (total return {worst['total_ret_pct']:.1f}%, Sharpe {worst['sharpe']:.2f}).")

    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `results/vix_regime_overlay/baseline_regime_stats.csv`")
    lines.append("- `results/vix_regime_overlay/baseline_tail_analysis.csv`")
    lines.append("- `results/vix_regime_overlay/overlay_impact_by_regime.csv`")
    lines.append("")

    report_text = "\n".join(lines)
    (REPORT_DIR / "segment_analysis.md").write_text(report_text)
    (RESULTS_DIR / "segment_analysis.md").write_text(report_text)
    print("Segment analysis report saved.")
    print(report_text)


def _df_to_md(df: pd.DataFrame) -> list[str]:
    """Convert DataFrame to simple markdown table lines."""
    header = df.columns.tolist()
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for _, row in df.iterrows():
        vals = []
        for col in header:
            v = row[col]
            if isinstance(v, (int, np.integer)):
                vals.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                vals.append(f"{float(v):.3f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


if __name__ == "__main__":
    main()
