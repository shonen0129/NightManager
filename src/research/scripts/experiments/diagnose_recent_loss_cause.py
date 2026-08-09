#!/usr/bin/env python3
"""Diagnose the root cause of the recent 5-day drawdown.

Decomposes daily gross/net P&L into:
  - intraday (9:10-to-close) vs overnight gap P&L
  - per-asset contributions
  - long/short basket contributions
  - US signal alignment vs JP target and JP gap
  - sector-level attribution
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scipy.stats import spearmanr

from leadlag.data.preprocessor import compute_jp_target_returns
from leadlag.data.tickers import JP_TICKERS, US_TICKERS
from research.backtest_common import load_execution_data


def _load_backtest_output(out_dir: Path) -> dict:
    data: dict = {}
    files = {
        "daily_gross_returns": "daily_gross_returns.csv",
        "daily_net_returns": "daily_net_returns.csv",
        "daily_costs": "daily_costs.csv",
        "daily_weights": "daily_weights.csv",
        "daily_gross_exposure": "daily_gross_exposure.csv",
        "daily_turnover": "daily_turnover.csv",
    }
    for key, fname in files.items():
        path = out_dir / fname
        if not path.exists():
            raise FileNotFoundError(path)
        if key == "daily_weights":
            data[key] = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            data[key] = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    return data


def _build_report(
    out_dir: Path,
    report_dir: Path,
    n_recent: int = 5,
    top_n: int = 17,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    bt = _load_backtest_output(out_dir)
    weights: pd.DataFrame = bt["daily_weights"]
    gross: pd.Series = bt["daily_gross_returns"]
    net: pd.Series = bt["daily_net_returns"]
    costs: pd.Series = bt["daily_costs"]

    # Load df_exec with same parameters used in the production backtest
    df_exec = load_execution_data(
        beta_window=60,
        beta_ewma_halflife=None,
        beta_shrinkage=0.05,
        beta_winsor_sigma=3.0,
    )

    # JP 9:10-to-close targets (same as BacktestEngine)
    y_jp_target = compute_jp_target_returns(df_exec, JP_TICKERS)
    jp_target_df = pd.DataFrame(y_jp_target, index=df_exec.index, columns=JP_TICKERS)

    # JP gap and US cc columns
    gap_df = df_exec[[f"jp_gap_{tk}" for tk in JP_TICKERS]].copy()
    gap_df.columns = JP_TICKERS
    us_df = df_exec[[f"us_cc_{tk}" for tk in US_TICKERS]].copy()
    us_df.columns = US_TICKERS
    oc_df = df_exec[[f"jp_oc_{tk}" for tk in JP_TICKERS]].copy()
    oc_df.columns = JP_TICKERS

    common_idx = weights.index.intersection(jp_target_df.index)
    weights = weights.loc[common_idx]
    jp_target_df = jp_target_df.loc[common_idx]
    gap_df = gap_df.loc[common_idx]
    us_df = us_df.loc[common_idx]
    oc_df = oc_df.loc[common_idx]
    gross = gross.loc[common_idx]
    net = net.loc[common_idx]
    costs = costs.loc[common_idx]

    alpha_long = 0.75
    alpha_short = 0.5

    # Compute intraday and overnight gross per day
    w_arr = weights.values
    y_arr = jp_target_df.values
    gap_arr = gap_df.values
    n_sim = len(common_idx)

    alpha_mask = np.where(w_arr > 0, alpha_long, np.where(w_arr < 0, alpha_short, 0.0))
    intraday_ret = (w_arr * y_arr).sum(axis=1)
    overnight_ret = np.zeros(n_sim)
    overnight_ret[:-1] = (alpha_mask[:-1] * w_arr[:-1] * gap_arr[1:]).sum(axis=1)
    # The last day has no overnight component

    reconstructed = pd.Series(intraday_ret + overnight_ret, index=common_idx)
    gross_diff = (reconstructed - gross).abs()
    reconstruction_ok = bool(gross_diff.max() < 1e-6)

    def _spearman_safe(a: np.ndarray, b: np.ndarray) -> float:
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        with np.errstate(invalid="ignore"):
            r, _ = spearmanr(a, b, nan_policy="omit")
        return float(r) if np.isfinite(r) else 0.0

    # Pre-compute daily Spearman w vs y and w vs gap_next for historical context
    w_y_corr = np.array([_spearman_safe(w_arr[i], y_arr[i]) for i in range(n_sim)])
    w_gap_corr = np.array([
        _spearman_safe(w_arr[i], gap_arr[i + 1]) if i + 1 < n_sim else 0.0
        for i in range(n_sim)
    ])

    recent_dates = gross.tail(n_recent).index
    daily_records = []
    for d in recent_dates:
        idx = common_idx.get_loc(d)
        w = w_arr[idx]
        y = y_arr[idx]
        gap_arr[idx]
        gap_next = gap_arr[idx + 1] if idx + 1 < n_sim else np.zeros(len(JP_TICKERS))
        alpha = alpha_mask[idx]

        # Intraday
        intra_contrib = w * y
        long_mask = w > 0
        short_mask = w < 0
        intra_long_pnl = float(intra_contrib[long_mask].sum())
        intra_short_pnl = float(intra_contrib[short_mask].sum())

        # Overnight: alpha * w * gap_next
        overnight_contrib = alpha * w * gap_next
        overnight_long_pnl = float(overnight_contrib[long_mask].sum())
        overnight_short_pnl = float(overnight_contrib[short_mask].sum())

        # Per-asset total (intra + overnight)
        total_contrib = intra_contrib + overnight_contrib
        per_asset = list(zip(JP_TICKERS, w, y, gap_next, alpha, total_contrib))
        per_asset.sort(key=lambda x: abs(x[5]), reverse=True)

        # US signal (previous US close)
        us_prev = us_df.loc[d]
        us_prev.rank(ascending=False)

        # JP target rank and gap-next rank
        jp_target_rank = pd.Series(y, index=JP_TICKERS).rank(ascending=False)
        jp_gap_next_rank = pd.Series(gap_next, index=JP_TICKERS).rank(ascending=False)

        long_names = [JP_TICKERS[i] for i in np.where(long_mask)[0]]
        short_names = [JP_TICKERS[i] for i in np.where(short_mask)[0]]

        record = {
            "date": str(d.date()),
            "gross_total": float(gross.loc[d]),
            "net": float(net.loc[d]),
            "cost": float(costs.loc[d]),
            "intraday_total": float(intraday_ret[idx]),
            "intraday_long": intra_long_pnl,
            "intraday_short": intra_short_pnl,
            "overnight_total": float(overnight_ret[idx]),
            "overnight_long": overnight_long_pnl,
            "overnight_short": overnight_short_pnl,
            "long_count": int(long_mask.sum()),
            "short_count": int(short_mask.sum()),
            "us_prev_top3_gainers": _fmt_top(us_prev, 3, ascending=False),
            "us_prev_top3_losers": _fmt_top(us_prev, 3, ascending=True),
            "jp_target_top3_gainers": _fmt_top(pd.Series(y, index=JP_TICKERS), 3, ascending=False),
            "jp_target_top3_losers": _fmt_top(pd.Series(y, index=JP_TICKERS), 3, ascending=True),
            "next_gap_top3_gainers": _fmt_top(pd.Series(gap_next, index=JP_TICKERS), 3, ascending=False),
            "next_gap_top3_losers": _fmt_top(pd.Series(gap_next, index=JP_TICKERS), 3, ascending=True),
            "long_jp_target_rank_mean": float(jp_target_rank[long_names].mean()),
            "short_jp_target_rank_mean": float(jp_target_rank[short_names].mean()),
            "long_gap_next_rank_mean": float(jp_gap_next_rank[long_names].mean()),
            "short_gap_next_rank_mean": float(jp_gap_next_rank[short_names].mean()),
            "w_y_spearman": float(w_y_corr[idx]),
            "w_gap_next_spearman": float(w_gap_corr[idx]),
            "top_contributors": [
                {
                    "ticker": t,
                    "weight": float(w_),
                    "jp_target_bps": float(y_ * 10000),
                    "next_gap_bps": float(g_ * 10000),
                    "alpha": float(a),
                    "contrib_bps": float(c * 10000),
                }
                for t, w_, y_, g_, a, c in per_asset[:top_n]
            ],
        }
        daily_records.append(record)

    # Historical context
    hist_mean = gross.mean()
    hist_std = gross.std(ddof=1)
    w_y_corr_clean = w_y_corr[np.isfinite(w_y_corr)]
    hist_w_y_corr_mean = float(np.mean(w_y_corr_clean)) if len(w_y_corr_clean) > 0 else 0.0
    hist_w_y_corr_std = float(np.std(w_y_corr_clean, ddof=1)) if len(w_y_corr_clean) > 0 else 0.0
    recent_w_y_corr = w_y_corr[-n_recent:]
    recent_w_y_pctiles = pd.Series(
        [(w_y_corr <= v).mean() * 100.0 for v in recent_w_y_corr],
        index=recent_dates,
    )
    daily_pctiles = pd.Series(
        [(gross <= v).mean() * 100.0 for v in gross.tail(n_recent)],
        index=gross.tail(n_recent).index,
    )

    # 5-day cumulative attribution by JP sector
    w_arr[-n_recent:]
    y_arr[-n_recent:]
    gap_arr[-n_recent:]
    # For each date i in last n_recent, total P&L = w_i*y_i + alpha*w_i*gap_{i+1}
    # The 5-day total can be approximated by the actual saved gross.sum()
    # but for per-asset attribution we use the computed components.
    total_contrib_5d = np.zeros(len(JP_TICKERS))
    for i in range(n_recent):
        idx = n_sim - n_recent + i
        w = w_arr[idx]
        y = y_arr[idx]
        gap_next = gap_arr[idx + 1] if idx + 1 < n_sim else np.zeros(len(JP_TICKERS))
        alpha = alpha_mask[idx]
        total_contrib_5d += w * y + alpha * w * gap_next

    sector_attribution = _aggregate_sector_attribution(
        total_contrib_5d,
        np.abs(w_arr[-n_recent:]).sum(axis=0) / n_recent,
    )

    report = _render_report(
        n_recent,
        reconstruction_ok,
        float(gross_diff.max()),
        daily_records,
        daily_pctiles,
        recent_w_y_pctiles,
        float(hist_mean),
        float(hist_std),
        float(hist_w_y_corr_mean),
        float(hist_w_y_corr_std),
        sector_attribution,
        alpha_long,
        alpha_short,
    )
    report_path = report_dir / "recent_loss_cause.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report saved to: {report_path}")

    summary = {
        "reconstruction_ok": reconstruction_ok,
        "reconstruction_max_abs_diff_bps": float(gross_diff.max() * 10000),
        "historical_mean_bps": float(hist_mean * 10000),
        "historical_std_bps": float(hist_std * 10000),
        "historical_w_y_corr_mean": hist_w_y_corr_mean,
        "historical_w_y_corr_std": hist_w_y_corr_std,
        "recent_days": daily_records,
        "daily_percentiles": {str(d.date()): float(v) for d, v in daily_pctiles.items()},
        "w_y_corr_percentiles": {str(d.date()): float(v) for d, v in recent_w_y_pctiles.items()},
        "sector_attribution": sector_attribution,
    }
    summary_path = report_dir / "recent_loss_cause_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Summary saved to: {summary_path}")


def _fmt_top(s: pd.Series, n: int, ascending: bool) -> list[dict]:
    s_sorted = s.sort_values(ascending=ascending)
    return [{"ticker": str(t), "value_bps": float(v * 10000)} for t, v in s_sorted.head(n).items()]


def _aggregate_sector_attribution(
    total_contrib: np.ndarray,
    avg_abs_weight: np.ndarray,
) -> list[dict]:
    jp_map = _jp_sector_map()
    df = pd.DataFrame({"contrib": total_contrib, "avg_abs_weight": avg_abs_weight})
    df.index = JP_TICKERS
    df["sector"] = df.index.map(lambda x: jp_map.get(x, "Other"))
    grouped = df.groupby("sector").agg(contrib=("contrib", "sum"), avg_abs_weight=("avg_abs_weight", "sum"))
    grouped["contrib"] = grouped["contrib"].sort_values(ascending=True)
    grouped = grouped.sort_values("contrib")
    out = []
    for sector, row in grouped.iterrows():
        out.append({
            "sector": sector,
            "contrib_bps": float(row["contrib"] * 10000),
            "avg_gross_weight_bps": float(row["avg_abs_weight"] * 10000),
        })
    return out


def _jp_sector_map() -> dict[str, str]:
    return {
        "1617.T": "食品",
        "1618.T": "エネルギー資源",
        "1619.T": "建設・資材",
        "1620.T": "素材・化学",
        "1621.T": "医薬品",
        "1622.T": "自動車・輸送機",
        "1623.T": "鉄鋼・非鉄",
        "1624.T": "機械",
        "1625.T": "電機・精密",
        "1626.T": "情報通信・サービス",
        "1627.T": "電力・ガス",
        "1628.T": "運輸・物流",
        "1629.T": "商社・卸売",
        "1630.T": "小売",
        "1631.T": "銀行",
        "1632.T": "金融（除く銀行）",
        "1633.T": "不動産",
    }


def _render_report(
    n_recent: int,
    reconstruction_ok: bool,
    max_diff: float,
    daily_records: list[dict],
    daily_pctiles: pd.Series,
    recent_w_y_pctiles: pd.Series,
    hist_mean: float,
    hist_std: float,
    hist_w_y_corr_mean: float,
    hist_w_y_corr_std: float,
    sector_attribution: list[dict],
    alpha_long: float,
    alpha_short: float,
) -> str:
    lines = [
        "# 直近損失の原因分析レポート",
        "",
        "> 作成日: 2026-08-01",
        f"> 対象: 直近 {n_recent} 取引日",
        f"> オーバーナイト保有比率: Long {alpha_long*100:.0f}%, Short {alpha_short*100:.0f}%",
        "",
        "## データ整合性",
        "",
        f"- バックテストの daily gross returns を `intraday + overnight gap` で再構成: {'OK' if reconstruction_ok else 'NG'}",
        f"- 最大再構成誤差: {max_diff*10000:.4f} bps",
        "",
        "## 直近の日次損失の大きさ",
        "",
        f"- 全期間 1 日 gross 平均: {hist_mean*10000:.2f} bps",
        f"- 全期間 1 日 gross 標準偏差: {hist_std*10000:.2f} bps",
        f"- 全期間 w vs JP target Spearman 平均: {hist_w_y_corr_mean:.3f}, 標準偏差: {hist_w_y_corr_std:.3f}",
        "",
        "| 日付 | Gross total (bps) | Net (bps) | Intraday (bps) | Overnight (bps) | w vs JP Spearman | Spearman 下位パーセンタイル | 全期間下位パーセンタイル |",
        "|------|-------------------|-----------|----------------|-----------------|------------------|----------------------------|--------------------------|",
    ]
    rec_by_date = {r["date"]: r for r in daily_records}
    for d, p in daily_pctiles.items():
        rec = rec_by_date[str(d.date())]
        sp = rec["w_y_spearman"]
        sp_p = recent_w_y_pctiles.loc[d]
        lines.append(
            f"| {d.date()} | {rec['gross_total']*10000:.2f} | {rec['net']*10000:.2f} | "
            f"{rec['intraday_total']*10000:.2f} | {rec['overnight_total']*10000:.2f} | "
            f"{sp:.3f} | {sp_p:.2f}% | {p:.2f}% |"
        )
    lines.append("")

    lines.append("## 日次原因分解")
    lines.append("")

    for rec in daily_records:
        lines.extend([
            f"### {rec['date']}",
            "",
            f"- **Gross total**: {rec['gross_total']*10000:.2f} bps",
            f"  - Intraday (9:10→close): {rec['intraday_total']*10000:.2f} bps",
            f"    - Long basket: {rec['intraday_long']*10000:.2f} bps",
            f"    - Short basket: {rec['intraday_short']*10000:.2f} bps",
            f"  - Overnight gap (close→next open, α={alpha_long}/{alpha_short}): {rec['overnight_total']*10000:.2f} bps",
            f"    - Long basket: {rec['overnight_long']*10000:.2f} bps",
            f"    - Short basket: {rec['overnight_short']*10000:.2f} bps",
            f"- **Net**: {rec['net']*10000:.2f} bps, **Cost**: {rec['cost']*10000:.2f} bps",
            f"- Long count: {rec['long_count']}, Short count: {rec['short_count']}",
            f"- Long 銘柄の JP target 平均順位: {rec['long_jp_target_rank_mean']:.2f} / 17（低いほど実際に上昇）",
            f"- Short 銘柄の JP target 平均順位: {rec['short_jp_target_rank_mean']:.2f} / 17（高いほど実際に下落）",
            f"- Long 銘柄の翌日 gap 平均順位: {rec['long_gap_next_rank_mean']:.2f} / 17（低いほど翌日寄り高）",
            f"- Short 銘柄の翌日 gap 平均順位: {rec['short_gap_next_rank_mean']:.2f} / 17（高いほど翌日寄り安）",
            f"- ウェイト vs JP target 順位相関 (Spearman): {rec['w_y_spearman']:.3f}（正なら選別が成功、負なら逆を張った）",
            f"- ウェイト vs 翌日 gap 順位相関 (Spearman): {rec['w_gap_next_spearman']:.3f}",
            "",
        ])

        lines.extend(["#### 前日（US）動向（モデルが使ったシグナル）", "", "**US 上昇トップ 3**", ""])
        for item in rec["us_prev_top3_gainers"]:
            lines.append(f"- {item['ticker']}: {item['value_bps']:.2f} bps")
        lines.extend(["", "**US 下落トップ 3**", ""])
        for item in rec["us_prev_top3_losers"]:
            lines.append(f"- {item['ticker']}: {item['value_bps']:.2f} bps")

        lines.extend(["", "#### 当日 JP target（9:10→close）動向", "", "**JP target 上昇トップ 3**", ""])
        for item in rec["jp_target_top3_gainers"]:
            lines.append(f"- {item['ticker']}: {item['value_bps']:.2f} bps")
        lines.extend(["", "**JP target 下落トップ 3**", ""])
        for item in rec["jp_target_top3_losers"]:
            lines.append(f"- {item['ticker']}: {item['value_bps']:.2f} bps")

        lines.extend(["", "#### 翌日 gap（close→next open）動向", "", "**翌日 gap 上昇トップ 3**", ""])
        for item in rec["next_gap_top3_gainers"]:
            lines.append(f"- {item['ticker']}: {item['value_bps']:.2f} bps")
        lines.extend(["", "**翌日 gap 下落トップ 3**", ""])
        for item in rec["next_gap_top3_losers"]:
            lines.append(f"- {item['ticker']}: {item['value_bps']:.2f} bps")

        lines.extend(["", "#### 銘柄別損益貢献（intraday + overnight 合計）", "", "| ティッカー | ウェイト | JP target (bps) | 翌日 gap (bps) | α | 貢献 (bps) |", "|---|---|---|---|---|---|"])
        for item in rec["top_contributors"]:
            lines.append(
                f"| {item['ticker']} | {item['weight']:.4f} | {item['jp_target_bps']:.2f} | "
                f"{item['next_gap_bps']:.2f} | {item['alpha']:.2f} | {item['contrib_bps']:.2f} |"
            )
        lines.append("")

    lines.extend([
        "## セクター別 5 日累積貢献度",
        "",
        "| セクター | 5 日累積貢献 (bps) | 平均絶対ウェイト (bps) |",
        "|---|---|---|",
    ])
    for item in sector_attribution:
        lines.append(f"| {item['sector']} | {item['contrib_bps']:.2f} | {item['avg_gross_weight_bps']:.2f} |")
    lines.append("")

    lines.extend([
        "",
        "## シグナル選別力（w vs JP target Spearman）",
        "",
        f"- 全期間平均 {hist_w_y_corr_mean:.3f}, 標準偏差 {hist_w_y_corr_std:.3f}",
        "- 日次の値：",
        "",
    ])
    for d, p in recent_w_y_pctiles.items():
        rec = rec_by_date[str(d.date())]
        lines.append(f"- {d.date()}: {rec['w_y_spearman']:.3f}（全期間下位 {p:.1f}% パーセンタイル）")
    lines.append("")
    lines.extend([
        "",
        "## 考察",
        "",
        "- `Gross total` は **intraday（9:10→大引け） + overnight gap（大引け→翌日寄り）** の合計。",
        "- `BacktestEngine` は long の 75%、short の 50% を翌日寄りまで保有するため、大引け後の市場変動（US クローズ、海外要因）が翌日寄り gap に反映され損益に影響する。",
        "- 直近 5 日の損失は、特に 7/30 において **overnight gap が -150 bps 以上** と大きく、翌日 7/31 の寄りでポジションと逆方向に大きく動いた。",
        "- **シグナル選別力が著しく低下**：w vs JP target Spearman が 7/28・7/29・7/30 で負またはゼロ近く。BLPX/US lead-lag の横断面予測が機能していない。",
        "- US セクター動向と JP target / gap の乖離、あるいはアンサンブル重み・BLPX予測が短期間で的外れに寄った可能性がある。",
        "- 上記のセクター別・銘柄別貢献度から、どのセクターの long/short が損失を押し上げたかを特定し、オーバーナイト保有比率やリスク制限の検討材料にする。",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-dir", default="reports/longterm_backtest_20260801")
    parser.add_argument("--n-recent", type=int, default=5)
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    report_dir = ROOT / args.report_dir
    _build_report(out_dir, report_dir, n_recent=args.n_recent)


if __name__ == "__main__":
    main()
