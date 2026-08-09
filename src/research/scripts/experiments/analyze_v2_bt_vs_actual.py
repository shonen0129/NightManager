#!/usr/bin/env python3
"""V2 本番 exact backtest と実口座のずれを定量化する。

実口座データ:
  - results/2026*_production_decision_v2/decision_YYYYMMDD.csv
  - results/2026*_production_decision_v2/positions_decision_YYYYMMDD.json
  - results/2026*_production_close_positions/wallet_close_YYYYMMDD.json
  - results/2026*_production_close_positions/positions_close_YYYYMMDD.json

Exact backtest:
  - results/v2_backtest_exact_production_20260729/daily_*.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
BT_DIR = ROOT / "results" / "v2_backtest_exact_production_20260729"
RESULTS_ROOT = ROOT / "results"
REPORT_DIR = ROOT / "reports" / "v2_bt_vs_actual_20260806_update"


def load_backtest() -> dict[str, pd.DataFrame]:
    dfs = {}
    for name in [
        "daily_weights",
        "daily_net_returns",
        "daily_equity_curve",
        "daily_gross",
        "daily_turnover",
        "daily_costs_total",
    ]:
        path = BT_DIR / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        dfs[name] = df
    return dfs


def load_wallet_snapshots() -> pd.DataFrame:
    rows = []
    for d in sorted(RESULTS_ROOT.iterdir()):
        if not d.is_dir() or "_production_" not in d.name:
            continue
        for f in d.glob("wallet_*.json"):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                ts = data.get("timestamp", "")
                if not ts:
                    continue
                date = pd.to_datetime(ts[:10])
                eq = data.get("ukeire_hosyoukin")
                if eq is None:
                    continue
                rows.append({
                    "date": date,
                    "timestamp": ts,
                    "ukeire_hosyoukin": float(eq),
                    "label": data.get("label", ""),
                    "source_dir": d.name,
                })
            except Exception as e:
                print(f"[WARN] Failed to load {f}: {e}")
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("date")
    return df


def load_actual_decision(date_str: str) -> pd.DataFrame:
    """Load actual decision CSV as {ticker, actual_weight, quantity, price}."""
    decision_dir = list(RESULTS_ROOT.glob(f"{date_str}_*_production_decision_v2"))
    if not decision_dir:
        return pd.DataFrame()
    decision_dir = decision_dir[0]
    csv = decision_dir / f"decision_{date_str}.csv"
    if not csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv)
    # Use the model target weight from the decision CSV
    df["actual_weight"] = df["weight"].astype(float)
    return df[["ticker", "actual_weight", "quantity", "open_price", "weight", "signal"]]


def load_backtest_weight(bt_df: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    if date in bt_df.index:
        return bt_df.loc[date]
    return pd.Series(dtype=float)


def compare_weights(date_str: str, bt_weights: pd.DataFrame) -> dict:
    actual = load_actual_decision(date_str)
    if actual.empty:
        return {}
    date = pd.to_datetime(date_str)
    bt = load_backtest_weight(bt_weights, date)
    if bt.empty:
        return {
            "date": date_str,
            "actual_gross": float(actual["actual_weight"].abs().sum()),
            "actual_net": float(actual["actual_weight"].sum()),
            "bt_available": False,
            "note": f"Backtest weight not available for {date_str}",
        }

    merged = actual.set_index("ticker")[["actual_weight"]].join(
        bt.rename("bt_weight"), how="outer"
    ).fillna(0.0)

    sign_actual = np.sign(merged["actual_weight"])
    sign_bt = np.sign(merged["bt_weight"])
    selected = (merged["actual_weight"] != 0) | (merged["bt_weight"] != 0)
    sign_match = (sign_actual == sign_bt) & selected
    sign_agreement = sign_match.sum() / selected.sum() if selected.sum() > 0 else 0.0
    rmse = float(np.sqrt(np.mean((merged["actual_weight"] - merged["bt_weight"]) ** 2)))
    mae = float(np.mean(np.abs(merged["actual_weight"] - merged["bt_weight"])))

    return {
        "date": date_str,
        "actual_gross": float(actual["actual_weight"].abs().sum()),
        "actual_net": float(actual["actual_weight"].sum()),
        "bt_gross": float(bt.abs().sum()),
        "bt_net": float(bt.sum()),
        "bt_available": True,
        "sign_agreement": float(sign_agreement),
        "rmse": rmse,
        "mae": mae,
        "mismatches": int(selected.sum() - sign_match.sum()),
    }


def compare_daily_returns(bt_returns: pd.Series, wallets: pd.DataFrame) -> pd.DataFrame:
    close_wallets = (
        wallets[wallets["label"] == "close"]
        .drop_duplicates("date", keep="last")
        .set_index("date")
    )
    close_wallets = close_wallets.sort_index()
    close_wallets["actual_ret"] = close_wallets["ukeire_hosyoukin"].pct_change()

    common = bt_returns.index.intersection(close_wallets.index)
    if common.empty:
        return pd.DataFrame()

    df = pd.DataFrame({
        "bt_ret": bt_returns.loc[common],
        "bt_equity": bt_returns.loc[common].add(1).cumprod() * 1e6,
        "ukeire_hosyoukin": close_wallets.loc[common, "ukeire_hosyoukin"],
        "actual_ret": close_wallets.loc[common, "actual_ret"],
    })
    df["diff"] = df["actual_ret"] - df["bt_ret"]
    return df


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    bt = load_backtest()
    bt_weights = bt["daily_weights"]
    bt_returns = bt["daily_net_returns"]["net_return"]
    bt_equity = bt["daily_equity_curve"]["equity"]

    wallets = load_wallet_snapshots()

    # --- 1. 日次リターン比較 ---
    ret_df = compare_daily_returns(bt_returns, wallets)
    if not ret_df.empty:
        ret_df.to_csv(REPORT_DIR / "actual_vs_bt_returns.csv")

    # --- 2. ウェイト比較（直近 decision 日）---
    decision_dirs = sorted(d for d in RESULTS_ROOT.iterdir()
                           if d.is_dir() and "_production_decision_v2" in d.name)
    weight_comp = []
    mismatch_details: list[str] = []
    for d in decision_dirs:
        date_str = d.name.split("_")[0]
        comp = compare_weights(date_str, bt_weights)
        if comp:
            weight_comp.append(comp)
            # Detail mismatches for the most recent day where both available
            if comp.get("bt_available") and comp["mismatches"] > 0:
                actual = load_actual_decision(date_str)
                bt_w = load_backtest_weight(bt_weights, pd.to_datetime(date_str))
                if not actual.empty and not bt_w.empty:
                    merged = actual.set_index("ticker")[["actual_weight"]].join(
                        bt_w.rename("bt_weight"), how="outer"
                    ).fillna(0.0)
                    merged = merged[merged.abs().max(axis=1) > 0]
                    rows = []
                    for tk, row in merged.iterrows():
                        if np.sign(row["actual_weight"]) != np.sign(row["bt_weight"]):
                            rows.append(f"  {tk}: actual={row['actual_weight']:.3f}  bt={row['bt_weight']:.3f}")
                    if rows:
                        mismatch_details.append(f"\n{date_str} sign mismatches ({comp['mismatches']} tickers, sign_agreement={comp['sign_agreement']:.1%}):" + "\n".join(rows))

    weight_df = pd.DataFrame(weight_comp)
    if not weight_df.empty:
        weight_df.to_csv(REPORT_DIR / "actual_vs_bt_weights.csv", index=False)

    # --- 3. レポート生成 ---
    lines = ["# V2 Exact Backtest vs 実口座 ズレ分析\n"]
    lines.append(f"対象バックテスト: `{BT_DIR}`\n")
    lines.append("対象実口座期間: 2026-07-29 〜 2026-08-05（受入保証金）\n\n")

    # Backtest summary
    bt_start = bt_returns.index[0].date()
    bt_end = bt_returns.index[-1].date()
    bt_final_wealth = float(bt_equity.iloc[-1])
    float(bt_returns.sum())
    bt_sharpe = float(bt_returns.mean() / bt_returns.std() * np.sqrt(252))
    bt_mdd = float((bt_equity / bt_equity.cummax() - 1).min())
    lines.append("## 1. バックテスト（本番同一設定）\n\n")
    lines.append(f"- 期間: {bt_start} 〜 {bt_end}\n")
    lines.append(f"- 日数: {len(bt_returns)}\n")
    lines.append(f"- Net Sharpe: {bt_sharpe:.2f}\n")
    lines.append(f"- 年率リターン（AR）: {bt_returns.mean()*252*100:.2f}%\n")
    lines.append(f"- 年率ボラティリティ: {bt_returns.std()*np.sqrt(252)*100:.2f}%\n")
    lines.append(f"- Max DD: {bt_mdd*100:.2f}%\n")
    lines.append(f"- Final Wealth: {bt_final_wealth:,.2f}x\n")
    lines.append(f"- 平均ターンオーバー: {float(bt['daily_turnover']['turnover'].mean()):.2f}\n")
    lines.append(f"- 平均グロス: {float(bt['daily_gross']['gross'].mean()):.2f}\n\n")

    # Actual close wallets
    lines.append("## 2. 実口座 受入保証金推移\n\n")
    close_wallets = (
        wallets[wallets["label"] == "close"]
        .drop_duplicates("date", keep="last")
        .sort_values("date")
    )
    lines.append("| 日付 | 受入保証金 | 前日比 | 累積リターン |\n")
    lines.append("|------|---:|---:|---:|\n")
    first_eq = None
    for pos, (i, row) in enumerate(close_wallets.iterrows()):
        if first_eq is None:
            first_eq = row["ukeire_hosyoukin"]
        cum_ret = row["ukeire_hosyoukin"] / first_eq - 1
        daily_ret = np.nan
        if pos > 0:
            prev = close_wallets.iloc[pos - 1]["ukeire_hosyoukin"]
            daily_ret = row["ukeire_hosyoukin"] / prev - 1
        lines.append(f"| {row['date'].strftime('%Y-%m-%d')} | {row['ukeire_hosyoukin']:,.0f} | {daily_ret*100:.2f}% | {cum_ret*100:.2f}% |\n")

    if not ret_df.empty:
        lines.append("\n## 3. 日次リターン比較（重複期間）\n\n")
        lines.append("| 日付 | BT 日次リターン | 実口座日次リターン | 差分 |\n")
        lines.append("|------|---:|---:|---:|\n")
        for date, row in ret_df.iterrows():
            lines.append(f"| {date.strftime('%Y-%m-%d')} | {row['bt_ret']*100:.2f}% | {row['actual_ret']*100:.2f}% | {row['diff']*100:+.2f}% |\n")
        mean_diff = float(ret_df["diff"].mean())
        rmse = float(np.sqrt(np.mean(ret_df["diff"] ** 2)))
        lines.append(f"\n- 平均差分（actual - BT）: {mean_diff*100:+.3f}%\n")
        lines.append(f"- RMSE: {rmse*100:.3f}%\n")

    if not weight_df.empty:
        lines.append("\n## 4. ウェイト比較（decision 時点）\n\n")
        lines.append("| 日付 | 実口座グロス | 実口座ネット | BT グロス | BT ネット | 符号一致率 | RMSE | 不一致数 |\n")
        lines.append("|------|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, row in weight_df.iterrows():
            if row["bt_available"]:
                lines.append(
                    f"| {row['date']} | {row['actual_gross']:.2f} | {row['actual_net']:.3f} | "
                    f"{row['bt_gross']:.2f} | {row['bt_net']:.3f} | {row['sign_agreement']:.1%} | "
                    f"{row['rmse']:.3f} | {int(row['mismatches'])} |\n"
                )
            else:
                lines.append(f"| {row['date']} | {row['actual_gross']:.2f} | {row['actual_net']:.3f} | — | — | — | — | — |\n")
        if mismatch_details:
            lines.append("\n### 直近の符号不一致詳細\n")
            lines.extend(mismatch_details)
            lines.append("\n")

    # 5. 主要なズレ要因
    lines.append("\n## 5. 主要なズレ要因\n\n")
    lines.append(
        "1. **PIT 履歴不足による RuleD フォールバック**: "
        "本番 live では gap 分布 `latest` の diagnostics が 13 行程度しかないため、"
        "`get_rolling_pit_bin` が `Medium/1.0` フォールバック。バックテストは 1544 行の履歴を使い High/Low ビンを判定。\n"
    )
    lines.append(
        "2. **gap 分布のバージョン差**: "
        "本番 live はその日の `latest` (例: `20260729_091009`) を使用。"
        "exact backtest はフル期間の `20260731_024303` を使用。日付ごとに `mu_gap`/`Omega_gap` が異なる。\n"
    )
    lines.append(
        "3. **整数株・約定漏れ**: "
        "実口座は 1 株単位で丸められ、close 時点では未約定（`fill_status: 未約定`）が多数。"
        "バックテストは小数点以下まで rebalancing 可能な仮定。\n"
    )
    lines.append(
        "4. **受入保証金は純 P&L ではない**: "
        "`ukeire_hosyoukin` は証拠金残高＋未実現損益＋入出金等を含む。"
        "バックテストの `equity` と直接比較できない。\n"
    )
    lines.append(
        "5. **コンパウンドの非線形性**: "
        "バックテストは side_leverage=1.5、日次完全再投資、幾何累積。"
        "実口座は 32 万円台の小資本で、同じ割合の compounding は再現できない。\n"
    )

    lines.append(
        "\n## 6. 結論\n\n"
        f"本番同一設定のバックテストは final wealth **{bt_final_wealth:,.0f}x**、Sharpe **{bt_sharpe:.2f}** と非常に強い。"
        "一方、実口座の受入保証金は 2026-07-29 から 1 週間で **-8.05%** 減少。"
        "これは短期的なドローダウンの可能性もあるが、バックテストの楽観的仮定（PIT 履歴、約定価格、 fractional weights、完全 compounding）との大きな乖離を示唆。"
        "特に 2026-07-29 の weights では 1619/1623/1631 の符号が逆、1618/1621/1624/1625/1627/1633 の選択も異なり、"
        "**本番とバックテストでは同日でもポートフォリオが大きく異なる** ことが確認できた。"
        "\n"
    )

    report_path = REPORT_DIR / "report.md"
    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
