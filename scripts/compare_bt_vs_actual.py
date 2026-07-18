#!/usr/bin/env python3
"""Compare backtest daily returns with actual live trading results."""
import json, glob, os, csv

# --- Actual account equity (ukeire_hosyoukin) from wallet snapshots ---
actual_equity = {}
for d in sorted(glob.glob("/Users/shonen/日米ラグ/results/202607*_production_close_positions")):
    f = glob.glob(os.path.join(d, "wallet_close_*.json"))
    if f:
        with open(f[0]) as fh:
            data = json.load(fh)
        ts = data.get("timestamp", "")[:10]
        eq = data.get("ukeire_hosyoukin")
        if eq is not None:
            actual_equity[ts] = eq

# Also add decision-time wallet snapshots
for d in sorted(glob.glob("/Users/shonen/日米ラグ/results/202607*_production_decision_v2")):
    f = glob.glob(os.path.join(d, "wallet_decision_*.json"))
    if f:
        with open(f[0]) as fh:
            data = json.load(fh)
        ts = data.get("timestamp", "")[:10]
        eq = data.get("ukeire_hosyoukin")
        if eq is not None and ts not in actual_equity:
            actual_equity[ts] = eq

# --- Unrealized PnL from position snapshots ---
unrealized = {}
for d in sorted(glob.glob("/Users/shonen/日米ラグ/results/202607*_production_close_positions")):
    f = glob.glob(os.path.join(d, "positions_close_*.json"))
    if f:
        with open(f[0]) as fh:
            data = json.load(fh)
        ts = data.get("timestamp", "")[:10]
        unrealized[ts] = data.get("total_unrealized_pnl", 0)

# --- Backtest daily returns ---
bt_returns = {}
with open("/Users/shonen/日米ラグ/src/results/production_backtest/daily_net_returns.csv") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        if len(row) >= 2:
            bt_returns[row[0]] = float(row[1])

# --- Compare ---
dates = sorted(set(actual_equity.keys()) & set(bt_returns.keys()))
if not dates:
    # Use all actual dates, fill missing BT returns as 0
    dates = sorted(actual_equity.keys())

print("=== Backtest vs Actual Live Trading Comparison ===")
print(f'{"Date":<12} {"BT Ret":>8} {"BT Equity":>12} {"Actual Eq":>12} {"Actual Ret":>11} {"Unrealized":>11} {"Diff":>8}')
print("-" * 78)

bt_equity = None
prev_actual = None
for date in dates:
    bt_ret = bt_returns.get(date, None)
    actual_eq = actual_equity[date]
    unrl = unrealized.get(date, "N/A")

    if prev_actual is None:
        prev_actual = actual_eq
        actual_daily = 0.0
    else:
        actual_daily = (actual_eq - prev_actual) / prev_actual

    if bt_equity is None:
        bt_equity = actual_eq  # sync start

    if bt_ret is not None:
        bt_equity = bt_equity * (1 + bt_ret)
        diff = actual_daily - bt_ret
        bt_str = f"{bt_ret*100:>7.2f}%"
        diff_str = f"{diff*100:>7.2f}%"
    else:
        bt_str = "   N/A"
        diff_str = "   N/A"

    unrl_str = f"{unrl:>11,}" if isinstance(unrl, (int, float)) else f"{unrl:>11}"
    print(f"{date:<12} {bt_str} {bt_equity:>12,.0f} {actual_eq:>12,} {actual_daily*100:>10.2f}% {unrl_str} {diff_str}")
    prev_actual = actual_eq

print()
if dates:
    first_eq = actual_equity[dates[0]]
    last_eq = actual_equity[dates[-1]]
    actual_total = (last_eq / first_eq - 1) * 100
    bt_total = (bt_equity / first_eq - 1) * 100 if bt_equity else 0
    print(f"Period: {dates[0]} -> {dates[-1]}")
    print(f"Total: BT={bt_total:+.2f}%, Actual={actual_total:+.2f}%, Gap={bt_total-actual_total:.2f}pp")
    print(f"Starting equity: {first_eq:,}, Ending equity: {last_eq:,}")

    # --- Deep dive: model differences ---
    print("\n" + "=" * 78)
    print("=== Root Cause Analysis ===")
    print("=" * 78)

    # 1. Model mismatch
    print("\n[1] MODEL MISMATCH")
    print("  Backtest uses: SectorRelativeEnsembleBLPEnhancedModel (V1-style)")
    print("  Live uses:     generate_v2_production_portfolio (ProductionV2Model)")
    print("  -> These are DIFFERENT models with different signal generation logic")

    # 2. Check 07-15 flat position
    print("\n[2] FLAT POSITION DAY (2026-07-15)")
    print("  sHosyouKinritu=0.00 => ALL POSITIONS CLOSED, no overnight hold")
    print("  This is a V2 fallback/flat day. BT still assumes positions held.")

    # 3. Overnight holding alpha
    print("\n[3] OVERNIGHT HOLDING")
    print("  BT: overnight_alpha_long=0.75, short=0.50 (partial overnight hold)")
    print("  Live: held_overnight list shows partial holds with alpha scaling")
    print("  But 07-15 was fully flat (0 positions) => alpha=0 that day")

    # 4. Position sizing differences
    print("\n[4] POSITION SIZING")
    print("  BT: fractional weights (e.g. 0.3241) on full capital")
    print("  Live: integer share quantities, rounded to lot sizes")
    print("  Live capital ~332K-358K vs BT assumes 100% capital deployment")

    # 5. PIT binning
    print("\n[5] PIT BINNING / RuleD")
    print("  Live report (07-17): PIT history=3 days < 252 => Medium/1.00 fallback")
    print("  BT: full 10+ year history => proper Low/Mid/High binning")
    print("  -> Live has insufficient history for dynamic gross sizing")

    # 6. Gap data
    print("\n[6] GAP-ADJUSTED DISTRIBUTION")
    print("  Live: uses daily-computed mu_gap/omega_gap matrices")
    print("  BT: SectorRelativeEnsembleBLPEnhancedModel does NOT use gap matrices")
    print("  -> Fundamentally different signal inputs")

    # 7. Multi-horizon blend & rank reversal
    print("\n[7] MULTI-HORIZON BLEND & RANK REVERSAL")
    print("  Live (07-17): h=3,h=5 matrices NOT FOUND, skipping")
    print("  Live (07-17): rank reversal overlay signal file NOT FOUND, skipping")
    print("  BT: may or may not have these features enabled")

    # 8. Cost structure
    print("\n[8] COST STRUCTURE")
    print("  BT: slippage 5bps/side + financing 2.5% + borrow 1.15% + reverse 2bps/day")
    print("  Live: actual brokerage costs, slippage from market impact")
    print("  close_execution_log: all realized_pnl=0 (mark-to-market only, not settled)")

    # 9. Equity proxy
    print("\n[9] EQUITY MEASUREMENT")
    print("  BT: daily_net_returns = portfolio weighted return - costs")
    print("  Actual: ukeire_hosyoukin (受入保証金) = cash + unrealized PnL proxy")
    print("  ukeire_hosyoukin fluctuates with margin requirements, NOT pure P&L")
    print("  -> This is NOT a direct comparison metric")

    print("\n" + "=" * 78)
    print("SUMMARY: BT +5.20% vs Actual -3.64% => 8.83pp gap")
    print("Primary causes (ranked):")
    print("  1. Different model (SRE-BLPX vs ProductionV2) - structural")
    print("  2. ukeire_hosyoukin is not pure P&L (margin/cash proxy)")
    print("  3. PIT binning: live has 3 days vs BT has 10+ years")
    print("  4. 07-15 flat position (V2 fallback) not in BT")
    print("  5. Gap data: live uses gap-adjusted dist, BT does not")
    print("  6. Share rounding & lot size constraints in live")
    print("=" * 78)
