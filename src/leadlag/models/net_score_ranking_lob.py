from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Any

from leadlag.execution.order_book_schema import OrderBookSnapshot
from leadlag.execution.slippage_model import compute_entry_cost_bps, compute_exit_cost_bps, CostSource
from leadlag.execution.execution_constraints import apply_hard_rules, replace_unavailable_short, ExecutionDecision

logger = logging.getLogger(__name__)

class NetScoreRankingLob:
    """Net Score Ranking Model with LOB Overlay.

    Ranks tickers by net score after deducting trading costs (spread, slippage, financing, borrow, reverse fee),
    selects top candidate long/shorts, applies LOB filters (spread cap, slippage cap, size-to-depth constraints),
    replaces unavailable short names, and outputs execution suggestions.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.exec_config = config.get("execution", {})
        self.n_long = self.exec_config.get("n_long", 5)
        self.n_short = self.exec_config.get("n_short", 5)
        self.min_net_score = self.exec_config.get("min_net_score", 0.0)
        self.aum = config.get("aum_jpy", 1000000)

    def run_selection(
        self,
        tickers: list[str],
        signals: dict[str, float],         # Ticker -> raw signal (mu_t)
        volatilities: dict[str, float],    # Ticker -> daily volatility (if needed, else 1.0)
        snapshots: dict[str, OrderBookSnapshot], # Ticker -> OrderBookSnapshot (or stub)
        short_available_dict: dict[str, bool],   # Ticker -> True/False
        reverse_fee_bps_dict: dict[str, float],  # Ticker -> bps_per_day
        adv_jpy_dict: dict[str, float],          # Ticker -> ADV in JPY
        gross_target: float = 1.0
    ) -> pd.DataFrame:
        """Executes the net score ranking with LOB overlay on a set of tickers for a single day.

        Returns:
            pd.DataFrame: Contains all detailed decision variables per ticker.
        """
        cost_opt = self.config.get("cost_aware_optimization", {})
        fallback_spread = cost_opt.get("default_spread_fallback_roundtrip_bps", 15.0)
        buy_interest = cost_opt.get("buy_interest_rate_annual", 0.025)
        borrow_fee = cost_opt.get("stock_borrow_fee_annual", 0.0115)
        
        # Calculate daily financing & borrow rate in decimal
        financing_rate_daily = buy_interest / 365.0
        borrow_fee_daily = borrow_fee / 365.0

        records = []
        for ticker in tickers:
            signal = signals.get(ticker, 0.0)
            snapshot = snapshots.get(ticker)
            vol = volatilities.get(ticker, 0.01)

            # We estimate entry cost assuming the base case position size (1/N of AUM)
            base_position_jpy = (self.aum * gross_target) / (2.0 * max(1, self.n_long))
            
            # Entry cost
            entry_cost_long_bps, cost_src_long = compute_entry_cost_bps(
                snapshot, base_position_jpy, "BUY", fallback_spread, self.config
            )
            entry_cost_short_bps, cost_src_short = compute_entry_cost_bps(
                snapshot, base_position_jpy, "SELL", fallback_spread, self.config
            )

            # Exit cost (close is always fixed spread)
            exit_cost_long_bps, _ = compute_exit_cost_bps(
                snapshot, base_position_jpy, "SELL", fallback_spread, self.config
            )
            exit_cost_short_bps, _ = compute_exit_cost_bps(
                snapshot, base_position_jpy, "BUY", fallback_spread, self.config
            )

            # Financing rates
            financing_bps = financing_rate_daily * 10000.0
            borrow_bps = borrow_fee_daily * 10000.0
            reverse_bps = reverse_fee_bps_dict.get(ticker, 0.0)

            # Net mu in decimals
            entry_cost_long = entry_cost_long_bps / 10000.0
            exit_cost_long = exit_cost_long_bps / 10000.0
            financing_cost = financing_bps / 10000.0

            entry_cost_short = entry_cost_short_bps / 10000.0
            exit_cost_short = exit_cost_short_bps / 10000.0
            borrow_cost = borrow_bps / 10000.0
            reverse_cost = reverse_bps / 10000.0

            net_mu_long = signal - (entry_cost_long + exit_cost_long + financing_cost)
            net_mu_short = -signal - (entry_cost_short + exit_cost_short + borrow_cost + reverse_cost)

            # Net score is net_mu
            score_long = net_mu_long
            score_short = net_mu_short

            records.append({
                "ticker": ticker,
                "signal": signal,
                "vol": vol,
                "score_long": score_long,
                "score_short": score_short,
                "net_mu_long": net_mu_long,
                "net_mu_short": net_mu_short,
                "entry_cost_long_bps": entry_cost_long_bps,
                "entry_cost_short_bps": entry_cost_short_bps,
                "exit_cost_long_bps": exit_cost_long_bps,
                "exit_cost_short_bps": exit_cost_short_bps,
                "financing_bps": financing_bps,
                "borrow_bps": borrow_bps,
                "reverse_bps": reverse_bps,
                "cost_source": cost_src_long.value if hasattr(cost_src_long, "value") else str(cost_src_long)
            })

        df = pd.DataFrame(records)

        # 1. Separate long/short candidates
        long_candidates = df[(df["signal"] > 0) & (df["score_long"] > self.min_net_score)].copy()
        short_candidates = df[(df["signal"] < 0) & (df["score_short"] > self.min_net_score)].copy()

        # Sort and select initial candidates (up to N_long, N_short)
        initial_longs = []
        if not long_candidates.empty:
            long_candidates = long_candidates.sort_values(by="score_long", ascending=False)
            initial_longs = long_candidates.head(self.n_long)["ticker"].tolist()

        initial_shorts = []
        if not short_candidates.empty:
            short_candidates = short_candidates.sort_values(by="score_short", ascending=False)
            initial_shorts = short_candidates.head(self.n_short)["ticker"].tolist()

        # Build initial weights before LOB overlay
        df["selected_before_lob"] = False
        df["weight_before_lob"] = 0.0

        if initial_longs:
            long_df = df[df["ticker"].isin(initial_longs)]
            # Weight is proportional to score_long
            scores = long_df["score_long"].values
            score_sum = scores.sum() if scores.sum() > 0 else 1.0
            weights = (gross_target / 2.0) * (scores / score_sum)
            df.loc[df["ticker"].isin(initial_longs), "weight_before_lob"] = weights
            df.loc[df["ticker"].isin(initial_longs), "selected_before_lob"] = True

        if initial_shorts:
            short_df = df[df["ticker"].isin(initial_shorts)]
            scores = short_df["score_short"].values
            score_sum = scores.sum() if scores.sum() > 0 else 1.0
            weights = -(gross_target / 2.0) * (scores / score_sum)
            df.loc[df["ticker"].isin(initial_shorts), "weight_before_lob"] = weights
            df.loc[df["ticker"].isin(initial_shorts), "selected_before_lob"] = True

        # Restrict individual names to ADV cap (e.g. 20% ADV)
        adv_cap = self.exec_config.get("adv_cap", 0.20)
        weight_cap = np.zeros(len(df))
        for idx, row in df.iterrows():
            tk = row["ticker"]
            adv = adv_jpy_dict.get(tk, 0.0)
            last_p = snapshots.get(tk).last_price if snapshots.get(tk) else None
            # If ADV is missing or 0, cap is standard AUM-based
            if adv > 0 and last_p is not None and last_p > 0:
                cap_jpy = adv * adv_cap
                weight_cap[idx] = cap_jpy / self.aum
            else:
                weight_cap[idx] = 0.25 # Default single stock cap (25% of AUM)

        # Apply standard single stock cap constraint on initial weights
        df["weight_before_lob"] = np.sign(df["weight_before_lob"]) * np.minimum(
            np.abs(df["weight_before_lob"]), np.maximum(weight_cap, 0.05)
        )
        
        # 2. Apply LOB overlay constraints & short replacements
        # Build lists of all valid candidates (reserve lists for replacements)
        reserve_longs = [t for t in long_candidates["ticker"].tolist() if t not in initial_longs]
        
        # Available shorts pool (all borrowable short candidates, ordered by score)
        available_shorts_pool = []
        if not short_candidates.empty:
            for _, row in short_candidates.iterrows():
                tk = row["ticker"]
                if short_available_dict.get(tk, True):
                    available_shorts_pool.append(tk)

        # Step 2a: Short Replacements (if initial short is unavailable)
        final_shorts = replace_unavailable_short(initial_shorts, available_shorts_pool, self.n_short)

        # Step 2b: Long Replacements (if initial long is unavailable or fails hard rules)
        # We process longs: if a long fails LOB hard rules, we try to replace it with a reserve long
        final_longs = []
        for ticker in initial_longs:
            snapshot = snapshots.get(ticker)
            est_weight = df.loc[df["ticker"] == ticker, "weight_before_lob"].values[0]
            order_jpy = abs(est_weight) * self.aum
            
            decision = apply_hard_rules(
                snapshot, "BUY", order_jpy, True, 0.0, self.config
            )
            
            if decision.selected:
                final_longs.append(ticker)
            else:
                # Try to replace
                replaced = False
                while reserve_longs:
                    candidate = reserve_longs.pop(0)
                    cand_snap = snapshots.get(candidate)
                    cand_score = df.loc[df["ticker"] == candidate, "score_long"].values[0]
                    # Estimate weight
                    cand_est_weight = (gross_target / 2.0) * (cand_score / (df.loc[df["ticker"].isin(initial_longs), "score_long"].sum() or 1.0))
                    cand_order = cand_est_weight * self.aum
                    cand_decision = apply_hard_rules(cand_snap, "BUY", cand_order, True, 0.0, self.config)
                    if cand_decision.selected:
                        final_longs.append(candidate)
                        replaced = True
                        break
                if not replaced:
                    # Could not replace, keep it but it will be skipped
                    pass

        # Similarly, check LOB rules on final shorts
        final_shorts_checked = []
        reserve_shorts_pool = [t for t in available_shorts_pool if t not in final_shorts]
        for ticker in final_shorts:
            snapshot = snapshots.get(ticker)
            est_weight = df.loc[df["ticker"] == ticker, "weight_before_lob"].values[0]
            order_jpy = abs(est_weight) * self.aum
            is_avail = short_available_dict.get(ticker, True)
            rev_fee = reverse_fee_bps_dict.get(ticker, 0.0)

            decision = apply_hard_rules(
                snapshot, "SELL", order_jpy, is_avail, rev_fee, self.config
            )
            
            if decision.selected:
                final_shorts_checked.append(ticker)
            else:
                # Try to replace
                replaced = False
                while reserve_shorts_pool:
                    candidate = reserve_shorts_pool.pop(0)
                    cand_snap = snapshots.get(candidate)
                    cand_score = df.loc[df["ticker"] == candidate, "score_short"].values[0]
                    cand_est_weight = (gross_target / 2.0) * (cand_score / (df.loc[df["ticker"].isin(initial_shorts), "score_short"].sum() or 1.0))
                    cand_order = cand_est_weight * self.aum
                    cand_decision = apply_hard_rules(cand_snap, "SELL", cand_order, True, reverse_fee_bps_dict.get(candidate, 0.0), self.config)
                    if cand_decision.selected:
                        final_shorts_checked.append(candidate)
                        replaced = True
                        break
                if not replaced:
                    pass

        # 3. Compute final weights with skips, replacements, and depth scaling
        df["selected_after_lob"] = False
        df["weight_after_lob"] = 0.0
        df["skip_reason"] = None
        df["scale_reason"] = None
        df["scale_factor"] = 1.0
        df["quoted_spread_bps"] = None
        df["estimated_slippage_bps"] = None
        df["order_depth_ratio"] = None

        # Helper to apply scaling and constraints to selected names
        for ticker in final_longs:
            snapshot = snapshots.get(ticker)
            score = df.loc[df["ticker"] == ticker, "score_long"].values[0]
            
            # Target weight allocation
            df.loc[df["ticker"] == ticker, "selected_after_lob"] = True
            raw_w = (gross_target / 2.0) * (score / (df.loc[df["ticker"].isin(final_longs), "score_long"].sum() or 1.0))
            order_jpy = raw_w * self.aum
            
            decision = apply_hard_rules(snapshot, "BUY", order_jpy, True, 0.0, self.config)
            
            df.loc[df["ticker"] == ticker, "weight_after_lob"] = raw_w * decision.scale_factor
            df.loc[df["ticker"] == ticker, "scale_factor"] = decision.scale_factor
            df.loc[df["ticker"] == ticker, "scale_reason"] = decision.scale_reason
            df.loc[df["ticker"] == ticker, "quoted_spread_bps"] = decision.quoted_spread_bps
            df.loc[df["ticker"] == ticker, "estimated_slippage_bps"] = decision.estimated_slippage_bps
            df.loc[df["ticker"] == ticker, "order_depth_ratio"] = decision.order_depth_ratio

        for ticker in final_shorts_checked:
            snapshot = snapshots.get(ticker)
            score = df.loc[df["ticker"] == ticker, "score_short"].values[0]
            is_avail = short_available_dict.get(ticker, True)
            rev_fee = reverse_fee_bps_dict.get(ticker, 0.0)

            df.loc[df["ticker"] == ticker, "selected_after_lob"] = True
            raw_w = -(gross_target / 2.0) * (score / (df.loc[df["ticker"].isin(final_shorts_checked), "score_short"].sum() or 1.0))
            order_jpy = abs(raw_w) * self.aum
            
            decision = apply_hard_rules(snapshot, "SELL", order_jpy, is_avail, rev_fee, self.config)
            
            df.loc[df["ticker"] == ticker, "weight_after_lob"] = raw_w * decision.scale_factor
            df.loc[df["ticker"] == ticker, "scale_factor"] = decision.scale_factor
            df.loc[df["ticker"] == ticker, "scale_reason"] = decision.scale_reason
            df.loc[df["ticker"] == ticker, "quoted_spread_bps"] = decision.quoted_spread_bps
            df.loc[df["ticker"] == ticker, "estimated_slippage_bps"] = decision.estimated_slippage_bps
            df.loc[df["ticker"] == ticker, "order_depth_ratio"] = decision.order_depth_ratio

        # Record skip reasons for names selected before LOB but excluded after LOB
        for idx, row in df.iterrows():
            tk = row["ticker"]
            if row["selected_before_lob"] and not row["selected_after_lob"]:
                # Determine skip reason
                snapshot = snapshots.get(tk)
                is_avail = short_available_dict.get(tk, True)
                rev_fee = reverse_fee_bps_dict.get(tk, 0.0)
                side = "BUY" if row["weight_before_lob"] > 0 else "SELL"
                ord_jpy = abs(row["weight_before_lob"]) * self.aum
                decision = apply_hard_rules(snapshot, side, ord_jpy, is_avail, rev_fee, self.config)
                df.loc[idx, "skip_reason"] = decision.skip_reason or "REPLACED_BY_BETTER_CANDIDATE"

        # Apply standard single stock cap constraint on final weights
        df["weight_after_lob"] = np.sign(df["weight_after_lob"]) * np.minimum(
            np.abs(df["weight_after_lob"]), np.maximum(weight_cap, 0.05)
        )

        # Restore dollar neutrality for the final portfolio
        final_w = df["weight_after_lob"].values
        from leadlag.models.net_score_ranking_lob import restore_dollar_neutrality_array
        df["weight_after_lob"] = restore_dollar_neutrality_array(final_w)

        return df


def restore_dollar_neutrality_array(w: np.ndarray) -> np.ndarray:
    w_new = w.copy()
    long_mask = w_new > 0.0
    short_mask = w_new < 0.0
    
    long_sum = np.sum(w_new[long_mask])
    short_sum = np.abs(np.sum(w_new[short_mask]))
    
    if long_sum == 0.0 or short_sum == 0.0:
        return np.zeros_like(w_new)
        
    target_gross = min(long_sum, short_sum)
    w_new[long_mask] = w_new[long_mask] * (target_gross / long_sum)
    w_new[short_mask] = w_new[short_mask] * (target_gross / short_sum)
    return w_new
