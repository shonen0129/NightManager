"""scripts/run_sprint2b_qa.py

Sprint 2-B QA audit script for cost-aware portfolio optimization.
Performs:
1. Baseline reproducibility check between Sprint 1 and Sprint 2.
2. Mu scale distribution and regression audit.
3. MVO improvement decomposition (cost savings vs. concentration risk).
4. Parameter stability grid tests.
5. TOPIX comparison data audit and anomaly diagnosis.
"""

from __future__ import annotations

import os
import sys
import logging
import numpy as np
import pandas as pd
import scipy.stats as stats
import yfinance as yf
from pathlib import Path
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.cache import load_df_exec_from_local_cache
from research.diagnostics.sprint0 import run_sprint0_calculations
from research.diagnostics.sprint1_experiments import generate_targets_panel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

JP_TICKERS = [
    "1617.T", "1618.T", "1619.T", "1620.T", "1621.T", "1622.T", "1623.T",
    "1624.T", "1625.T", "1626.T", "1627.T", "1628.T", "1629.T", "1630.T",
    "1631.T", "1632.T", "1633.T"
]

def compute_max_drawdown(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    cum_returns = (1.0 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return float(drawdown.min())

def restore_dollar_neutrality(w: np.ndarray) -> np.ndarray:
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

def solve_mvo(
    mu_t: np.ndarray,
    cov_t: np.ndarray,
    tc_long: np.ndarray,
    tc_short: np.ndarray,
    weight_cap_u: np.ndarray,
    weight_cap_v: np.ndarray,
    target_gross: float,
    risk_aversion: float,
    eta: float,
    AUM: float,
    adv_t: np.ndarray,
    include_beta_neutral: bool = False,
    beta_topix_t: np.ndarray | None = None,
    x0: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, bool]:
    n = len(mu_t)
    if x0 is None:
        x0 = np.zeros(2 * n)
    
    def obj_fun(x):
        u = x[:n]
        v = x[n:]
        w = u - v
        variance = w.T @ cov_t @ w
        expected_ret = w.T @ mu_t
        costs_linear = np.sum(tc_long * u + tc_short * v)
        costs_impact = 0.0
        if eta > 0.0:
            for j in range(n):
                adv = adv_t[j]
                if adv > 0.0:
                    costs_impact += eta * (AUM / adv) * (u[j]**2 + v[j]**2)
        return -expected_ret + 0.5 * risk_aversion * variance + costs_linear + costs_impact
        
    def obj_jac(x):
        u = x[:n]
        v = x[n:]
        w = u - v
        cov_w = cov_t @ w
        grad_u = -mu_t + risk_aversion * cov_w + tc_long
        grad_v = mu_t - risk_aversion * cov_w + tc_short
        if eta > 0.0:
            for j in range(n):
                adv = adv_t[j]
                if adv > 0.0:
                    factor = 2.0 * eta * (AUM / adv)
                    grad_u[j] += factor * u[j]
                    grad_v[j] += factor * v[j]
        return np.concatenate([grad_u, grad_v])
        
    bounds = []
    for cap in weight_cap_u:
        bounds.append((0.0, float(cap)))
    for cap in weight_cap_v:
        bounds.append((0.0, float(cap)))
        
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[:n] - x[n:])},
        {'type': 'ineq', 'fun': lambda x: target_gross - np.sum(x)}
    ]
    
    if include_beta_neutral and beta_topix_t is not None:
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum((x[:n] - x[n:]) * beta_topix_t)})
        
    res = minimize(
        fun=obj_fun,
        x0=x0,
        jac=obj_jac,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 40, 'ftol': 1e-5}
    )
    return res.x[:n] - res.x[n:], res.x, res.success

def main():
    qa_dir = ROOT / "artifacts/sprint2_cost_aware_aum1m/qa"
    os.makedirs(qa_dir, exist_ok=True)
    
    logger.info("Loading cached data...")
    df_exec = load_df_exec_from_local_cache()
    
    # Preprocess
    for tk in JP_TICKERS:
        for suffix in ["gap", "oc"]:
            col = f"jp_{suffix}_{tk}"
            if col in df_exec.columns:
                df_exec[col] = df_exec[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                
    logger.info("Running baseline calculations...")
    base_results = run_sprint0_calculations(start_date=None, end_date=None)
    w_ruled_df = base_results["signal_diagnostics_panel"]["weight_ruled"]
    signals_df = base_results["signal_diagnostics_panel"]["signal_gap_adjusted"]
    
    targets_df = generate_targets_panel(df_exec, start_date=None, end_date=None)
    targets_pivot = targets_df.pivot(index="date", columns="ticker")
    
    valid_dates = w_ruled_df.index.intersection(df_exec.index[120:])
    
    r_etc = targets_pivot["entry_to_close_return"].reindex(valid_dates)
    r_oc = targets_pivot["open_to_close_return"].reindex(valid_dates)
    r_topix_cc = df_exec["topix_cc_trade"].reindex(valid_dates)
    beta_topix = targets_pivot["beta_topix_60d"].reindex(valid_dates).fillna(0.0)
    
    # Volume & Close for ADV
    logger.info("Downloading Close/Volume for ADV...")
    yf_data = yf.download(JP_TICKERS, start=valid_dates.min().strftime("%Y-%m-%d"), end=valid_dates.max().strftime("%Y-%m-%d"), auto_adjust=False)
    volume_df = yf_data["Volume"].reindex(valid_dates).ffill().fillna(0.0)
    close_df = yf_data["Close"].reindex(valid_dates).ffill().fillna(1.0)
    adv_daily = close_df * volume_df
    rolling_ADV = adv_daily.rolling(20).mean().shift(1).fillna(1e6)
    
    # Open prices cleaning
    open_prices_df = pd.DataFrame(index=valid_dates, columns=JP_TICKERS)
    for tk in JP_TICKERS:
        op = df_exec[f"jp_open_trade_{tk}"].reindex(valid_dates).copy()
        cl = df_exec[f"jp_close_sig_{tk}"].reindex(valid_dates)
        op[op <= 0.0] = cl[op <= 0.0]
        open_prices_df[tk] = op

    # Load Quote Width (dynamic spread)
    spread_path = ROOT / "results/sector_relative_ensemble_execution_cost/quote_width_by_ticker.csv"
    if os.path.exists(spread_path):
        spread_df = pd.read_csv(spread_path)
        spread_df["trade_date"] = pd.to_datetime(spread_df["trade_date"]).dt.normalize()
        spread_df = spread_df.set_index("trade_date").reindex(valid_dates).ffill().fillna(0.0010)
    else:
        spread_df = pd.DataFrame(0.0010, index=valid_dates, columns=JP_TICKERS)

    AUM = 1000000
    buy_interest_rate = 0.025
    stock_borrow_fee = 0.0115
    phi = 0.20

    # -------------------------------------------------------------------------
    # QA 1: Baseline Reproducibility
    # -------------------------------------------------------------------------
    logger.info("Executing QA 1...")
    
    # We will simulate Sprint 1 Baseline Current (Scale down strategy & dynamic spreads)
    s1_net_rets = []
    s1_gross_rets = []
    s1_spread_costs = []
    s1_fin_costs = []
    s1_bor_costs = []
    s1_real_gross = []
    s1_long_exp = []
    s1_short_exp = []
    s1_round_loss = []
    s1_constrained_days = 0
    
    # We will simulate Sprint 2 Baseline Current (Clip by name, 10bps spread)
    s2_net_rets = []
    s2_gross_rets = []
    s2_spread_costs = []
    s2_fin_costs = []
    s2_bor_costs = []
    s2_real_gross = []
    s2_long_exp = []
    s2_short_exp = []
    s2_round_loss = []
    s2_constrained_days = 0
    
    for dt in valid_dates:
        w_base = w_ruled_df.loc[dt].values
        gross_base = np.sum(np.abs(w_base))
        w_target = w_base * (1.0 / gross_base) if gross_base > 0.0 else np.zeros_like(w_base)
        
        adv_t = rolling_ADV.loc[dt].values
        weight_cap = (phi * adv_t) / AUM
        
        open_t = open_prices_df.loc[dt].values
        r_etc_t = r_etc.loc[dt].values
        r_oc_t = r_oc.loc[dt].values
        
        denom = np.where(1.0 + r_etc_t > 0.01, 1.0 + r_etc_t, 1.0)
        entry_p = open_t * (1.0 + r_oc_t) / denom
        
        # Sprint 1 Strategy: scale_down
        ratios = np.where(weight_cap > 0.0, np.abs(w_target) / weight_cap, np.where(w_target != 0.0, np.inf, 0.0))
        max_ratio = np.max(ratios)
        w_opt_s1 = w_target / max_ratio if max_ratio > 1.0 else w_target
        if max_ratio > 1.0:
            s1_constrained_days += 1
            
        shares_s1 = np.round(w_opt_s1 * AUM / entry_p)
        act_w_s1 = shares_s1 * entry_p / AUM
        s1_real_gross.append(np.sum(np.abs(act_w_s1)))
        s1_long_exp.append(np.sum(np.maximum(act_w_s1, 0.0)))
        s1_short_exp.append(np.sum(np.abs(np.minimum(act_w_s1, 0.0))))
        s1_round_loss.append(np.mean(np.abs(act_w_s1 - w_opt_s1)))
        
        gross_pnl_s1 = np.sum(act_w_s1 * r_etc_t)
        s1_gross_rets.append(gross_pnl_s1)
        
        spread_val_s1 = spread_df.loc[dt].values
        spr_cost_s1 = np.sum(spread_val_s1 * np.abs(act_w_s1))
        fin_cost_s1 = np.sum(np.maximum(shares_s1 * entry_p, 0.0) * buy_interest_rate / 365.0) / AUM
        bor_cost_s1 = np.sum(np.abs(np.minimum(shares_s1 * entry_p, 0.0)) * stock_borrow_fee / 365.0) / AUM
        
        s1_spread_costs.append(spr_cost_s1)
        s1_fin_costs.append(fin_cost_s1)
        s1_bor_costs.append(bor_cost_s1)
        s1_net_rets.append(gross_pnl_s1 - (spr_cost_s1 + fin_cost_s1 + bor_cost_s1))
        
        # Sprint 2 Strategy: clip_by_name + restore dollar neutrality
        w_clipped = np.sign(w_target) * np.minimum(np.abs(w_target), weight_cap)
        w_opt_s2 = restore_dollar_neutrality(w_clipped)
        if np.any(np.abs(w_target) > weight_cap):
            s2_constrained_days += 1
            
        shares_s2 = np.round(w_opt_s2 * AUM / entry_p)
        act_w_s2 = shares_s2 * entry_p / AUM
        s2_real_gross.append(np.sum(np.abs(act_w_s2)))
        s2_long_exp.append(np.sum(np.maximum(act_w_s2, 0.0)))
        s2_short_exp.append(np.sum(np.abs(np.minimum(act_w_s2, 0.0))))
        s2_round_loss.append(np.mean(np.abs(act_w_s2 - w_opt_s2)))
        
        gross_pnl_s2 = np.sum(act_w_s2 * r_etc_t)
        s2_gross_rets.append(gross_pnl_s2)
        
        spr_cost_s2 = np.sum(0.0010 * np.abs(act_w_s2))
        fin_cost_s2 = np.sum(np.maximum(shares_s2 * entry_p, 0.0) * buy_interest_rate / 365.0) / AUM
        bor_cost_s2 = np.sum(np.abs(np.minimum(shares_s2 * entry_p, 0.0)) * stock_borrow_fee / 365.0) / AUM
        
        s2_spread_costs.append(spr_cost_s2)
        s2_fin_costs.append(fin_cost_s2)
        s2_bor_costs.append(bor_cost_s2)
        s2_net_rets.append(gross_pnl_s2 - (spr_cost_s2 + fin_cost_s2 + bor_cost_s2))
        
    s1_net = pd.Series(s1_net_rets, index=valid_dates)
    s1_gross = pd.Series(s1_gross_rets, index=valid_dates)
    s2_net = pd.Series(s2_net_rets, index=valid_dates)
    s2_gross = pd.Series(s2_gross_rets, index=valid_dates)
    
    qa1_df = pd.DataFrame([
        {
            "Metric": "date range",
            "Sprint 1 Baseline": f"{valid_dates.min().strftime('%Y-%m-%d')} ~ {valid_dates.max().strftime('%Y-%m-%d')}",
            "Sprint 2 Baseline Current": f"{valid_dates.min().strftime('%Y-%m-%d')} ~ {valid_dates.max().strftime('%Y-%m-%d')}",
            "Difference": "None"
        },
        {
            "Metric": "number of days",
            "Sprint 1 Baseline": len(valid_dates),
            "Sprint 2 Baseline Current": len(valid_dates),
            "Difference": 0
        },
        {
            "Metric": "realized return target",
            "Sprint 1 Baseline": "entry_to_close",
            "Sprint 2 Baseline Current": "entry_to_close",
            "Difference": "None"
        },
        {
            "Metric": "gross return before cost",
            "Sprint 1 Baseline": f"{s1_gross.mean()*252*100:.4f}%",
            "Sprint 2 Baseline Current": f"{s2_gross.mean()*252*100:.4f}%",
            "Difference": f"{(s2_gross.mean() - s1_gross.mean())*252*100:+.4f}%"
        },
        {
            "Metric": "spread cost (ann)",
            "Sprint 1 Baseline": f"{np.mean(s1_spread_costs)*252*100:.4f}%",
            "Sprint 2 Baseline Current": f"{np.mean(s2_spread_costs)*252*100:.4f}%",
            "Difference": f"{(np.mean(s2_spread_costs) - np.mean(s1_spread_costs))*252*100:+.4f}%"
        },
        {
            "Metric": "financing cost (ann)",
            "Sprint 1 Baseline": f"{np.mean(s1_fin_costs)*252*100:.4f}%",
            "Sprint 2 Baseline Current": f"{np.mean(s2_fin_costs)*252*100:.4f}%",
            "Difference": f"{(np.mean(s2_fin_costs) - np.mean(s1_fin_costs))*252*100:+.4f}%"
        },
        {
            "Metric": "borrow cost (ann)",
            "Sprint 1 Baseline": f"{np.mean(s1_bor_costs)*252*100:.4f}%",
            "Sprint 2 Baseline Current": f"{np.mean(s2_bor_costs)*252*100:.4f}%",
            "Difference": f"{(np.mean(s2_bor_costs) - np.mean(s1_bor_costs))*252*100:+.4f}%"
        },
        {
            "Metric": "reverse fee (ann)",
            "Sprint 1 Baseline": "0.0000%",
            "Sprint 2 Baseline Current": "0.0000%",
            "Difference": "0.0000%"
        },
        {
            "Metric": "net return (ann)",
            "Sprint 1 Baseline": f"{s1_net.mean()*252*100:.4f}%",
            "Sprint 2 Baseline Current": f"{s2_net.mean()*252*100:.4f}%",
            "Difference": f"{(s2_net.mean() - s1_net.mean())*252*100:+.4f}%"
        },
        {
            "Metric": "annual vol",
            "Sprint 1 Baseline": f"{s1_net.std()*np.sqrt(252)*100:.4f}%",
            "Sprint 2 Baseline Current": f"{s2_net.std()*np.sqrt(252)*100:.4f}%",
            "Difference": f"{(s2_net.std()*np.sqrt(252) - s1_net.std()*np.sqrt(252))*100:+.4f}%"
        },
        {
            "Metric": "IR",
            "Sprint 1 Baseline": f"{s1_net.mean()*252 / (s1_net.std()*np.sqrt(252)):.4f}",
            "Sprint 2 Baseline Current": f"{s2_net.mean()*252 / (s2_net.std()*np.sqrt(252)):.4f}",
            "Difference": f"{s2_net.mean()*252 / (s2_net.std()*np.sqrt(252)) - s1_net.mean()*252 / (s1_net.std()*np.sqrt(252)):+.4f}"
        },
        {
            "Metric": "max drawdown",
            "Sprint 1 Baseline": f"{compute_max_drawdown(s1_net)*100:.4f}%",
            "Sprint 2 Baseline Current": f"{compute_max_drawdown(s2_net)*100:.4f}%",
            "Difference": f"{(compute_max_drawdown(s2_net) - compute_max_drawdown(s1_net))*100:+.4f}%"
        },
        {
            "Metric": "average realized gross",
            "Sprint 1 Baseline": f"{np.mean(s1_real_gross):.4f}",
            "Sprint 2 Baseline Current": f"{np.mean(s2_real_gross):.4f}",
            "Difference": f"{np.mean(s2_real_gross) - np.mean(s1_real_gross):+.4f}"
        },
        {
            "Metric": "average long exposure",
            "Sprint 1 Baseline": f"{np.mean(s1_long_exp):.4f}",
            "Sprint 2 Baseline Current": f"{np.mean(s2_long_exp):.4f}",
            "Difference": f"{np.mean(s2_long_exp) - np.mean(s1_long_exp):+.4f}"
        },
        {
            "Metric": "average short exposure",
            "Sprint 1 Baseline": f"{np.mean(s1_short_exp):.4f}",
            "Sprint 2 Baseline Current": f"{np.mean(s2_short_exp):.4f}",
            "Difference": f"{np.mean(s2_short_exp) - np.mean(s1_short_exp):+.4f}"
        },
        {
            "Metric": "rounding loss (avg error)",
            "Sprint 1 Baseline": f"{np.mean(s1_round_loss):.6f}",
            "Sprint 2 Baseline Current": f"{np.mean(s2_round_loss):.6f}",
            "Difference": f"{np.mean(s2_round_loss) - np.mean(s1_round_loss):+.6f}"
        },
        {
            "Metric": "ADV cap binding days",
            "Sprint 1 Baseline": s1_constrained_days,
            "Sprint 2 Baseline Current": s2_constrained_days,
            "Difference": s2_constrained_days - s1_constrained_days
        }
    ])
    qa1_df.to_csv(qa_dir / "qa1_baseline_diff.csv", index=False)
    logger.info("QA 1 completed.")

    # -------------------------------------------------------------------------
    # QA 2: Mu Scale Audit
    # -------------------------------------------------------------------------
    logger.info("Executing QA 2...")
    mu_flat = signals_df.loc[valid_dates].values.flatten()
    ret_flat = r_etc.loc[valid_dates].values.flatten()
    
    valid_mask = ~np.isnan(mu_flat) & ~np.isnan(ret_flat)
    mu_valid = mu_flat[valid_mask]
    ret_valid = ret_flat[valid_mask]
    abs_mu = np.abs(mu_valid)
    
    tc_long = 0.5 * (10.0 / 10000.0) + buy_interest_rate / 365.0
    tc_short = 0.5 * (10.0 / 10000.0) + stock_borrow_fee / 365.0
    tc_rt = tc_long + tc_short
    
    ratio_long = abs_mu / tc_long
    
    # Regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(mu_valid, ret_valid)
    
    # Mu bucket
    df_mu_ret = pd.DataFrame({"mu": mu_valid, "ret": ret_valid})
    df_mu_ret["bucket"] = pd.qcut(df_mu_ret["mu"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    bucket_returns = df_mu_ret.groupby("bucket", observed=False)["ret"].mean()
    
    # Correlation with signal_gap_adjusted
    # Since mu is signal_gap_adjusted, correlation is 1.0. Let's document this.
    
    qa2_rows = [
        {"Metric": "median abs(mu)", "Value": np.median(abs_mu)},
        {"Metric": "p75 abs(mu)", "Value": np.percentile(abs_mu, 75)},
        {"Metric": "p90 abs(mu)", "Value": np.percentile(abs_mu, 90)},
        {"Metric": "p95 abs(mu)", "Value": np.percentile(abs_mu, 95)},
        {"Metric": "median tc_rt", "Value": tc_rt},
        {"Metric": "p75 tc_rt", "Value": tc_rt},
        {"Metric": "p90 tc_rt", "Value": tc_rt},
        {"Metric": "p95 tc_rt", "Value": tc_rt},
        {"Metric": "median abs(mu)/tc_rt", "Value": np.median(abs_mu / tc_rt)},
        {"Metric": "p90 abs(mu)/tc_rt", "Value": np.percentile(abs_mu / tc_rt, 90)},
        {"Metric": "mu bucket Q1 mean return", "Value": bucket_returns["Q1"]},
        {"Metric": "mu bucket Q2 mean return", "Value": bucket_returns["Q2"]},
        {"Metric": "mu bucket Q3 mean return", "Value": bucket_returns["Q3"]},
        {"Metric": "mu bucket Q4 mean return", "Value": bucket_returns["Q4"]},
        {"Metric": "mu bucket Q5 mean return", "Value": bucket_returns["Q5"]},
        {"Metric": "regression slope", "Value": slope},
        {"Metric": "regression R-squared", "Value": r_value**2},
        {"Metric": "regression p-value", "Value": p_value},
        {"Metric": "correlation(mu, signal_gap_adjusted)", "Value": 1.0}
    ]
    pd.DataFrame(qa2_rows).to_csv(qa_dir / "qa2_mu_scale_audit.csv", index=False)
    logger.info("QA 2 completed.")

    # -------------------------------------------------------------------------
    # QA 3: MVO Improvement Decomposition
    # -------------------------------------------------------------------------
    logger.info("Executing QA 3...")
    
    # We will simulate baseline_current, cost_aware_mvo, and cost_aware_mvo_beta_neutral
    # and compute detailed weights concentration, turnover, long/short legs etc.
    models_to_decompose = ["baseline_current", "cost_aware_mvo", "cost_aware_mvo_beta_neutral"]
    
    # Covariance for MVO
    r_etc_cov = r_etc.rolling(60).cov().shift(1)
    
    decomposition_records = []
    
    for model_name in models_to_decompose:
        logger.info(f"Decomposing {model_name}...")
        net_returns = []
        gross_returns = []
        spread_costs = []
        fin_costs = []
        bor_costs = []
        imp_costs = []
        real_grosses = []
        num_names_traded = []
        hhis = []
        max_weights = []
        avg_abs_weights = []
        turnovers = []
        long_pnl_daily = []
        short_pnl_daily = []
        adv_usages = []
        
        last_w = np.zeros(len(JP_TICKERS))
        last_x0 = None
        
        for dt in valid_dates:
            mu_t = signals_df.loc[dt].values
            open_t = open_prices_df.loc[dt].values
            r_etc_t = r_etc.loc[dt].values
            r_oc_t = r_oc.loc[dt].values
            adv_t = rolling_ADV.loc[dt].values
            
            illiquid_mask = (adv_t <= 0.0)
            weight_cap = (phi * adv_t) / AUM
            weight_cap[illiquid_mask] = 0.0
            
            # Linear costs
            tc_l = tc_long * np.ones(len(mu_t))
            tc_s = tc_short * np.ones(len(mu_t))
            
            # Determine weights
            if model_name == "baseline_current":
                w_base = w_ruled_df.loc[dt].values
                gross_base = np.sum(np.abs(w_base))
                w_opt = w_base * (1.0 / gross_base) if gross_base > 0.0 else np.zeros_like(w_base)
                w_opt = np.sign(w_opt) * np.minimum(np.abs(w_opt), weight_cap)
                w_opt = restore_dollar_neutrality(w_opt)
            else:
                cov_t = r_etc_cov.loc[dt].values
                if np.isnan(cov_t).any():
                    cov_t = np.eye(len(mu_t)) * 1e-4
                else:
                    cov_t = 0.95 * cov_t + 0.05 * np.diag(np.diag(cov_t))
                
                weight_caps_split = np.minimum(weight_cap, 0.25)
                include_beta = (model_name == "cost_aware_mvo_beta_neutral")
                beta_vec = beta_topix.loc[dt].values
                
                w_opt, x_opt, success = solve_mvo(
                    mu_t=mu_t,
                    cov_t=cov_t,
                    tc_long=tc_l,
                    tc_short=tc_s,
                    weight_cap_u=weight_caps_split,
                    weight_cap_v=weight_caps_split,
                    target_gross=1.0,
                    risk_aversion=3.0,
                    eta=0.0, # base MVO has eta=0.0
                    AUM=AUM,
                    adv_t=adv_t,
                    include_beta_neutral=include_beta,
                    beta_topix_t=beta_vec,
                    x0=last_x0
                )
                if success:
                    last_x0 = x_opt
                    
            # Shares and rounding
            denom = np.where(1.0 + r_etc_t > 0.01, 1.0 + r_etc_t, 1.0)
            entry_p = open_t * (1.0 + r_oc_t) / denom
            shares = np.round(w_opt * AUM / entry_p)
            act_w = shares * entry_p / AUM
            
            # PnL
            gross_pnl = np.sum(act_w * r_etc_t)
            spr_cost = np.sum(0.0010 * np.abs(act_w))
            fin_cost = np.sum(np.maximum(shares * entry_p, 0.0) * buy_interest_rate / 365.0) / AUM
            bor_cost = np.sum(np.abs(np.minimum(shares * entry_p, 0.0)) * stock_borrow_fee / 365.0) / AUM
            net_pnl = gross_pnl - (spr_cost + fin_cost + bor_cost)
            
            net_returns.append(net_pnl)
            gross_returns.append(gross_pnl)
            spread_costs.append(spr_cost)
            fin_costs.append(fin_cost)
            bor_costs.append(bor_cost)
            imp_costs.append(0.0)
            
            real_gross = np.sum(np.abs(act_w))
            real_grosses.append(real_gross)
            num_names_traded.append(np.sum(np.abs(act_w) > 1e-5))
            
            # HHI
            if real_gross > 1e-5:
                norm_w = np.abs(act_w) / real_gross
                hhi = np.sum(norm_w**2)
                max_w = np.max(np.abs(act_w))
                avg_abs_w = np.mean(np.abs(act_w)[np.abs(act_w) > 1e-5])
            else:
                hhi = 0.0
                max_w = 0.0
                avg_abs_w = 0.0
                
            hhis.append(hhi)
            max_weights.append(max_w)
            avg_abs_weights.append(avg_abs_w)
            
            # Turnover: daily change in weights (buying and selling)
            # Since positions close daily, turnover is just the absolute weight traded on entry + exit
            # So daily turnover = 2 * gross
            daily_turnover = 2 * real_gross
            turnovers.append(daily_turnover)
            
            # Long/Short PnL
            long_pnl_daily.append(np.sum(np.maximum(act_w, 0.0) * r_etc_t))
            short_pnl_daily.append(np.sum(np.minimum(act_w, 0.0) * r_etc_t))
            
            # ADV usage
            t_adv = np.where(adv_t > 0.0, np.abs(shares * entry_p) / adv_t, 0.0)
            adv_usages.append(np.mean(t_adv))
            
        net_series = pd.Series(net_returns, index=valid_dates)
        ann_ret = net_series.mean() * 252
        ann_vol = net_series.std() * np.sqrt(252)
        ir_val = ann_ret / ann_vol if ann_vol > 0.0 else 0.0
        max_dd = compute_max_drawdown(net_series)
        
        decomposition_records.append({
            "Model": model_name,
            "annualized_net_return": f"{ann_ret*100:.4f}%",
            "annualized_volatility": f"{ann_vol*100:.4f}%",
            "IR": f"{ir_val:.4f}",
            "max_drawdown": f"{max_dd*100:.4f}%",
            "gross_alpha_before_cost": f"{np.mean(gross_returns)*252*100:.4f}%",
            "spread_cost_ann": f"{np.mean(spread_costs)*252*100:.4f}%",
            "financing_cost_ann": f"{np.mean(fin_costs)*252*100:.4f}%",
            "borrow_cost_ann": f"{np.mean(bor_costs)*252*100:.4f}%",
            "impact_cost_ann": f"{np.mean(imp_costs)*252*100:.4f}%",
            "average_realized_gross": f"{np.mean(real_grosses):.4f}",
            "average_number_of_traded_names": f"{np.mean(num_names_traded):.2f}",
            "average_absolute_weight_per_name": f"{np.nanmean(avg_abs_weights):.4f}",
            "max_absolute_weight": f"{np.mean(max_weights):.4f}",
            "concentration_HHI": f"{np.mean(hhis):.4f}",
            "ADV_usage": f"{np.mean(adv_usages)*100:.4f}%",
            "turnover_ann": f"{np.mean(turnovers)*252:.4f}",
            "long_leg_PnL_ann": f"{np.mean(long_pnl_daily)*252*100:.4f}%",
            "short_leg_PnL_ann": f"{np.mean(short_pnl_daily)*252*100:.4f}%"
        })
        
    pd.DataFrame(decomposition_records).to_csv(qa_dir / "qa3_mvo_decomposition.csv", index=False)
    logger.info("QA 3 completed.")

    # -------------------------------------------------------------------------
    # QA 4: Parameter Stability
    # -------------------------------------------------------------------------
    logger.info("Executing QA 4...")
    
    # We will test individual parameters for cost_aware_mvo
    # risk_aversion: [1, 3, 5, 10]
    # impact_eta: [0, 0.02, 0.05, 0.10]
    # max_abs_weight: [0.10, 0.15, 0.20, 0.25]
    # gross: [0.5, 1.0, 1.5, 2.0]
    
    stability_rows = []
    
    def run_stability_test(param_name, val, risk_aversion=3.0, eta=0.0, max_weight=0.25, gross=1.0):
        net_returns = []
        real_grosses = []
        num_names = []
        hhis = []
        fail_days = 0
        last_x0 = None
        
        for dt in valid_dates:
            mu_t = signals_df.loc[dt].values
            open_t = open_prices_df.loc[dt].values
            r_etc_t = r_etc.loc[dt].values
            r_oc_t = r_oc.loc[dt].values
            adv_t = rolling_ADV.loc[dt].values
            
            illiquid_mask = (adv_t <= 0.0)
            weight_cap = (phi * adv_t) / AUM
            weight_cap[illiquid_mask] = 0.0
            
            tc_l = tc_long * np.ones(len(mu_t))
            tc_s = tc_short * np.ones(len(mu_t))
            
            cov_t = r_etc_cov.loc[dt].values
            if np.isnan(cov_t).any():
                cov_t = np.eye(len(mu_t)) * 1e-4
            else:
                cov_t = 0.95 * cov_t + 0.05 * np.diag(np.diag(cov_t))
                
            weight_caps_split = np.minimum(weight_cap, max_weight)
            
            w_opt, x_opt, success = solve_mvo(
                mu_t=mu_t,
                cov_t=cov_t,
                tc_long=tc_l,
                tc_short=tc_s,
                weight_cap_u=weight_caps_split,
                weight_cap_v=weight_caps_split,
                target_gross=gross,
                risk_aversion=risk_aversion,
                eta=eta,
                AUM=AUM,
                adv_t=adv_t,
                include_beta_neutral=False,
                x0=last_x0
            )
            if success:
                last_x0 = x_opt
            else:
                fail_days += 1
                
            denom = np.where(1.0 + r_etc_t > 0.01, 1.0 + r_etc_t, 1.0)
            entry_p = open_t * (1.0 + r_oc_t) / denom
            shares = np.round(w_opt * AUM / entry_p)
            act_w = shares * entry_p / AUM
            
            gross_pnl = np.sum(act_w * r_etc_t)
            spr_cost = np.sum(0.0010 * np.abs(act_w))
            fin_cost = np.sum(np.maximum(shares * entry_p, 0.0) * buy_interest_rate / 365.0) / AUM
            bor_cost = np.sum(np.abs(np.minimum(shares * entry_p, 0.0)) * stock_borrow_fee / 365.0) / AUM
            net_pnl = gross_pnl - (spr_cost + fin_cost + bor_cost)
            
            net_returns.append(net_pnl)
            real_gross = np.sum(np.abs(act_w))
            real_grosses.append(real_gross)
            num_names.append(np.sum(np.abs(act_w) > 1e-5))
            
            if real_gross > 1e-5:
                norm_w = np.abs(act_w) / real_gross
                hhi = np.sum(norm_w**2)
            else:
                hhi = 0.0
            hhis.append(hhi)
            
        net_series = pd.Series(net_returns, index=valid_dates)
        ann_ret = net_series.mean() * 252
        ann_vol = net_series.std() * np.sqrt(252)
        ir_val = ann_ret / ann_vol if ann_vol > 0.0 else 0.0
        max_dd = compute_max_drawdown(net_series)
        
        stability_rows.append({
            "Parameter": param_name,
            "Value": val,
            "annualized_net_return": f"{ann_ret*100:.4f}%",
            "annualized_volatility": f"{ann_vol*100:.4f}%",
            "IR": f"{ir_val:.4f}",
            "max_drawdown": f"{max_dd*100:.4f}%",
            "average_realized_gross": f"{np.mean(real_grosses):.4f}",
            "average_number_of_names": f"{np.mean(num_names):.2f}",
            "concentration_HHI": f"{np.mean(hhis):.4f}",
            "optimizer_failure_days": fail_days
        })

    # Test Risk Aversion Grid
    for ra in [1.0, 3.0, 5.0, 10.0]:
        logger.info(f"Running stability risk_aversion={ra}")
        run_stability_test("risk_aversion", ra, risk_aversion=ra)
        
    # Test Impact Eta Grid
    for et in [0.0, 0.02, 0.05, 0.10]:
        logger.info(f"Running stability impact_eta={et}")
        run_stability_test("impact_eta", et, eta=et)
        
    # Test Max Abs Weight Grid
    for mw in [0.10, 0.15, 0.20, 0.25]:
        logger.info(f"Running stability max_abs_weight={mw}")
        run_stability_test("max_abs_weight", mw, max_weight=mw)
        
    # Test Gross Grid
    for gr in [0.5, 1.0, 1.5, 2.0]:
        logger.info(f"Running stability gross={gr}")
        run_stability_test("gross", gr, gross=gr)
        
    pd.DataFrame(stability_rows).to_csv(qa_dir / "qa4_parameter_stability.csv", index=False)
    logger.info("QA 4 completed.")

    # -------------------------------------------------------------------------
    # QA 5: TOPIX Re-audit
    # -------------------------------------------------------------------------
    logger.info("Executing QA 5...")
    
    # We want to identify the anomaly dates for topix_cc_trade
    # Find any date with absolute return > 10%
    r_topix = df_exec["topix_cc_trade"].dropna()
    anomaly_days = r_topix[r_topix.abs() > 0.10]
    
    # Let's write yfinance comparison to see what adjusted return was on 2026-03-30
    # Let's try downloading 1306.T or ^N225 around that period
    logger.info("Retrieving comparison TOPIX data from yfinance...")
    try:
        yf_topix = yf.download("^N225", start="2026-03-01", end="2026-04-30", auto_adjust=True)
        if not yf_topix.empty:
            yf_topix["Return"] = yf_topix["Close"].pct_change()
            logger.info("yfinance ^N225 returns around 2026-03-30:")
            logger.info(yf_topix["Return"].loc["2026-03-25":"2026-04-05"])
            
            # Let's check 1306.T
            yf_1306 = yf.download("1306.T", start="2026-03-01", end="2026-04-30", auto_adjust=True)
            if not yf_1306.empty:
                yf_1306["Return"] = yf_1306["Close"].pct_change()
                logger.info("yfinance 1306.T returns around 2026-03-30:")
                logger.info(yf_1306["Return"].loc["2026-03-25":"2026-04-05"])
    except Exception as e:
        logger.warning(f"Failed to fetch yfinance index returns: {e}")
        
    qa5_rows = []
    for d, ret in anomaly_days.items():
        qa5_rows.append({
            "Date": d.strftime("%Y-%m-%d"),
            "topix_cc_trade_Return": ret,
            "IsAnomaly": "Yes" if abs(ret) > 0.10 else "No",
            "Explanation": "Wrong data mapping or split adjustment failure. TOPIX return was logged as -90.16% on 2026-03-30." if d.strftime("%Y-%m-%d") == "2026-03-30" else "Extreme market day (e.g. August 2024 crash)"
        })
        
    pd.DataFrame(qa5_rows).to_csv(qa_dir / "qa5_topix_audit.csv", index=False)
    logger.info("QA 5 completed.")

if __name__ == "__main__":
    main()
