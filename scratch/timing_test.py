"""Quick timing test for a single backtest run."""
import sys, time
sys.path.insert(0, "src")

print("step1: imports...", flush=True)
from leadlag.data.cache import load_df_exec_from_local_cache
from leadlag.models.sector_relative_ensemble_blp_enhanced import SectorRelativeEnsembleBLPEnhancedModel
from leadlag.execution.backtester import BacktestEngine
import yaml, numpy as np

print("step2: data load...", flush=True)
df = load_df_exec_from_local_cache()
print(f"  shape={df.shape}", flush=True)

print("step3: model build...", flush=True)
with open("configs/production.yaml") as f:
    cfg = yaml.safe_load(f)
model = SectorRelativeEnsembleBLPEnhancedModel(cfg)

print("step4: predict_signals...", flush=True)
t0 = time.perf_counter()
pred = model.predict_signals(df)
t1 = time.perf_counter()
print(f"  predict_signals: {t1-t0:.1f}s", flush=True)

print("step5: run_backtest...", flush=True)
t2 = time.perf_counter()
results = BacktestEngine.run_backtest(model, df_exec=df, start_date="2015-01-01",
    overnight_alpha_long=0.75, overnight_alpha_short=0.5,
    buy_interest_annual=0.025, borrow_fee_annual=0.0115, reverse_fee_bps=2.0, slippage_bps=5.0)
t3 = time.perf_counter()
print(f"  run_backtest: {t3-t2:.1f}s (total: {t3-t0:.1f}s)", flush=True)

dr = results["daily_returns"]
ar = float(dr.mean() * 245)
vol = float(dr.std(ddof=1) * np.sqrt(245))
sharpe = ar / vol if vol > 0 else np.nan
wealth = (1.0 + dr).cumprod()
mdd = float(((wealth / wealth.cummax()) - 1.0).min())
print(f"Sharpe={sharpe:.4f} AR={ar*100:.2f}% MDD={mdd*100:.2f}%", flush=True)
