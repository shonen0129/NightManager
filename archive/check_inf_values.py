import pandas as pd
import numpy as np

# Load panels
returns_panel = pd.read_parquet("artifacts/sprint0/returns_panel.parquet")
residual_returns_panel = pd.read_parquet("artifacts/sprint0/residual_returns_panel.parquet")
signal_diagnostics = pd.read_parquet("artifacts/sprint0/signal_diagnostics.parquet")

print("=== Returns Panel Null/Inf counts ===")
for col in returns_panel.columns:
    vals = returns_panel[col]
    nulls = vals.isna().sum()
    infs = np.isinf(vals).sum()
    print(f"{col}: NaNs={nulls}, Infs={infs}, Min={vals.min()}, Max={vals.max()}")

print("\n=== Residual Returns Panel Null/Inf counts ===")
for col in residual_returns_panel.columns:
    vals = residual_returns_panel[col]
    nulls = vals.isna().sum()
    infs = np.isinf(vals).sum()
    print(f"{col}: NaNs={nulls}, Infs={infs}, Min={vals.min()}, Max={vals.max()}")

print("\n=== Signal Diagnostics Panel Null/Inf counts ===")
for col in signal_diagnostics.columns:
    vals = signal_diagnostics[col]
    nulls = vals.isna().sum()
    infs = np.isinf(vals).sum()
    print(f"{col}: NaNs={nulls}, Infs={infs}, Min={vals.min()}, Max={vals.max()}")
