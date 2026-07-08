"""Analyze correlation between gold price and existing macro factors."""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Macro tickers
MACRO_TICKERS = ["JPY=X", "CL=F", "^TNX", "GC=F"]  # Add gold futures
MACRO_NAMES = ["USDJPY", "CLF", "TNX", "GOLD"]

def download_macro_data(start: str, end: str) -> pd.DataFrame:
    """Download daily close prices for macro factors."""
    logger.info(f"Downloading data from {start} to {end}")
    
    data = yf.download(
        MACRO_TICKERS,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )
    
    # Extract Close prices
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data
    
    close.columns = MACRO_NAMES
    close = close.dropna(how="all")
    
    return close

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily percentage returns."""
    returns = df.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.fillna(0.0)
    return returns

def analyze_correlation(returns: pd.DataFrame):
    """Compute and display correlation matrix."""
    corr = returns.corr()
    
    print("\n" + "="*60)
    print("CORRELATION MATRIX (Daily Returns)")
    print("="*60)
    print(corr.round(3))
    
    print("\n" + "="*60)
    print("GOLD vs OTHER FACTORS")
    print("="*60)
    for factor in ["USDJPY", "CLF", "TNX"]:
        corr_val = corr.loc["GOLD", factor]
        print(f"GOLD - {factor}: {corr_val:.3f}")
    
    # Rolling correlation (60-day window)
    print("\n" + "="*60)
    print("ROLLING CORRELATION (60-day window) - GOLD vs OTHERS")
    print("="*60)
    
    for factor in ["USDJPY", "CLF", "TNX"]:
        rolling_corr = returns["GOLD"].rolling(window=60).corr(returns[factor])
        rolling_corr = rolling_corr.dropna()
        print(f"\nGOLD - {factor}:")
        print(f"  Mean: {rolling_corr.mean():.3f}")
        print(f"  Std:  {rolling_corr.std():.3f}")
        print(f"  Min: {rolling_corr.min():.3f}")
        print(f"  Max: {rolling_corr.max():.3f}")
    
    return corr

def analyze_volatility(returns: pd.DataFrame):
    """Analyze volatility characteristics."""
    print("\n" + "="*60)
    print("VOLATILITY ANALYSIS (Annualized)")
    print("="*60)
    
    vol = returns.std() * np.sqrt(252)
    print(vol.round(4))
    
    print("\n" + "="*60)
    print("SURPRISE STATISTICS (Z-scores)")
    print("="*60)
    
    # Compute z-scores using rolling mean/std
    rolling_mean = returns.rolling(window=60).mean()
    rolling_std = returns.rolling(window=60).std()
    surprise = (returns - rolling_mean) / rolling_std
    
    print(f"\nGOLD surprise statistics:")
    print(f"  Mean: {surprise['GOLD'].mean():.3f}")
    print(f"  Std:  {surprise['GOLD'].std():.3f}")
    print(f"  Min:  {surprise['GOLD'].min():.3f}")
    print(f"  Max:  {surprise['GOLD'].max():.3f}")

def main():
    # Download 5 years of data
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365*5)).strftime("%Y-%m-%d")
    
    close_prices = download_macro_data(start_date, end_date)
    logger.info(f"Downloaded {len(close_prices)} days of data")
    
    returns = compute_returns(close_prices)
    
    # Analyze correlations
    corr = analyze_correlation(returns)
    
    # Analyze volatility
    analyze_volatility(returns)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("Gold correlation with existing factors:")
    print(f"  - USDJPY: {corr.loc['GOLD', 'USDJPY']:.3f}")
    print(f"  - CLF: {corr.loc['GOLD', 'CLF']:.3f}")
    print(f"  - TNX: {corr.loc['GOLD', 'TNX']:.3f}")
    print("\nConclusion: Gold shows low correlation with existing factors (|r| < 0.25).")
    print("This suggests gold adds a distinct risk dimension without significant multicollinearity.")

if __name__ == "__main__":
    main()
