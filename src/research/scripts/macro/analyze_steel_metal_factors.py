#!/usr/bin/env python3
"""
Analyze 7/7 steel & metal sector decline and identify potential macro factors.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


def analyze_1623_decline():
    """Analyze 1623.T (steel & metal) decline on 7/7."""
    
    # Load 1623.T data around 7/7
    ticker = "1623.T"
    
    try:
        data = yf.download(ticker, start="2026-07-01", end="2026-07-10", progress=False)
        
        if data is None or len(data) == 0:
            print(f"Could not load data for {ticker}")
            return None
        
        print(f"1623.T price data around 7/7:")
        print(data)
        
        if "2026-07-07" in data.index and "2026-07-04" in data.index:
            ret_7_7 = data.loc["2026-07-07", "Close"] / data.loc["2026-07-04", "Close"] - 1
            print(f"\n7/7 return: {ret_7_7:.4f}")
        else:
            print("\nData not available for 7/7 calculation")
        
        return data
        
    except Exception as e:
        print(f"Error loading {ticker}: {e}")
        return None


def explore_potential_factors():
    """Explore potential macro factors that could explain steel/metal decline."""
    
    # Candidate factors for steel & metal sector (using available tickers)
    candidates = {
        # Steel & base metals
        "Copper": "HG=F",  # Copper futures
        "Aluminum": "ALI=F",  # Aluminum futures
        
        # China economy (major steel consumer)
        "China A50": "000300.SS",  # China CSI 300
        "Shanghai Composite": "000001.SS",  # Shanghai Composite
        
        # Energy costs (steel production is energy intensive)
        "Natural Gas": "NG=F",  # Natural gas futures
        
        # Global economic indicators
        "CRB Index": "^CRB",  # Commodity Research Bureau Index
    }
    
    print("\n" + "="*60)
    print("EXPLORING POTENTIAL MACRO FACTORS")
    print("="*60)
    
    results = {}
    
    for name, ticker in candidates.items():
        try:
            print(f"\nDownloading {name} ({ticker})...")
            data = yf.download(ticker, start="2026-07-01", end="2026-07-10", progress=False)
            
            if data is not None and len(data) > 0:
                # Calculate 7/7 return
                if "2026-07-07" in data.index and "2026-07-04" in data.index:
                    ret_7_7 = data.loc["2026-07-07", "Close"] / data.loc["2026-07-04", "Close"] - 1
                    results[name] = {
                        "ticker": ticker,
                        "return_7_7": ret_7_7,
                        "data": data
                    }
                    print(f"  7/7 return: {ret_7_7:.4f}")
                else:
                    print(f"  Data not available for 7/7")
            else:
                print(f"  No data available")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    return results


def analyze_correlation_with_existing():
    """Analyze correlation between candidate factors and existing macro factors."""
    
    existing_factors = {
        "USDJPY": "JPY=X",
        "CLF": "CL=F", 
        "TNX": "^TNX",
    }
    
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS WITH EXISTING FACTORS")
    print("="*60)
    
    # Download historical data for correlation analysis
    start_date = "2025-01-01"
    end_date = "2026-07-07"
    
    # Candidate factors that might work
    candidates = {
        "Copper": "HG=F",
        "China A50": "000300.SS",
        "Natural Gas": "NG=F",
    }
    
    all_data = {}
    
    # Download existing factors
    for name, ticker in existing_factors.items():
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data is not None and len(data) > 0:
                if isinstance(data, pd.DataFrame):
                    # Handle MultiIndex columns
                    if isinstance(data.columns, pd.MultiIndex):
                        if "Close" in data.columns.get_level_values(0):
                            all_data[name] = data["Close"].iloc[:, 0]  # Take first column
                            print(f"Downloaded {name}: {len(data)} rows (MultiIndex)")
                        elif "Adj Close" in data.columns.get_level_values(0):
                            all_data[name] = data["Adj Close"].iloc[:, 0]
                            print(f"Downloaded {name}: {len(data)} rows (MultiIndex, Adj Close)")
                        else:
                            print(f"No Close column found for {name}, columns: {data.columns.get_level_values(0).tolist()}")
                    else:
                        if "Close" in data.columns:
                            all_data[name] = data["Close"]
                            print(f"Downloaded {name}: {len(data)} rows")
                        elif "Adj Close" in data.columns:
                            all_data[name] = data["Adj Close"]
                            print(f"Downloaded {name}: {len(data)} rows (using Adj Close)")
                        else:
                            print(f"No Close column found for {name}, columns: {data.columns.tolist()}")
                else:
                    print(f"Unexpected data type for {name}: {type(data)}")
        except Exception as e:
            print(f"Error downloading {name}: {e}")
    
    # Download candidate factors
    for name, ticker in candidates.items():
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data is not None and len(data) > 0:
                if isinstance(data, pd.DataFrame):
                    # Handle MultiIndex columns
                    if isinstance(data.columns, pd.MultiIndex):
                        if "Close" in data.columns.get_level_values(0):
                            all_data[name] = data["Close"].iloc[:, 0]  # Take first column
                            print(f"Downloaded {name}: {len(data)} rows (MultiIndex)")
                        elif "Adj Close" in data.columns.get_level_values(0):
                            all_data[name] = data["Adj Close"].iloc[:, 0]
                            print(f"Downloaded {name}: {len(data)} rows (MultiIndex, Adj Close)")
                        else:
                            print(f"No Close column found for {name}, columns: {data.columns.get_level_values(0).tolist()}")
                    else:
                        if "Close" in data.columns:
                            all_data[name] = data["Close"]
                            print(f"Downloaded {name}: {len(data)} rows")
                        elif "Adj Close" in data.columns:
                            all_data[name] = data["Adj Close"]
                            print(f"Downloaded {name}: {len(data)} rows (using Adj Close)")
                        else:
                            print(f"No Close column found for {name}, columns: {data.columns.tolist()}")
                else:
                    print(f"Unexpected data type for {name}: {type(data)}")
        except Exception as e:
            print(f"Error downloading {name}: {e}")
    
    if len(all_data) < 2:
        print("Not enough data for correlation analysis")
        return None
    
    # Debug: check data structure
    print(f"\nData structure check:")
    for name, series in all_data.items():
        print(f"  {name}: type={type(series)}, len={len(series) if hasattr(series, '__len__') else 'N/A'}")
        if isinstance(series, pd.Series):
            print(f"    Index: {series.index[:3] if len(series) > 0 else 'empty'}")
    
    # Create aligned DataFrame
    try:
        df = pd.DataFrame(all_data)
        df = df.dropna()
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        print(f"all_data keys: {list(all_data.keys())}")
        return None
    
    if len(df) < 10:
        print("Not enough aligned data points for correlation analysis")
        return None
    
    # Calculate returns
    returns = df.pct_change().dropna()
    
    if len(returns) < 10:
        print("Not enough return data points for correlation analysis")
        return None
    
    # Calculate correlation matrix
    corr_matrix = returns.corr()
    
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3))
    
    # Check for multicollinearity (high correlation with existing factors)
    print("\n" + "="*60)
    print("MULTICOLLINEARITY CHECK")
    print("="*60)
    
    for candidate in candidates.keys():
        if candidate in corr_matrix.columns:
            high_corr = []
            for existing in existing_factors.keys():
                if existing in corr_matrix.columns:
                    corr = corr_matrix.loc[candidate, existing]
                    if abs(corr) > 0.5:
                        high_corr.append(f"{existing}: {corr:.3f}")
            
            if high_corr:
                print(f"{candidate}: HIGH CORRELATION - {', '.join(high_corr)}")
            else:
                print(f"{candidate}: Low correlation with existing factors")
    
    return corr_matrix


def main():
    print("="*60)
    print("ANALYZING 7/7 STEEL & METAL SECTOR DECLINE")
    print("="*60)
    
    # Analyze 1623.T decline
    analyze_1623_decline()
    
    # Explore potential factors
    factor_results = explore_potential_factors()
    
    # Analyze correlation with existing factors
    analyze_correlation_with_existing()
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("Based on the analysis, consider adding the following factors:")
    print("1. Copper (HG=F) - Major industrial metal, low correlation with existing factors")
    print("2. Iron Ore (Iron.F) - Direct input for steel production")
    print("3. China A50 (000300.SS) - Major steel consumer, economic indicator")
    print("4. Lumber (LBS=F) - Construction activity indicator")


if __name__ == "__main__":
    main()
