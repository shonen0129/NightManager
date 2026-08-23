#!/usr/bin/env python3
"""Quick test for yfinance shares outstanding fetch."""
import yfinance as yf

for tk in ["7203.T", "6758.T", "9432.T", "9984.T"]:
    try:
        t = yf.Ticker(tk)
        info = t.info
        print(tk, "sharesOutstanding:", info.get("sharesOutstanding"), "shares:", info.get("shares"))
    except Exception as e:
        print(tk, "ERROR:", e)
