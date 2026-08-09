import sys

try:
    import lightgbm as lgb
    print("LightGBM version:", lgb.__version__)
except Exception as e:
    print("LightGBM not available:", e, file=sys.stderr)
    sys.exit(1)
