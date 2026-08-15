"""Check environment dependencies for redesign prototype."""

import sys

def main() -> None:
    print(f"Python version: {sys.version}")
    packages = ["numpy", "pandas", "scipy", "pydantic", "polars", "cvxpy", "osqp", "httpx", "pytest"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [OK] {pkg}: {ver}")
        except ImportError:
            print(f"  [MISSING] {pkg}")

if __name__ == "__main__":
    main()
