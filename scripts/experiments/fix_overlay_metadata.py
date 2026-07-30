#!/usr/bin/env python3
"""Regenerate metadata.json for existing overlay models to include p_trade_scale.

Reads p_trade_scale from each model.pkl and rewrites metadata.json so that the
on-disk metadata matches the pickled MLOrderOverlayModel attributes.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from leadlag.data.tickers import JP_TICKERS
from leadlag.models.ml_order_overlay import MLOrderOverlayModel

BASE_DIR = ROOT / "models" / "ml_order_overlay"


def fix_one(model_dir: Path) -> None:
    model_path = model_dir / "model.pkl"
    meta_path = model_dir / "metadata.json"
    if not model_path.exists():
        print(f"[skip] {model_dir.name}: model.pkl not found")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    if not isinstance(model, MLOrderOverlayModel):
        print(f"[skip] {model_dir.name}: not an MLOrderOverlayModel ({type(model)})")
        return

    p_trade_scale = float(getattr(model, "p_trade_scale", 1.0))

    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata["cont_cols"] = model.cont_cols
    metadata["target_std"] = float(model.target_std)
    metadata["use_ticker"] = model.use_ticker
    metadata["use_classification"] = model.use_classification
    metadata["per_ticker_interactions"] = model.per_ticker_interactions
    metadata["n_tickers"] = len(JP_TICKERS)
    metadata["p_trade_scale"] = p_trade_scale

    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"[ok] {model_dir.name}: p_trade_scale={p_trade_scale} -> {meta_path}")


def main() -> int:
    if not BASE_DIR.exists():
        print(f"[error] {BASE_DIR} does not exist")
        return 1

    for sub in sorted(BASE_DIR.iterdir()):
        if not sub.is_dir():
            continue
        # If the directory itself has a model.pkl, fix it directly.
        if (sub / "model.pkl").exists():
            fix_one(sub)
        # Otherwise, look for nested fold_* subdirectories (walk-forward layout).
        else:
            for fold in sorted(sub.iterdir()):
                if fold.is_dir() and (fold / "model.pkl").exists():
                    fix_one(fold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
