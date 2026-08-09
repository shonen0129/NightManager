#!/usr/bin/env python3
"""Verify overlay model loading and metadata consistency.

Loads each saved MLOrderOverlayModel and compares its attributes with the
on-disk metadata.json (in particular p_trade_scale). Exits with 1 on mismatch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from leadlag.models.ml_order_overlay import load_overlay_model

MODEL_DIRS = [
    ROOT / "models" / "ml_order_overlay" / "phase2_8",
    ROOT / "models" / "ml_order_overlay" / "phase2_13",
    ROOT / "models" / "ml_order_overlay" / "phase2_13_reg",
]
WF_BASE = ROOT / "models" / "ml_order_overlay" / "phase2_13_reg_wf"


def verify_dir(model_dir: Path) -> bool:
    model_path = model_dir / "model.pkl"
    meta_path = model_dir / "metadata.json"
    if not model_path.exists():
        print(f"[skip] {model_dir}: model.pkl not found")
        return True

    model = load_overlay_model(model_dir)
    p_trade_scale = float(getattr(model, "p_trade_scale", 1.0))

    if not meta_path.exists():
        print(f"[fail] {model_dir}: metadata.json missing but model exists")
        return False

    with open(meta_path) as f:
        meta = json.load(f)

    ok = True
    if meta.get("p_trade_scale") != p_trade_scale:
        print(
            f"[fail] {model_dir}: p_trade_scale mismatch "
            f"(pkl={p_trade_scale}, json={meta.get('p_trade_scale')})"
        )
        ok = False

    if meta.get("use_ticker") != model.use_ticker:
        ok = False
    if meta.get("use_classification") != model.use_classification:
        ok = False
    if meta.get("per_ticker_interactions") != model.per_ticker_interactions:
        ok = False
    if meta.get("target_std") != model.target_std:
        ok = False

    if ok:
        print(f"[ok] {model_dir}: p_trade_scale={p_trade_scale} (metadata matches model.pkl)")
    return ok


def main() -> int:
    all_ok = True
    for d in MODEL_DIRS:
        if d.exists():
            all_ok &= verify_dir(d)

    if WF_BASE.exists():
        for fold in sorted(WF_BASE.iterdir()):
            if fold.is_dir():
                all_ok &= verify_dir(fold)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
