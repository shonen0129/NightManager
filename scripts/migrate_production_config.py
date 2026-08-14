#!/usr/bin/env python
"""Migrate nested production.yaml to flat schema."""

from __future__ import annotations

import argparse

import yaml

from leadlag.config.schemas import _map_flat_to_nested


def flatten(src: dict) -> dict:
    """Flatten nested YAML into a dict suitable for ``load_config_from_yaml``."""
    v2 = _map_flat_to_nested(src)
    blpx = v2.pop("blpx", {}) or {}
    costs = v2.pop("costs", {}) or {}

    flat = dict(v2)

    # BLPX-specific fields become blpx_* top-level keys unless they already
    # exist as top-level V2 aliases (e.g. macro_kappas, minvar_enabled).
    for k, v in blpx.items():
        if k not in flat:
            flat[f"blpx_{k}"] = v

    # Cost/finance fields stay flat (no prefix) unless already present.
    for k, v in costs.items():
        if k not in flat:
            flat[k] = v

    # Preserve AppConfig-level sections that are not part of ProductionV2RunConfig.
    for section in ["model", "risk", "output", "broker"]:
        if section in src:
            flat[section] = src[section]

    if "start_date" in src:
        flat["start_date"] = src["start_date"]

    return flat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        src = yaml.safe_load(f)

    flat = flatten(src)

    with open(args.output, "w", encoding="utf-8") as f:
        yaml.safe_dump(flat, f, sort_keys=False, allow_unicode=True)

    print(f"Migrated {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
