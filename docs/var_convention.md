# Consolidated Runtime Output Directory (`var/`)

## Goal

Unify runtime outputs that were previously scattered across the repository root
into a single `var/` tree. This makes the project easier to clean, archive, and
deploy.

## Directory Layout

```
var/
  outputs/      # generated backtest / research outputs
  results/      # backtest result bundles and diffs
  logs/         # runtime logs
  shadow_runs/  # shadow/live comparison runs
  live/         # daily production live data
  artifacts/    # large artifacts (parquet, npz, plots)
  experiments/  # experiment registry and temporary outputs
```

## Migration

Run the migration helper to move old root-level directories and leave symlinks
for backward compatibility::

    python3 tools/migrate_outputs_to_var.py --dry-run
    python3 tools/migrate_outputs_to_var.py

## Configuration

New code should default output paths under ``var/``. The canonical location for
a script is controlled by ``AppConfig.output_base_dir``; for new development
prefer ``var/results/<experiment>`` over ``results/<experiment>``.
