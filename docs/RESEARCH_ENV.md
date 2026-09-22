# Research environment

`src/research/` contains training and experiment entry points. It is not part
of the production `leadlag` wheel. Create the research environment from the
repository root with the `research` extra:

```bash
uv sync --extra research
```

The extra installs dependencies; it does not add `research` to the production
wheel. Run research entry points from this checkout with `PYTHONPATH=src`.

The production runtime keeps the `nonlinear` extra because LightGBM is needed
to load and run an already trained overlay. Training remains an explicit
research operation:

```bash
PYTHONPATH=src .venv/bin/python reports/20260912_workspace_audit/watchdog.py 7200 \
  .venv/bin/python tools/production/train_ml_order_overlay.py \
  --train-start 2015-01-05 \
  --train-end 2024-12-31 \
  --gap-input-dir var/results/gap_adjusted_distribution/<run> \
  --output-dir models/ml_order_overlay/<version>
```

Training artifacts are versioned and must contain verified provenance. The
production CLI only loads the active `CURRENT` version and does not import the
research package at startup.

The current phase 1/2 backtest drivers pass their overlay through
`BacktestEngine.run_v2_backtest(decision_transform=...)`. The transformed result
is audited before use; patching the deleted portfolio generator is unsupported.
