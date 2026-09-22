"""Train a verified overlay artifact from the available local snapshot."""

from __future__ import annotations

import signal

from leadlag.config.paths import project_root
from leadlag.data.market_data_cache import load_df_exec_from_local_cache
from leadlag.execution.config import load_config_from_yaml
from research.experiments.ml_overlay_training import train_overlay_model


def main() -> None:
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("training timeout")))
    signal.alarm(240)
    root = project_root()
    config = load_config_from_yaml(root / "configs/production/production.yaml", strict=True)
    df_exec = load_df_exec_from_local_cache()
    train_overlay_model(
        df_exec=df_exec,
        gap_input_dir=root / "var/live/pipeline_data/gap_adjusted_distribution/gap_store.sqlite",
        run_cfg=config.v2,
        train_start="2026-01-05",
        train_end="2026-08-13",
        output_dir=root / "var/results/20260920_structural_completion/ml_overlay_retrained",
        lgbm_kwargs={
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 3,
            "num_leaves": 20,
            "min_child_samples": 300,
            "reg_alpha": 0.5,
            "reg_lambda": 1.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        },
        use_ticker=True,
        per_ticker_interactions=True,
        target_type="raw",
    )


if __name__ == "__main__":
    main()
