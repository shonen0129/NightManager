from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/research"))

from leadlag.data.intraday_inputs import build_open_910_returns, compute_jp_target_returns
from leadlag.data.tickers import JP_TICKERS
from leadlag.execution.config import load_config_from_yaml
from leadlag.models.ml_overlay_features import _precompute_market_vol
from research.experiments.ml_overlay_training import _collect_training_data


def main() -> None:
    cached = pd.read_pickle(ROOT / "var/live/pipeline_data/cache/preprocessed/20260728/preprocessed_data.pkl")
    frame = cached["df_exec"] if isinstance(cached, dict) else cached
    frame = frame.loc["2024-01-01":"2024-12-31"]
    gap_dir = ROOT / "var/results/beta_shift_comparison/gap_old/20260811_151355"
    dates = pd.DatetimeIndex(sorted(
        pd.Timestamp(path.stem.removeprefix("mu_gap_"))
        for path in (gap_dir / "matrices").glob("mu_gap_*.npy")
        if path.stem.removeprefix("mu_gap_")[:4] == "2024"
    )[:10])
    dates = dates.intersection(frame.index)
    open_910 = build_open_910_returns(frame, JP_TICKERS)
    target = compute_jp_target_returns(frame, JP_TICKERS, open_910_returns=open_910)
    cfg = load_config_from_yaml(ROOT / "configs/production/production.yaml", strict=True).v2
    cfg = cfg.model_copy(update={"gap_input_dir": gap_dir})
    train = _collect_training_data(
        dates,
        frame,
        target,
        gap_dir,
        cfg,
        _precompute_market_vol(frame),
        open_910_returns=open_910,
    )
    print({"dates": len(dates), "rows": len(train), "unique_dates": int(train["ticker"].count() / len(JP_TICKERS)) if not train.empty else 0})


if __name__ == "__main__":
    main()
