import functools
import sys
from pathlib import Path

# Ensure project `src` is on sys.path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import production

if __name__ == "__main__":
    # Patch ProductionConfig factory to force topix_beta_coef=0 (disable Proposal B)
    production.ProductionConfig = functools.partial(
        production.ProductionConfig,
        topix_beta_coef=0.0,
    )

    out_dir = production.run_production(
        start_date="2015-01-01",
        output_root=production.get_default_results_root(),
        run_tag="baseline_no_topix",
        skip_chart=True,
    )
    print(out_dir)
