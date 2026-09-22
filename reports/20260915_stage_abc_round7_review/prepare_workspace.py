"""Prepare isolated review artifacts; never overwrite prior review evidence."""
from pathlib import Path

OUT = Path(__file__).resolve().parent
PRIOR = OUT.parent / "20260914_stage_abc_round6_review"
source = (PRIOR / "capture_manifest.py").read_text()
source = source.replace("Round 5", "Round 6").replace("20260914_stage_abc_round5_review", "20260914_stage_abc_round6_review").replace("changed_since_round5", "changed_since_round6")
(OUT / "capture_manifest.py").write_text(source)
for name in ("probe_review.py", "probe_schema_cache.py", "probe_var_gap_version.py", "probe_contracts.py", "probe_boundaries.py"):
    (OUT / name).write_bytes((PRIOR / name).read_bytes())
print("Prepared Round 7 report scripts")
