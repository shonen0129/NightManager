"""Create independent Round 8 evidence; never modify prior review artifacts."""
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PREVIOUS = ROOT / "reports/20260915_stage_abc_round7_review"
for name in ("probe_review.py", "probe_schema_cache.py", "probe_var_gap_version.py", "probe_contracts.py", "probe_boundaries.py", "probe_deadline.py", "probe_provenance.py"):
    target = OUT / name
    if not target.exists():
        target.write_bytes((PREVIOUS / name).read_bytes())
tracked = subprocess.check_output(["git", "diff", "--name-only", "-z"], cwd=ROOT).decode().split("\0")
untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
paths = sorted({p for p in tracked + untracked if p and not p.startswith("reports/")})
hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths if (ROOT / p).is_file()}
result = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(), "source_target_sha256": hashes}
for filename in ("source_snapshot.json", "review_manifest.json"):
    old = json.loads((PREVIOUS / filename).read_text())["source_target_sha256"]
    result["changed_since_round7_" + filename.removesuffix(".json")] = sorted(p for p in set(old) | set(hashes) if old.get(p) != hashes.get(p))
target = OUT / "source_snapshot.json"
if target.exists():
    old = json.loads(target.read_text())["source_target_sha256"]
    result["source_changes_during_review"] = sorted(p for p in set(old) | set(hashes) if old.get(p) != hashes.get(p))
    target = OUT / "review_manifest.json"
target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "source_target_sha256"}, ensure_ascii=False, indent=2))
