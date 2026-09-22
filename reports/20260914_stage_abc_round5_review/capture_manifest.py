"""Record the reviewed dirty tree without changing production files."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
previous = json.loads((ROOT / "reports/20260914_stage_abc_round4_review/review_manifest.json").read_text())
tracked = subprocess.check_output(["git", "diff", "--name-only", "-z"], cwd=ROOT).decode().split("\0")
untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
paths = sorted({p for p in tracked + untracked if p and not p.startswith("reports/")})
hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths if (ROOT / p).is_file()}
changed = [p for p, digest in hashes.items() if previous["source_target_sha256"].get(p) != digest]
result = {
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
    "tracked_changed_file_count": len([p for p in tracked if p]),
    "source_target_sha256": hashes,
    "changed_since_round4": changed,
}
dest = OUT / "source_snapshot.json"
if dest.exists():
    initial = json.loads(dest.read_text())
    result["source_changes_during_review"] = [p for p in set(hashes) | set(initial["source_target_sha256"]) if hashes.get(p) != initial["source_target_sha256"].get(p)]
    dest = OUT / "review_manifest.json"
dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"manifest": str(dest), "changed_since_round4": changed, "source_changes_during_review": result.get("source_changes_during_review")}, ensure_ascii=False, indent=2))
