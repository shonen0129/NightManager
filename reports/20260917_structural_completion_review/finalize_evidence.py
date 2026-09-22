"""Record this review's changes separately from pre-existing workspace edits."""

import difflib
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    out = Path(__file__).resolve().parent
    archive = out / "before_source.tar.gz"
    if not archive.exists():
        shutil.copy2("/private/tmp/leadlag_s0_s8_review_before.tar.gz", archive)
    paths = []
    for directory in ("src", "tests", "tools", "scripts", "configs", ".github"):
        paths.extend(p for p in (root / directory).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts
                     and p.suffix in {".py", ".sh", ".yaml", ".yml", ".toml", ".json"})
    paths.extend(root / name for name in ("pyproject.toml", "uv.lock"))
    before = json.loads((out / "before_manifest.json").read_text())
    after = {str(p.relative_to(root)): digest(p) for p in sorted(paths)}
    (out / "after_manifest.json").write_text(json.dumps(after, indent=2) + "\n")
    changed = [p for p in sorted(before.keys() | after.keys()) if before.get(p) != after.get(p)]
    diff = []
    with tarfile.open(archive) as bundle:
        for name in changed:
            previous = bundle.extractfile(name).read().decode() if name in before else ""
            current = (root / name).read_text() if name in after else ""
            diff.extend(difflib.unified_diff(previous.splitlines(keepends=True),
                                            current.splitlines(keepends=True),
                                            fromfile=f"before/{name}", tofile=f"after/{name}"))
    (out / "review_changes.patch").write_text("".join(diff))
    replay_before = json.loads((out / "replay_before.json").read_text())
    replay_after = json.loads((out / "replay_after.json").read_text())
    assert replay_before == replay_after, "Numerical before/after replay changed"
    fixtures = root / "tests/regression/baselines"
    evidence = {
        "baseline_archive_sha256": digest(archive),
        "changed": changed,
        "removed": sorted(before.keys() - after.keys()),
        "added": sorted(after.keys() - before.keys()),
        "replay": {key: "exactly_equal" for key in replay_before},
        "replay_fixture_sha256": {str(p.relative_to(root)): digest(p) for p in sorted(fixtures.rglob("*"))
                                  if p.is_file() and (p.suffix == ".npy" or p.name.endswith(".csv.gz"))},
        "wheel_sha256": digest(out / "dist/leadlag-2.1.0-py3-none-any.whl"),
        "limitations": ["Does not establish original S0 parity or cross-source equivalence",
                        "Replay excludes production ML/macro/ADR; intraday adjustment is fixed to zero",
                        "Before archive contains code/config/JSON; binary fixtures remain separately hashed",
                        "Review patch excludes documentation: no pre-review documentation snapshot was taken"],
    }
    (out / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"changed_files": len(changed), "replay_cases": len(replay_before), "exact_equality": True}))


if __name__ == "__main__":
    main()
