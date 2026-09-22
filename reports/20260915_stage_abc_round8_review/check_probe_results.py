"""Check collected regression values while the independent full suite runs."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
source = (OUT / "validate_review.py").read_text().split("test_log =", 1)[0]
source = source.replace('summary = read_json("review_summary.json")\n', "")
namespace = {"__file__": str(OUT / "validate_review.py")}
exec(compile(source, str(OUT / "validate_review.py"), "exec"), namespace)
checks = {key: bool(value) for key, value in namespace["checks"].items()}
result = {"passed": all(checks.values()), "checks": checks, "note": "This validates probe outcomes only; full-suite and report checks run separately."}
(OUT / "probe_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"passed": result["passed"], "checks": len(checks), "failed": [key for key, value in checks.items() if not value]}, ensure_ascii=False))
raise SystemExit(0 if result["passed"] else 1)
