"""Build the final Round 7 summary after implementation and verification."""

from __future__ import annotations

import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
PRIOR = OUT.parent / "20260914_stage_abc_round6_review"
summary = json.loads((PRIOR / "review_summary.json").read_text())

summary.update(
    review_date="2026-09-15",
    round=7,
    verdict="CONDITIONAL_PASS",
    scope=(
        "前回レビューのW01/X01（日付来歴）とW02/X02（VaR総deadline）を実装修正し、"
        "元のF01〜F22および後続R/S/T/U/Vの関連再発を再確認"
    ),
    production_source_modified_by_review=True,
)
summary["findings"] = []
summary["resolved_fixes"] = [
    {
        "id": "X01",
        "status": "resolved",
        "file": "src/leadlag/utils/distribution_provenance.py",
        "change": "pd.to_datetime後のNaTをnormalize・timezone処理前に拒否し、loaderのDataValidationError/fallback契約へ統一",
        "evidence": "provenance.json#results",
    },
    {
        "id": "X02",
        "status": "resolved",
        "file": "src/leadlag/execution/var_history.py",
        "change": "絶対deadlineをcache初期化、入力load/snapshot/hash、cache read/write、BT、結果採用へ伝播。directory copyはチャンク単位で期限確認",
        "evidence": "deadline.json",
    },
]
summary["previous_round"] = {
    "W01": "矛盾alias/null/配列/Infと文字列NaT/空文字を共通validatorが拒否し、下流fallbackへ進むことを確認。",
    "W02": "SQLite lock、worker中のsnapshot寿命、hash/cache/copyを含む総deadlineを確認。",
}
for row in summary["original_findings"]:
    if row.get("id") == "F06":
        row["status"] = "config_and_version_identity_resolved"
        row.pop("finding", None)
    if row.get("id") == "F12":
        row["status"] = "atomicity_and_provenance_resolved"
        row.pop("finding", None)
log = (OUT / "pytest_full.log").read_text()
watchdog_log = log + (OUT / "pytest_full.stderr").read_text()
match = re.search(r"(\d+) passed, (\d+) warnings in ([\d.]+)s", log)
if not match or "WATCHDOG: exit=0" not in watchdog_log:
    raise RuntimeError("Full test run has not completed successfully")
summary["verification"]["pytest"].update(
    passed=int(match[1]), warnings=int(match[2]), seconds=float(match[3]), exit_code=0
)
summary["verification"]["mypy"] = "passed, 122 production source files"
summary["verification"]["provenance_contract"] = {
    "cases": 72,
    "passed": 72,
    "failed": 0,
    "evidence": "provenance.json",
}
summary["limitations"] = [
    "修正確認は記載の故障注入・境界条件と既存回帰範囲に基づく。全API応答・市場状態を保証しない。",
    "ファイルのOSレベルread自体が停止不能な場合はdaemon workerが終了を待つため、運用では外側watchdogも維持する。",
    "実約定照合・新規OOS収益実験・全外部系列の公表時刻の証明は未実施。",
    "VaR検証では実cache・一時DBを使い、重いBT数値ループをマーカーまたは待機workerへ置換した。",
    "未来target摂動は合成入力でmacro無効。モデルexposureと約定後exposureは別。",
]
summary["operational_pending"] = [
    "models/ml_order_overlay/phase2_8はlegacy artifactとして実runnerに拒否される。検証可能な学習データからversioned artifactを再生成する必要がある。",
    "同じ入力・同じ本番ML artifactでの本番/BT weights照合は未実施。",
]
(OUT / "review_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
)
print(
    json.dumps(
        {
            "verdict": summary["verdict"],
            "findings": len(summary["findings"]),
            "resolved_fixes": len(summary["resolved_fixes"]),
            "pytest": summary["verification"]["pytest"],
        },
        ensure_ascii=False,
    )
)
