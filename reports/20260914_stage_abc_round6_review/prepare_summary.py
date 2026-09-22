"""Build Round 6 handoff metadata from reviewed evidence, preserving old reports."""
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
PREVIOUS = OUT.parent / "20260914_stage_abc_round5_review"
summary = json.loads((PREVIOUS / "review_summary.json").read_text())
summary.update(round=6, verdict="BLOCK", scope="前回V01〜V04の修正差分と、F01〜F22/R〜Uの関連再発確認")
summary["findings"] = [
    {"id": "W01", "priority": "P2", "stage": "B", "title": "来歴validatorの契約が分かれ、矛盾・不正型を拒否しきれない", "file": "src/leadlag/utils/gap_matrix_io.py", "line": 75, "related": ["V01", "F12", "T03"], "evidence": "boundaries.json#metadata", "impact": "sig_dateとsignal_dateが矛盾するbundleや配列型trade_dateを互換APIが受理する。null日付等ではon-demand前に未処理例外になる。通常decideの再検証は別経路。"},
    {"id": "W02", "priority": "P2", "stage": "B", "title": "gap snapshotの取得に停止期限がなく、ロック待ちがVaR timeoutを迂回する", "file": "src/leadlag/execution/var_history.py", "line": 146, "related": ["V04", "F06", "R05"], "evidence": "boundaries.json#locked_snapshot", "impact": "実SQLite排他ロック下でvar_history_timeout=1でも35秒を超えて待機し、解除後に計算続行。BT timeout後にはworkerが使用中のsnapshotをcallerが削除する寿命管理の問題も確認。"}
]
summary["previous_round"] = {
    "V01": "元の保存障害・manifest/metadata欠落は拒否。別の来歴境界がW01として残存。",
    "V02": "再現解消。部分失効CANCELLED、訂正/取消失敗SUBMITTED、部分約定数量は保持。",
    "V03": "再現解消。実builderでcache再利用なし、freshとの差0。",
    "V04": "元のA/B版競合は再現解消。snapshotの停止期限・寿命管理がW02として残存。"
}
for row in summary["original_findings"]:
    row.pop("finding", None)
    if row["id"] in {"F01", "F11"}:
        row["status"] = "reproduction_resolved"
    elif row["id"] == "F06":
        row.update(status="config_and_version_identity_resolved_snapshot_deadline_residual", finding="W02")
    elif row["id"] == "F12":
        row.update(status="atomicity_reproduction_resolved_provenance_boundary_residual", finding="W01")
log = (OUT / "pytest_full.log").read_text()
match = re.search(r"(\d+) passed, (\d+) warnings in ([\d.]+)s", log)
if not match or "WATCHDOG: exit=0" not in log:
    raise RuntimeError("Full test run has not completed successfully")
summary["verification"]["pytest"].update(passed=int(match[1]), warnings=int(match[2]), seconds=float(match[3]))
summary["verification"]["mypy"] = "passed, 121 production source files"
summary["limitations"] = [
    "再現解消は記載の条件と境界に限る。全API応答・市場状態を保証しない。",
    "実約定照合・新規OOS収益実験・全外部系列の公表時刻の証明は未実施。",
    "VaR競合/ロック/worker寿命の確認では実cache・一時DBを使い、重いBT数値ループをマーカーまたは待機workerへ置換した。",
    "未来target摂動は合成入力でmacro無効。モデルexposureと約定後exposureは別。"
]
(OUT / "review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"verdict": summary["verdict"], "findings": len(summary["findings"]), "pytest": summary["verification"]["pytest"]}, ensure_ascii=False))
