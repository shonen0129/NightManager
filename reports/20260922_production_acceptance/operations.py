"""Bounded operations checks; credentials and account values are never printed."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import copy
import hashlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def save(label, payload):
    (OUT / f"{label}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(payload, ensure_ascii=False, default=str))


def fix_key():
    path = ROOT / ".env"
    text = path.read_text()
    pattern = r"(?m)^(?:export\s+)?TACHIBANA_PRIVATE_KEY_PATH\s*=.*$"
    matches = re.findall(pattern, text)
    if len(matches) != 1:
        raise ValueError("Expected exactly one private-key path setting")
    replacement = ROOT / "creds/e_api_private_key.pem"
    if not replacement.is_file():
        raise FileNotFoundError("Current-workspace private key is absent")
    updated = re.sub(pattern, f"TACHIBANA_PRIVATE_KEY_PATH={replacement}", text)
    if updated != text:
        # Keep a private rollback copy outside reports and git-managed content.
        backup_dir = ROOT / "var/artifacts/acceptance_private"
        backup_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        backup = backup_dir / "env_before_key_path"
        descriptor = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".env.acceptance-")
        try:
            with os.fdopen(descriptor, "w") as handle:
                handle.write(updated)
            os.chmod(temporary, path.stat().st_mode & 0o777)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    save("key_path_result", {"changed": updated != text, "key_exists": True,
                            "only_setting_changed": "TACHIBANA_PRIVATE_KEY_PATH"})


def github():
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    auth_source = "environment" if token else "none"
    if not token:
        result = subprocess.run(
            ["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
            capture_output=True, text=True, timeout=15,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        token = values.get("password")
        if token:
            auth_source = "configured_git_credential_helper"
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    base = "https://api.github.com/repos/shonen0129/NightManager"
    payload = {"auth_available": bool(token), "auth_source": auth_source, "repository": base}
    for label, suffix in (("repository", ""), ("workflows", "/actions/workflows"), ("runs", "/actions/runs?per_page=5")):
        response = requests.get(base + suffix, headers=headers, timeout=20)
        data = response.json()
        if label == "repository":
            payload[label] = {k: data.get(k) for k in ("full_name", "private", "default_branch", "permissions")}
        elif label == "workflows":
            payload[label] = {"total_count": data.get("total_count"), "workflows": [
                {k: item.get(k) for k in ("id", "name", "path", "state")}
                for item in data.get("workflows", [])]}
        else:
            payload[label] = {"total_count": data.get("total_count"), "runs": [
                {k: item.get(k) for k in ("html_url", "head_sha", "status", "conclusion")}
                for item in data.get("workflow_runs", [])]}
        payload[label]["http_status"] = response.status_code
    save("github_status", payload)


def scheduler():
    services = ["update-market-data", "distribution-diagnostics", "decision", "close", "pnl_report"]
    payload = {"checked_at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(), "services": []}
    for name in services:
        result = subprocess.run(["launchctl", "print", f"gui/{os.getuid()}/com.leadlag.{name}"],
                                capture_output=True, text=True, timeout=10)
        def field(pattern):
            match = re.search(pattern, result.stdout)
            return match.group(1) if match else None
        payload["services"].append({
            "name": name, "registered": result.returncode == 0,
            "current_workspace": f"working directory = {ROOT}" in result.stdout,
            "state": field(r"\bstate = ([^\n]+)"),
            "last_exit": field(r"last exit code = ([^\n]+)"),
            "runs": field(r"\bruns = (\d+)"),
        })
    save("scheduler_status", payload)


def fills():
    from leadlag.execution.broker_ops import build_api_client
    from leadlag.execution.pricing import fetch_fill_prices
    files = sorted((ROOT / "var/results").glob("*production_close_positions/close_execution_log.json"))
    evidence = []
    orders = []
    for path in files:
        raw = json.loads(path.read_text())
        records = raw.get("close_results", [])
        real = [record for record in records if record.get("order_id") and not raw.get("dry_run")]
        copied = copy.deepcopy(real)
        for record in copied:
            record.setdefault("eigyou_day", path.parent.name[:8])
        orders.extend(copied)
        evidence.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "dry_run": raw.get("dry_run"), "records": len(records), "real_order_records": len(real),
                         "saved_fill_details": sum(bool(r.get("fill_detail")) for r in real)})
    unique = {str(order["order_id"]): order for order in orders}
    records = list(unique.values())
    errors = []
    if records:
        broker = build_api_client(None, None, False)
        try:
            try:
                fetch_fill_prices(broker, records, wait_seconds=0.0)
            except Exception as exc:
                errors.append(type(exc).__name__)
        finally:
            broker.close()
    private = ROOT / "var/results/20260922_production_acceptance/historical_fills"
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = private / "readonly_results.json"
    target.write_text(json.dumps(records, indent=2, ensure_ascii=False, default=str))
    target.chmod(0o600)
    save("historical_fill_status", {"logs": evidence, "unique_order_count": len(records), "errors": errors,
                                   "fresh_fill_details": sum(bool(r.get("fill_detail")) and r.get("fill_status") != "FETCH_ERROR" for r in records),
                                   "query_failed_records": sum(r.get("fill_status") == "FETCH_ERROR" for r in records),
                                   "snapshot_path": str(target.relative_to(ROOT))})


if __name__ == "__main__":
    {"fix-key": fix_key, "github": github, "scheduler": scheduler, "fills": fills}[sys.argv[1]]()
