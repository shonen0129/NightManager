"""Poll one GitHub Actions run using the configured git credential."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import requests


def main() -> int:
    owner_repo, run_id = sys.argv[1], sys.argv[2]
    result = subprocess.run(
        ["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
        capture_output=True, text=True, check=True, timeout=15,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
               "Authorization": f"Bearer {values['password']}"}
    url = f"https://api.github.com/repos/{owner_repo}/actions/runs/{run_id}"
    for _ in range(60):
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()
        payload = {key: data.get(key) for key in ("html_url", "head_sha", "status", "conclusion",
                                                    "run_started_at", "updated_at")}
        print(json.dumps(payload), flush=True)
        if data.get("status") == "completed":
            jobs = requests.get(data["jobs_url"], headers=headers, timeout=20).json().get("jobs", [])
            print(json.dumps({"jobs": [{key: job.get(key) for key in ("name", "status", "conclusion", "html_url")}
                               for job in jobs]}), flush=True)
            return 0 if data.get("conclusion") == "success" else 1
        time.sleep(10)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
