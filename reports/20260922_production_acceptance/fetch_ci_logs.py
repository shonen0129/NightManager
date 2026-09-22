from __future__ import annotations
import os, subprocess, sys
import requests

def main() -> None:
    repo, job_id = sys.argv[1], sys.argv[2]
    result = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n", capture_output=True, text=True, check=True)
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {values['password']}", "X-GitHub-Api-Version": "2022-11-28"}
    response = requests.get(f"https://api.github.com/repos/{repo}/actions/jobs/{job_id}/logs", headers=headers, timeout=30)
    response.raise_for_status()
    sys.stdout.write(response.text)

if __name__ == "__main__":
    main()
