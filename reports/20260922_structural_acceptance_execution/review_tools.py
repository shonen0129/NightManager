"""Read-only review snapshot and deadline-bounded local verification."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def snapshot(name: str) -> None:
    paths = subprocess.check_output(
        ['rg', '--files', 'src', 'tests', 'tools', 'scripts', 'configs', 'docs',
         '.github', 'pyproject.toml', 'uv.lock'], cwd=ROOT, text=True,
    ).splitlines()
    result = {
        'recorded_at': datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'subject': 'Current dirty worktree, not HEAD alone',
        'sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sorted(paths)},
    }
    (OUT / f'{name}.json').write_text(json.dumps(result, indent=2) + '\n')
    print('snapshot', name, len(paths))


def run(label: str, timeout: float, cmd: list[str]) -> int:
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONPATH=str(ROOT / 'src'))
    start = time.monotonic()
    with (OUT / f'{label}.log').open('wb') as log:
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            code = 124
    result = {'command': cmd, 'timeout_seconds': timeout, 'exit_code': code,
              'timed_out': timed_out, 'elapsed_seconds': time.monotonic() - start}
    (OUT / f'{label}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(label, json.dumps(result))
    return code


if __name__ == '__main__':
    if sys.argv[1] == 'snapshot':
        snapshot(sys.argv[2])
    elif sys.argv[1] == 'compare':
        before = json.loads((OUT / 'before_manifest.json').read_text())
        after = json.loads((OUT / 'after_manifest.json').read_text())
        changed = [p for p in sorted(before['sha256'].keys() | after['sha256'].keys())
                   if before['sha256'].get(p) != after['sha256'].get(p)]
        result = {'changed_paths': changed, 'before_files': len(before['sha256']),
                  'after_files': len(after['sha256']), 'same_head': before['head'] == after['head']}
        (OUT / 'source_comparison.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))
    else:
        raise SystemExit(run(sys.argv[2], float(sys.argv[3]), sys.argv[4:]))
