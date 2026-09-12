import os
import signal
import subprocess
import sys
import time

limit = float(sys.argv[1])
started = time.monotonic()
proc = subprocess.Popen(sys.argv[2:], start_new_session=True)
try:
    code = proc.wait(timeout=limit)
except subprocess.TimeoutExpired:
    print(f'WATCHDOG: deadline {limit}s exceeded', flush=True, file=sys.stderr)
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()
    code = 124
print(f'WATCHDOG: exit={code} elapsed={time.monotonic()-started:.1f}s', flush=True, file=sys.stderr)
sys.exit(code)
