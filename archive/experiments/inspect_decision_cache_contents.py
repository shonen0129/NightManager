import numpy as np
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "data" / "decision_cache.npz"
if not p.exists():
    raise SystemExit(f"decision_cache not found at {p}")

nz = np.load(p, allow_pickle=True)
keys = list(nz.keys())
print('npz keys:', keys)

if 'numeric_cols' in nz:
    cols = [c.decode() if isinstance(c, bytes) else str(c) for c in nz['numeric_cols']]
    print('num cols (count):', len(cols))
    print('sample cols:', cols[:120])
    print('contains any topix substring?:', any('topix' in c.lower() for c in cols))
    print('contains jp_beta substring?:', any('beta' in c.lower() for c in cols))
else:
    print('numeric_cols not present; available keys:', keys)
