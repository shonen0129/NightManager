import numpy as np
from pathlib import Path

p = Path(__file__).resolve().parents[1] / 'data' / 'decision_cache.npz'
print('cache path:', p)
npz = np.load(p, allow_pickle=False)
cols = list(npz['numeric_cols'])
print('columns count:', len(cols))
cols = [c for c in cols]
# decode bytes if necessary
cols = [c.decode('utf-8') if isinstance(c, bytes) else c for c in cols]
print('sample cols:', cols[:10])
arr = npz['numeric_data']
print('numeric_data shape:', arr.shape)

# find topix and jp_beta cols
for name in ['topix_night_return'] + [f'jp_beta_{i}.T' for i in range(1617,1634)]:
    if name in cols:
        idx = cols.index(name)
        col = arr[:, idx]
        n_finite = np.isfinite(col).sum()
        print(name, 'finite count =', n_finite)
    else:
        print(name, 'NOT PRESENT')
