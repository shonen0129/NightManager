"""Stable identity helpers for DataFrame-backed caches."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    """Return a content and schema fingerprint for *frame*.

    ``hash_pandas_object`` covers index and cell values but intentionally does
    not include column labels, labels' order, dtypes, or index names.  Those
    fields carry ticker semantics in ``df_exec`` and therefore belong to the
    cache identity as well.  The returned digest is deterministic for an
    unchanged DataFrame and changes when its schema or values change.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(frame).__name__}")

    schema: dict[str, Any] = {
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "columns": [
            {"type": type(column).__qualname__, "value": repr(column), "dtype": str(dtype)}
            for column, dtype in zip(frame.columns, frame.dtypes, strict=True)
        ],
        "index": {
            "type": type(frame.index).__qualname__,
            "dtype": str(frame.index.dtype),
            "names": [repr(name) for name in frame.index.names],
        },
    }
    value_hash = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update(json.dumps(schema, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(value_hash).tobytes())
    return digest.hexdigest()
