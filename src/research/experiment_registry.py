"""Backward-compatible re-export of the canonical experiment registry.

The canonical implementation lives in ``leadlag.experiment_registry``.
Research scripts that were written against ``research.experiment_registry``
continue to work while the research package is being consolidated.
"""

from __future__ import annotations

from leadlag.experiment_registry import *  # noqa: F401,F403
