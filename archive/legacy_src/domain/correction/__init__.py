"""domain.correction – Nonlinear GBT correction layer package.

Public API
----------
NonlinearCorrectionLayer : main class (fit / predict / predict_with_attribution)
TimeSeriesPurgeSplit      : leak-safe time-series CV splitter
audit_no_leak             : data-leak assertion helper
evaluate_correction       : benchmark comparison and adoption gate
CostModel                 : transaction-cost model
"""

from .nonlinear_layer import NonlinearCorrectionLayer  # noqa: F401
from .time_series_cv import TimeSeriesPurgeSplit, audit_no_leak  # noqa: F401
from .evaluation import CostModel, evaluate_correction_adoption  # noqa: F401
