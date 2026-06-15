"""domain.gp – GP 不確実性推定モジュール パッケージ

Public API
----------
GPUncertaintyModule  : fit / predict_mean_var / compute_confidence / apply_sizing
GPConfig             : モジュール設定データクラス
ConfidenceScorer     : 確信度スコア κ_t の算出（ステートフル）
ConfidenceConfig     : 確信度算出の設定
KernelConfig         : GP カーネルの設定
CalibrationResult    : 較正テストの結果
coverage_test        : GP 不確実性の較正テスト
reliability_diagram  : 較正チャートの生成
"""

from .calibration import (
    CalibrationResult,
    coverage_test,
    coverage_test_by_sector,
    reliability_diagram,
)
from .confidence import (
    ConfidenceConfig,
    ConfidenceScorer,
    compute_confidence_from_sigma2,
    quantile_return_decomposition,
    regime_kappa_decomposition,
)
from .gp_uncertainty import GPConfig, GPUncertaintyModule
from .kernels import KernelConfig, build_structured_kernel, extract_kernel_params

__all__ = [
    # コア
    "GPUncertaintyModule",
    "GPConfig",
    # カーネル
    "KernelConfig",
    "build_structured_kernel",
    "extract_kernel_params",
    # 確信度
    "ConfidenceScorer",
    "ConfidenceConfig",
    "compute_confidence_from_sigma2",
    "regime_kappa_decomposition",
    "quantile_return_decomposition",
    # 較正
    "CalibrationResult",
    "coverage_test",
    "coverage_test_by_sector",
    "reliability_diagram",
]
