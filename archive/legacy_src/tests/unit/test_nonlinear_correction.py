"""tests/unit/test_nonlinear_correction.py

Unit tests for the nonlinear GBT correction layer.

Coverage
--------
1. データリーク監査
   - audit_no_leak: sig_date < trade_date の保証
   - leaky データに対する AssertionError の発生
   - contribution_cap の警告動作

2. FeatureBuilder
   - 特徴量名・次元数の整合性
   - 単日/パネル構築の shape 検証
   - デルタ特徴量の値検証

3. TimeSeriesPurgeSplit
   - 時系列順序の保証（バリデーションは常に学習より未来）
   - purge ギャップの確認
   - 最小訓練サイズのスキップ動作
   - embargo の適用確認

4. NonlinearCorrectionLayer (モック学習データで検証)
   - correction_enabled=False 時に z_lin をそのまま返すこと
   - fit 後の predict の shape と型
   - z_final = z_lin + g の恒等式確認
   - 補正クリッピング: |g| ≤ clip_scale * |z_lin|
   - 未 fit 時の RuntimeError
   - save / load の往復整合性

5. 評価・採用判定
   - compute_net_returns: コスト控除後にグロスより小さくなること
   - CostModel の計算値
   - deflated_sharpe_ratio の p 値域 [0, 1]
   - evaluate_correction_adoption: ADOPT / REJECT の条件分岐
"""

from __future__ import annotations

import math
import os
import pickle
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from domain.correction.feature_builder import FeatureBuilder, FeatureFlags
from domain.correction.time_series_cv import (
    TimeSeriesPurgeSplit,
    audit_no_leak,
    check_contribution_cap,
)
from domain.correction.evaluation import (
    CostModel,
    PerformanceMetrics,
    compute_net_returns,
    compute_performance_metrics,
    deflated_sharpe_ratio,
    evaluate_correction_adoption,
)
from domain.correction.nonlinear_layer import (
    GBTHyperparams,
    NonlinearCorrectionLayer,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_dates(n: int, start: str = "2018-01-02") -> pd.DatetimeIndex:
    """Generate n business days starting from start."""
    return pd.bdate_range(start=start, periods=n)


def _make_panel(
    T: int = 300,
    K: int = 6,
    N_J: int = 17,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic (f_matrix, z_lin_matrix, y_matrix) panel."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((T, K)).astype(np.float32)
    z = rng.standard_normal((T, N_J)) * 0.02
    # y is correlated with z_lin plus noise to give the correction something to learn
    y = z * 0.5 + rng.standard_normal((T, N_J)) * 0.01
    return f, z, y


def _make_layer(
    enabled: bool = True,
    n_estimators: int = 20,
    clip_scale: float = 0.5,
    seed: int = 42,
) -> NonlinearCorrectionLayer:
    """Build a minimal NonlinearCorrectionLayer for fast unit tests."""
    hp = GBTHyperparams(
        library="lightgbm",
        max_depth=2,
        n_estimators=n_estimators,
        learning_rate=0.05,
        min_child_samples=10,
        reg_lambda=1.0,
        subsample=1.0,
        colsample_bytree=1.0,
        early_stopping_rounds=5,
        clip_scale=clip_scale,
        seed=seed,
    )
    return NonlinearCorrectionLayer(
        n_factors=6,
        n_sectors=17,
        hyperparams=hp,
        correction_enabled=enabled,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. データリーク監査
# ─────────────────────────────────────────────────────────────────────────────


class TestAuditNoLeak:
    def test_valid_signal_before_trade(self):
        """sig_date < trade_date は通過すること。"""
        trade = pd.bdate_range("2020-01-02", periods=10)
        sig = trade - pd.Timedelta(days=1)
        audit_no_leak(sig, trade)  # should not raise

    def test_simultaneous_dates_raises(self):
        """sig_date == trade_date はデータリーク → AssertionError。"""
        dates = pd.bdate_range("2020-01-02", periods=5)
        with pytest.raises(AssertionError, match="DATA LEAK DETECTED"):
            audit_no_leak(dates, dates)

    def test_future_sig_raises(self):
        """sig_date > trade_date（未来シグナル）はデータリーク → AssertionError。"""
        trade = pd.bdate_range("2020-01-02", periods=5)
        sig = trade + pd.Timedelta(days=1)
        with pytest.raises(AssertionError, match="DATA LEAK DETECTED"):
            audit_no_leak(sig, trade)

    def test_length_mismatch_raises(self):
        sig = pd.bdate_range("2020-01-02", periods=5)
        trade = pd.bdate_range("2020-01-02", periods=6)
        with pytest.raises(AssertionError):
            audit_no_leak(sig, trade)


class TestCheckContributionCap:
    def test_no_warn_when_below_cap(self):
        """補正が小さければ警告なし。"""
        z = np.ones(17) * 0.02
        g = np.ones(17) * 0.001  # ratio = 0.05 < 0.3
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            exceeded = check_contribution_cap(g, z, cap=0.3)
        assert not exceeded
        assert len(w) == 0

    def test_warn_when_above_cap(self):
        """補正が大きければ警告。"""
        z = np.ones(17) * 0.01
        g = np.ones(17) * 0.005  # ratio = 0.5 > 0.3
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            exceeded = check_contribution_cap(g, z, cap=0.3)
        assert exceeded
        assert any("cap exceeded" in str(wi.message) for wi in w)


# ─────────────────────────────────────────────────────────────────────────────
# 2. FeatureBuilder
# ─────────────────────────────────────────────────────────────────────────────


class TestFeatureBuilder:
    def test_base_features_names(self):
        fb = FeatureBuilder(n_factors=6, n_sectors=17, single_model=True)
        names = fb.feature_names
        assert "f_0" in names
        assert "f_5" in names
        assert "sector_id" in names

    def test_base_features_n_features(self):
        """デフォルト: 6 factor + 1 sector_id = 7。"""
        fb = FeatureBuilder(n_factors=6, n_sectors=17, single_model=True)
        assert fb.n_features == 7

    def test_squared_features(self):
        flags = FeatureFlags(use_squared=True)
        fb = FeatureBuilder(n_factors=6, n_sectors=17, flags=flags, single_model=True)
        assert fb.n_features == 6 + 6 + 1  # base + sq + sector_id
        assert "f_0_sq" in fb.feature_names

    def test_absolute_features(self):
        flags = FeatureFlags(use_absolute=True)
        fb = FeatureBuilder(n_factors=6, n_sectors=17, flags=flags, single_model=True)
        assert "f_0_abs" in fb.feature_names

    def test_build_single_day_shape(self):
        fb = FeatureBuilder(n_factors=6, n_sectors=17, single_model=True)
        f_t = np.random.randn(6).astype(np.float32)
        X = fb.build_single_day(f_t)
        assert X.shape == (17, fb.n_features)
        assert X.dtype == np.float32

    def test_sector_id_values(self):
        """sector_id 列が 0..16 であること。"""
        fb = FeatureBuilder(n_factors=6, n_sectors=17, single_model=True)
        f_t = np.ones(6, dtype=np.float32)
        X = fb.build_single_day(f_t)
        sid_col = fb.feature_names.index("sector_id")
        np.testing.assert_array_equal(X[:, sid_col], np.arange(17))

    def test_build_panel_shape(self):
        T, K, N_J = 50, 6, 17
        fb = FeatureBuilder(n_factors=K, n_sectors=N_J, single_model=True)
        f_matrix = np.random.randn(T, K).astype(np.float32)
        X, s, dates = fb.build_panel(f_matrix)
        assert X.shape == (T * N_J, fb.n_features)
        assert s.shape == (T * N_J,)

    def test_wrong_factor_dim_raises(self):
        fb = FeatureBuilder(n_factors=6, n_sectors=17, single_model=True)
        f_wrong = np.ones(5, dtype=np.float32)
        with pytest.raises(ValueError, match="f_t must have 6 elements"):
            fb.build_single_day(f_wrong)


# ─────────────────────────────────────────────────────────────────────────────
# 3. TimeSeriesPurgeSplit
# ─────────────────────────────────────────────────────────────────────────────


class TestTimeSeriesPurgeSplit:
    def _dates(self, n=500):
        return pd.bdate_range("2015-01-02", periods=n)

    def test_splits_are_chronological(self):
        """バリデーションは常に学習より未来であること。"""
        splitter = TimeSeriesPurgeSplit(n_splits=3, embargo=2, gap_purge=1)
        dates = self._dates(300)
        for train_idx, val_idx in splitter.split(dates):
            assert train_idx[-1] < val_idx[0], (
                f"Val starts {val_idx[0]} before train ends {train_idx[-1]}"
            )

    def test_purge_gap_respected(self):
        """purge 分だけ学習末尾とバリデーション先頭の間にギャップがあること。"""
        gap = 3
        splitter = TimeSeriesPurgeSplit(n_splits=3, gap_purge=gap, embargo=0)
        dates = self._dates(400)
        for train_idx, val_idx in splitter.split(dates):
            assert val_idx[0] - train_idx[-1] >= gap

    def test_min_train_size_skipped(self):
        """min_train_size を超えないフォールドはスキップされること。"""
        splitter = TimeSeriesPurgeSplit(n_splits=10, min_train_size=999)
        dates = self._dates(100)
        splits = list(splitter.split(dates))
        assert len(splits) == 0, "Should skip all folds with insufficient data"

    def test_non_datetime_index_raises(self):
        """pd.DatetimeIndex 以外は AssertionError。"""
        splitter = TimeSeriesPurgeSplit(n_splits=3)
        with pytest.raises(AssertionError):
            list(splitter.split(pd.RangeIndex(100)))

    def test_unsorted_index_raises(self):
        dates = pd.bdate_range("2020-01-01", periods=50)[::-1]  # reversed
        splitter = TimeSeriesPurgeSplit(n_splits=3)
        with pytest.raises(AssertionError, match="sorted"):
            list(splitter.split(dates))

    def test_no_overlap_between_folds(self):
        """学習とバリデーションのインデックスが重複しないこと。"""
        splitter = TimeSeriesPurgeSplit(n_splits=4, gap_purge=1, embargo=2)
        dates = self._dates(500)
        seen_val: set = set()
        for train_idx, val_idx in splitter.split(dates):
            # val_idx must not overlap with previous val_idx
            overlap = seen_val & set(val_idx.tolist())
            assert len(overlap) == 0, f"Val index overlap: {overlap}"
            seen_val.update(val_idx.tolist())


# ─────────────────────────────────────────────────────────────────────────────
# 4. NonlinearCorrectionLayer
# ─────────────────────────────────────────────────────────────────────────────

# Skip GBT tests if lightgbm is not installed
try:
    import lightgbm  # noqa: F401
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

lgbm_required = pytest.mark.skipif(
    not LIGHTGBM_AVAILABLE,
    reason="lightgbm not installed",
)


class TestNonlinearCorrectionLayerDisabled:
    def test_disabled_returns_z_lin(self):
        """correction_enabled=False は z_lin をそのまま返すこと。"""
        layer = _make_layer(enabled=False)
        z = np.random.randn(17)
        f_t = np.random.randn(6).astype(np.float32)
        result = layer.predict(f_t, z)
        np.testing.assert_array_almost_equal(result, z)

    def test_unfitted_raises(self):
        """未 fit で predict() は RuntimeError。"""
        layer = _make_layer(enabled=True)
        with pytest.raises(RuntimeError, match="not fitted"):
            layer.predict(np.zeros(6), np.zeros(17))


@lgbm_required
class TestNonlinearCorrectionLayerFit:
    def test_fit_predict_shape(self):
        """fit + predict が (N_J,) の ndarray を返すこと。"""
        T = 200
        f, z, y = _make_panel(T=T)
        dates = _make_dates(T)
        sig_dates = dates - pd.Timedelta(days=1)

        layer = _make_layer()
        layer.fit(f, z, y, dates=dates, signal_dates=sig_dates)

        f_t = f[-1]
        z_t = z[-1]
        result = layer.predict(f_t, z_t)
        assert result.shape == (17,)
        assert result.dtype == float

    def test_z_final_equals_z_lin_plus_g(self):
        """z_final と z_lin + g_clipped が一致すること。"""
        T = 200
        f, z, y = _make_panel(T=T)
        dates = _make_dates(T)
        sig_dates = dates - pd.Timedelta(days=1)

        layer = _make_layer()
        layer.fit(f, z, y, dates=dates, signal_dates=sig_dates)

        f_t, z_t = f[100], z[100]
        attr = layer.predict_with_attribution(f_t, z_t)
        np.testing.assert_array_almost_equal(
            attr["z_final"], attr["z_lin"] + attr["g_clipped"]
        )

    def test_clipping_enforced(self):
        """補正が clip_scale * |z_lin| 以内に収まること。"""
        T = 200
        clip = 0.3
        f, z, y = _make_panel(T=T)
        dates = _make_dates(T)
        sig_dates = dates - pd.Timedelta(days=1)

        layer = _make_layer(clip_scale=clip)
        layer.fit(f, z, y, dates=dates, signal_dates=sig_dates)

        for t in range(T - 20, T):
            attr = layer.predict_with_attribution(f[t], z[t])
            g_clip = attr["g_clipped"]
            z_lin = attr["z_lin"]
            bound = clip * np.abs(z_lin)
            assert np.all(g_clip <= bound + 1e-9)
            assert np.all(g_clip >= -bound - 1e-9)

    def test_leak_detection_on_fit(self):
        """signal_dates >= trade_dates は AssertionError を raise すること。"""
        T = 200
        f, z, y = _make_panel(T=T)
        dates = _make_dates(T)
        leaky_sig_dates = dates + pd.Timedelta(days=1)  # future!

        layer = _make_layer()
        with pytest.raises(AssertionError, match="DATA LEAK DETECTED"):
            layer.fit(f, z, y, dates=dates, signal_dates=leaky_sig_dates)

    def test_reproducibility(self):
        """同じシードで2回 fit すると同じ予測になること。"""
        T = 200
        f, z, y = _make_panel(T=T, seed=123)
        dates = _make_dates(T)
        sig_dates = dates - pd.Timedelta(days=1)

        def _fit_predict():
            layer = _make_layer(seed=42)
            layer.fit(f, z, y, dates=dates, signal_dates=sig_dates)
            return layer.predict(f[-1], z[-1])

        r1 = _fit_predict()
        r2 = _fit_predict()
        np.testing.assert_array_equal(r1, r2)

    def test_save_load_roundtrip(self, tmp_path):
        """save / load 後に同じ予測が得られること。"""
        T = 200
        f, z, y = _make_panel(T=T)
        dates = _make_dates(T)
        sig_dates = dates - pd.Timedelta(days=1)

        layer = _make_layer()
        layer.fit(f, z, y, dates=dates, signal_dates=sig_dates)
        pred_before = layer.predict(f[-1], z[-1])

        save_path = str(tmp_path / "correction_layer")
        layer.save(save_path)

        layer2 = NonlinearCorrectionLayer.load(save_path)
        pred_after = layer2.predict(f[-1], z[-1])

        np.testing.assert_array_almost_equal(pred_before, pred_after)

    @lgbm_required
    def test_per_sector_mode(self):
        """per_sector モードでも正常に動作すること。"""
        T = 200
        f, z, y = _make_panel(T=T)
        dates = _make_dates(T)
        sig_dates = dates - pd.Timedelta(days=1)

        hp = GBTHyperparams(
            max_depth=2, n_estimators=10, learning_rate=0.05,
            min_child_samples=10, reg_lambda=1.0, clip_scale=0.5,
        )
        layer = NonlinearCorrectionLayer(
            multi_output_mode="per_sector", hyperparams=hp
        )
        layer.fit(f, z, y, dates=dates, signal_dates=sig_dates)
        result = layer.predict(f[-1], z[-1])
        assert result.shape == (17,)


# ─────────────────────────────────────────────────────────────────────────────
# 5. 評価・採用判定
# ─────────────────────────────────────────────────────────────────────────────


class TestCostModel:
    def test_total_one_way(self):
        cm = CostModel(commission_rate=0.001, spread_rate=0.0005, impact_rate=0.0002)
        assert cm.total_one_way == pytest.approx(0.0017)

    def test_round_trip(self):
        cm = CostModel(commission_rate=0.001, spread_rate=0.0005, impact_rate=0.0002)
        assert cm.round_trip == pytest.approx(0.0034)

    def test_compute_net_returns_reduces(self):
        """ネットリターンはグロスより小さいこと（コストが正のため）。"""
        gross = pd.Series(np.full(100, 0.001), index=_make_dates(100))
        net = compute_net_returns(gross, cost_model=CostModel())
        assert (net < gross).all()


class TestDeflatedSharpeRatio:
    def test_pvalue_in_unit_interval(self):
        """DSR p 値は [0, 1] に収まること。"""
        p, sr_star = deflated_sharpe_ratio(sharpe_hat=2.0, n_trials=50, n_obs=500)
        assert 0.0 <= p <= 1.0

    def test_single_trial_reference(self):
        """試行1回なら SR* ≈ 0。"""
        p, sr_star = deflated_sharpe_ratio(sharpe_hat=2.0, n_trials=1, n_obs=500)
        assert sr_star == pytest.approx(0.0)

    def test_more_trials_lower_pvalue(self):
        """試行数が多いほど p 値が低く（厳しく）なること。"""
        p1, _ = deflated_sharpe_ratio(sharpe_hat=3.0, n_trials=1, n_obs=500)
        p100, _ = deflated_sharpe_ratio(sharpe_hat=3.0, n_trials=100, n_obs=500)
        assert p100 <= p1


class TestEvaluateAdoption:
    def _make_metrics(
        self,
        label: str,
        ar: float,
        rr: float,
        dsr_p: float,
        n_trials: int = 1,
    ) -> PerformanceMetrics:
        return PerformanceMetrics(
            label=label,
            ar=ar,
            risk=ar / rr if rr > 0 else float("nan"),
            rr=rr,
            mdd=-0.05,
            sharpe=rr,
            ic_mean=0.05,
            dsr_pvalue=dsr_p,
            sr_star=0.5,
            n_obs=500,
            n_trials=n_trials,
        )

    def test_adopt_when_all_conditions_met(self):
        m_lin = self._make_metrics("lin", ar=0.10, rr=1.0, dsr_p=0.10)
        m_cor = self._make_metrics("cor", ar=0.15, rr=1.5, dsr_p=0.03, n_trials=5)
        decision = evaluate_correction_adoption(m_lin, m_cor, significance_level=0.05)
        assert decision.decision == "ADOPT"

    def test_reject_when_rr_not_improved(self):
        m_lin = self._make_metrics("lin", ar=0.10, rr=1.5, dsr_p=0.10)
        m_cor = self._make_metrics("cor", ar=0.15, rr=1.0, dsr_p=0.03, n_trials=5)
        decision = evaluate_correction_adoption(m_lin, m_cor)
        assert decision.decision == "REJECT"
        assert "R/R" in decision.reason

    def test_reject_when_ar_not_improved(self):
        m_lin = self._make_metrics("lin", ar=0.15, rr=1.0, dsr_p=0.10)
        m_cor = self._make_metrics("cor", ar=0.10, rr=1.5, dsr_p=0.03, n_trials=5)
        decision = evaluate_correction_adoption(m_lin, m_cor)
        assert decision.decision == "REJECT"
        assert "AR" in decision.reason

    def test_reject_when_dsr_not_significant(self):
        m_lin = self._make_metrics("lin", ar=0.10, rr=1.0, dsr_p=0.10)
        m_cor = self._make_metrics("cor", ar=0.15, rr=1.5, dsr_p=0.20, n_trials=5)
        decision = evaluate_correction_adoption(m_lin, m_cor, significance_level=0.05)
        assert decision.decision == "REJECT"
        assert "DSR" in decision.reason

    def test_reject_all_conditions_failed(self):
        """全条件失敗でも 'REJECT' を返し、3 項目が理由に含まれること。"""
        m_lin = self._make_metrics("lin", ar=0.20, rr=2.0, dsr_p=0.01)
        m_cor = self._make_metrics("cor", ar=0.10, rr=1.0, dsr_p=0.50, n_trials=20)
        decision = evaluate_correction_adoption(m_lin, m_cor, significance_level=0.05)
        assert decision.decision == "REJECT"
        # All three failure reasons should appear
        assert "R/R" in decision.reason
        assert "AR" in decision.reason
        assert "DSR" in decision.reason


# ─────────────────────────────────────────────────────────────────────────────
# 6. GBTHyperparams バリデーション
# ─────────────────────────────────────────────────────────────────────────────


class TestGBTHyperparamsValidation:
    def test_valid_defaults(self):
        """デフォルト値で例外が出ないこと。"""
        hp = GBTHyperparams()
        assert hp.max_depth == 2

    def test_depth_too_deep_raises(self):
        with pytest.raises(AssertionError, match="max_depth must be 2 or 3"):
            GBTHyperparams(max_depth=5)

    def test_depth_too_shallow_raises(self):
        with pytest.raises(AssertionError, match="max_depth must be 2 or 3"):
            GBTHyperparams(max_depth=1)

    def test_n_estimators_too_large_raises(self):
        with pytest.raises(AssertionError, match="n_estimators"):
            GBTHyperparams(n_estimators=500)

    def test_learning_rate_too_large_raises(self):
        with pytest.raises(AssertionError, match="learning_rate"):
            GBTHyperparams(learning_rate=0.5)

    def test_lightgbm_params_conversion(self):
        hp = GBTHyperparams()
        params = hp.to_lightgbm_params()
        assert "max_depth" in params
        assert "reg_lambda" in params
        assert params["max_depth"] == hp.max_depth

    def test_monotone_constraints_in_lightgbm_params(self):
        hp = GBTHyperparams()
        constraints = [1, 0, -1, 0, 0, 0, 0]
        params = hp.to_lightgbm_params(monotone_constraints=constraints)
        assert "monotone_constraints" in params
        assert params["monotone_constraints"] == constraints
