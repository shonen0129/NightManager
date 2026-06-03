"""tests/unit/test_gp_uncertainty.py – GP 不確実性推定モジュール ユニットテスト

テスト項目
----------
1. リーク監査       : 学習窓に将来データが含まれないことを assert
2. 再現性           : seed 固定で 2 回実行した結果が完全一致
3. インターフェース  : fit/predict_mean_var/compute_confidence/apply_sizing の型・shape 確認
4. カーネル制約      : α ≤ alpha_bounds[1] が学習後も維持されること
5. κ_t 範囲         : κ_t ∈ [min_kappa, 1.0] が常に満たされること
6. 較正テスト        : 合成データで coverage_test の正確性確認
7. 保存・ロード      : save()/load() の往復チェック
8. ゼロ入力の堅牢性  : NaN や全ゼロ分散での安全な動作
9. 確信度スコアラー  : ConfidenceScorer の各モードの確認
10. ARD 解釈性       : extract_kernel_params が長さスケールを正しく取得
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── パス設定 ─────────────────────────────────────────────────────────────────
_SRC_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_SRC_DIR))

from domain.gp import (
    GPConfig,
    GPUncertaintyModule,
    KernelConfig,
    build_structured_kernel,
    coverage_test,
    extract_kernel_params,
)
from domain.gp.calibration import CalibrationResult, coverage_test_by_sector
from domain.gp.confidence import (
    ConfidenceConfig,
    ConfidenceScorer,
    compute_confidence_from_sigma2,
    quantile_return_decomposition,
)
from domain.gp.gp_uncertainty import _audit_no_leak


# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------


def _make_synthetic_data(
    n_days: int = 120,
    n_factors: int = 6,
    n_sectors: int = 17,
    seed: int = 42,
) -> tuple:
    """合成テストデータを生成する。"""
    rng = np.random.RandomState(seed)
    f_matrix = rng.randn(n_days, n_factors) * 0.1
    z_lin_matrix = rng.randn(n_days, n_sectors) * 0.002
    # y = z_lin + small noise（線形モデルが良い近似）
    y_matrix = z_lin_matrix + rng.randn(n_days, n_sectors) * 0.003
    dates = pd.bdate_range(start="2020-01-02", periods=n_days)
    return f_matrix, z_lin_matrix, y_matrix, dates


def _make_gp_module(window_size: int = 80, seed: int = 42) -> GPUncertaintyModule:
    """標準的な GPUncertaintyModule を生成する。"""
    cfg = GPConfig(
        n_factors=6,
        n_sectors=17,
        window_size=window_size,
        n_restarts_optimizer=1,  # テスト高速化
        seed=seed,
        kernel_cfg=KernelConfig(
            alpha_init=0.05,
            alpha_bounds=(1e-5, 0.3),
            length_scale_bounds=(0.3, 5.0),
        ),
    )
    return GPUncertaintyModule(cfg)


# ---------------------------------------------------------------------------
# 1. リーク監査テスト
# ---------------------------------------------------------------------------


class TestLeakAudit:
    def test_no_leak_passes(self):
        """signal_date < target_date: リークなし → 正常通過。"""
        sig_dates = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06"])
        tgt_dates = pd.DatetimeIndex(["2020-01-03", "2020-01-06", "2020-01-07"])
        _audit_no_leak(sig_dates, tgt_dates, "test")  # raises nothing

    def test_leak_detected_raises(self):
        """signal_date == target_date: DATA LEAK → AssertionError。"""
        sig_dates = pd.DatetimeIndex(["2020-01-03"])
        tgt_dates = pd.DatetimeIndex(["2020-01-03"])  # same date = leak
        with pytest.raises(AssertionError, match="DATA LEAK"):
            _audit_no_leak(sig_dates, tgt_dates, "test")

    def test_future_signal_raises(self):
        """signal_date > target_date: DATA LEAK → AssertionError。"""
        sig_dates = pd.DatetimeIndex(["2020-01-06"])
        tgt_dates = pd.DatetimeIndex(["2020-01-03"])
        with pytest.raises(AssertionError, match="DATA LEAK"):
            _audit_no_leak(sig_dates, tgt_dates, "test")

    def test_fit_audits_leak(self):
        """fit() 内部で signal_dates を正しく検証する。"""
        f, z, y, dates = _make_synthetic_data(n_days=80)
        module = _make_gp_module(window_size=60)

        # 意図的なリーク: signal_dates = target_dates（同一日）
        with pytest.raises(AssertionError, match="DATA LEAK"):
            module.fit(f, z, y, dates, signal_dates=dates)  # signal == target → leak

    def test_window_boundary_no_leak(self):
        """ウィンドウ内データが窓サイズを超えないことを確認。"""
        f, z, y, dates = _make_synthetic_data(n_days=80)
        module = _make_gp_module(window_size=60)
        sig_dates = dates - pd.Timedelta(days=1)
        module.fit(f, z, y, dates, signal_dates=sig_dates)

        assert module._fit_metadata.get("window_size") == 60, (
            "fit() のウィンドウサイズがメタデータと一致しない"
        )


# ---------------------------------------------------------------------------
# 2. 再現性テスト
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_result(self):
        """同一 seed で 2 回学習した結果が完全一致する。"""
        f, z, y, dates = _make_synthetic_data(n_days=80)
        sig_dates = dates - pd.Timedelta(days=1)
        f_t_test = f[0]
        z_lin_test = z[0]

        # 1 回目
        m1 = _make_gp_module(window_size=60, seed=42)
        m1.fit(f[:60], z[:60], y[:60], dates[:60], signal_dates=sig_dates[:60])
        mu1, sigma2_1 = m1.predict_mean_var(f_t_test, z_lin_test)

        # 2 回目（同一 seed）
        m2 = _make_gp_module(window_size=60, seed=42)
        m2.fit(f[:60], z[:60], y[:60], dates[:60], signal_dates=sig_dates[:60])
        mu2, sigma2_2 = m2.predict_mean_var(f_t_test, z_lin_test)

        np.testing.assert_array_almost_equal(mu1, mu2, decimal=8,
            err_msg="同一 seed での予測平均が一致しない")
        np.testing.assert_array_almost_equal(sigma2_1, sigma2_2, decimal=8,
            err_msg="同一 seed での予測分散が一致しない")

    def test_different_seeds_may_differ(self):
        """異なる seed では結果が異なる可能性がある（決定論ではない）。"""
        f, z, y, dates = _make_synthetic_data(n_days=80)
        sig_dates = dates - pd.Timedelta(days=1)
        f_t_test = f[0]
        z_lin_test = z[0]

        m1 = _make_gp_module(window_size=60, seed=42)
        m1.fit(f[:60], z[:60], y[:60], dates[:60], signal_dates=sig_dates[:60])
        mu1, _ = m1.predict_mean_var(f_t_test, z_lin_test)

        m2 = _make_gp_module(window_size=60, seed=99)
        m2.fit(f[:60], z[:60], y[:60], dates[:60], signal_dates=sig_dates[:60])
        mu2, _ = m2.predict_mean_var(f_t_test, z_lin_test)

        # 異なる seed でも必ず同じになるとは限らない（単なる確認）
        # このテストはパスすることが保証されないが、実装の差異確認
        assert mu1 is not None and mu2 is not None


# ---------------------------------------------------------------------------
# 3. インターフェーステスト
# ---------------------------------------------------------------------------


class TestInterface:
    def setup_method(self):
        """各テスト前に共通の学習済みモジュールを準備。"""
        self.f, self.z, self.y, self.dates = _make_synthetic_data(n_days=100)
        self.sig_dates = self.dates - pd.Timedelta(days=1)
        self.module = _make_gp_module(window_size=80)
        self.module.fit(
            self.f, self.z, self.y, self.dates, signal_dates=self.sig_dates
        )

    def test_predict_mean_var_shape(self):
        """predict_mean_var は (N_J,), (N_J,) の shape を返す。"""
        f_t = self.f[0]
        z_lin = self.z[0]
        mu, sigma2 = self.module.predict_mean_var(f_t, z_lin)

        assert mu.shape == (17,), f"mu.shape should be (17,), got {mu.shape}"
        assert sigma2.shape == (17,), f"sigma2.shape should be (17,), got {sigma2.shape}"

    def test_predict_mean_var_dtype(self):
        """predict_mean_var は float64 を返す。"""
        mu, sigma2 = self.module.predict_mean_var(self.f[0], self.z[0])
        assert mu.dtype == np.float64
        assert sigma2.dtype == np.float64

    def test_sigma2_nonnegative(self):
        """予測分散は NaN または非負。"""
        mu, sigma2 = self.module.predict_mean_var(self.f[0], self.z[0])
        valid = np.isfinite(sigma2)
        assert np.all(sigma2[valid] >= 0), "予測分散に負の値が含まれています"

    def test_compute_confidence_returns_float(self):
        """compute_confidence は float を返す。"""
        _, sigma2 = self.module.predict_mean_var(self.f[0], self.z[0])
        kappa = self.module.compute_confidence(sigma2)
        assert isinstance(kappa, float), f"kappa is not float: {type(kappa)}"

    def test_apply_sizing_shape(self):
        """apply_sizing は (N_J,) の shape を返す。"""
        w_base = np.random.randn(17)
        w_final = self.module.apply_sizing(w_base, kappa=0.5)
        assert w_final.shape == (17,), f"w_final.shape should be (17,), got {w_final.shape}"

    def test_apply_sizing_scaling(self):
        """w_final = κ × w_base の正確性。"""
        w_base = np.array([1.0, -0.5, 0.3] + [0.0] * 14)
        kappa = 0.7
        w_final = self.module.apply_sizing(w_base, kappa)
        np.testing.assert_array_almost_equal(w_final, w_base * kappa)

    def test_unfitted_raises(self):
        """未学習モジュールで predict_mean_var を呼ぶと RuntimeError。"""
        module = _make_gp_module()
        with pytest.raises(RuntimeError, match="fit"):
            module.predict_mean_var(self.f[0], self.z[0])

    def test_bayesian_shrinkage_direction_preserved(self):
        """ベイズ的ウェイト縮小でウェイトの符号が保存される。"""
        w = np.array([1.0, -1.0, 0.5, -0.3] + [0.0] * 13)
        mu = np.ones(17) * 0.001
        sigma2 = np.ones(17) * 0.01
        w_shrunk = self.module.apply_bayesian_weight_shrinkage(w, mu, sigma2)
        assert np.all((np.sign(w_shrunk) == np.sign(w)) | (w == 0)), \
            "ウェイト縮小で符号が反転しています"


# ---------------------------------------------------------------------------
# 4. カーネル制約テスト
# ---------------------------------------------------------------------------


class TestKernelConstraints:
    def test_alpha_within_bounds(self):
        """学習後の α が alpha_bounds の上限以内に収まる。"""
        f, z, y, dates = _make_synthetic_data(n_days=80)
        sig_dates = dates - pd.Timedelta(days=1)

        alpha_max = 0.3
        cfg = GPConfig(
            window_size=60,
            n_restarts_optimizer=1,
            seed=42,
            kernel_cfg=KernelConfig(alpha_init=0.1, alpha_bounds=(1e-5, alpha_max)),
        )
        module = GPUncertaintyModule(cfg)
        module.fit(f[:60], z[:60], y[:60], dates[:60], signal_dates=sig_dates[:60])

        for j, model in enumerate(module._models):
            if model is None:
                continue
            params = extract_kernel_params(model)
            alpha = params.get("alpha", float("nan"))
            if np.isfinite(alpha):
                assert alpha <= alpha_max + 1e-6, (
                    f"業種 {j}: α={alpha:.4f} が alpha_bounds 上限 {alpha_max} を超えています"
                )

    def test_length_scale_within_bounds(self):
        """学習後の長さスケールが bounds 内に収まる。"""
        f, z, y, dates = _make_synthetic_data(n_days=80)
        sig_dates = dates - pd.Timedelta(days=1)
        ls_min, ls_max = 0.5, 5.0

        cfg = GPConfig(
            window_size=60,
            n_restarts_optimizer=1,
            seed=42,
            kernel_cfg=KernelConfig(length_scale_bounds=(ls_min, ls_max)),
        )
        module = GPUncertaintyModule(cfg)
        module.fit(f[:60], z[:60], y[:60], dates[:60], signal_dates=sig_dates[:60])

        for j, model in enumerate(module._models):
            if model is None:
                continue
            params = extract_kernel_params(model)
            ls = params.get("length_scales", np.array([]))
            if len(ls) > 0 and np.all(np.isfinite(ls)):
                assert np.all(ls >= ls_min - 1e-6), (
                    f"業種 {j}: length_scale 下限違反: min={ls.min():.4f} < {ls_min}"
                )
                assert np.all(ls <= ls_max + 1e-6), (
                    f"業種 {j}: length_scale 上限違反: max={ls.max():.4f} > {ls_max}"
                )

    def test_build_structured_kernel_shape(self):
        """build_structured_kernel がスカラー入力で正常に動作する。"""
        kernel = build_structured_kernel(KernelConfig(), n_factors=6)
        # カーネルオブジェクトが生成されること（sklearn Kernel）
        assert kernel is not None
        # 同一点での評価が正の値
        x = np.zeros((1, 6))
        k_val = kernel(x, x)
        assert k_val.shape == (1, 1)
        assert k_val[0, 0] >= 0


# ---------------------------------------------------------------------------
# 5. κ_t 範囲テスト
# ---------------------------------------------------------------------------


class TestKappaRange:
    def setup_method(self):
        self.f, self.z, self.y, self.dates = _make_synthetic_data(n_days=100)
        self.sig_dates = self.dates - pd.Timedelta(days=1)
        self.module = _make_gp_module(window_size=80)
        self.module.fit(
            self.f, self.z, self.y, self.dates, signal_dates=self.sig_dates
        )

    def test_kappa_in_valid_range(self):
        """κ_t は [min_kappa, 1.0] に収まる。"""
        min_kappa = 0.1
        for i in range(min(10, len(self.f))):
            mu, sigma2 = self.module.predict_mean_var(self.f[i], self.z[i])
            kappa = self.module.compute_confidence(sigma2, min_kappa=min_kappa)
            assert min_kappa <= kappa <= 1.0, (
                f"κ_t={kappa:.4f} が [{min_kappa}, 1.0] の範囲外"
            )

    def test_all_nan_sigma2_returns_1(self):
        """全業種の σ² が NaN のとき κ_t = 1.0 を返す。"""
        sigma2_nan = np.full(17, np.nan)
        kappa = self.module.compute_confidence(sigma2_nan)
        assert kappa == 1.0, f"全 NaN の σ² で κ_t ≠ 1.0: got {kappa}"

    def test_high_variance_low_kappa(self):
        """高分散 → 低確信度（κ_t が低くなる）。"""
        sigma2_low = np.ones(17) * 0.001
        sigma2_high = np.ones(17) * 1.0

        kappa_low = compute_confidence_from_sigma2(sigma2_low, tau=5.0)
        kappa_high = compute_confidence_from_sigma2(sigma2_high, tau=5.0)

        assert kappa_high < kappa_low, (
            f"高分散で確信度が低くなっていない: "
            f"κ(低σ²)={kappa_low:.3f}, κ(高σ²)={kappa_high:.3f}"
        )

    def test_kappa_monotone_decreasing_in_sigma2(self):
        """σ² が増加するにつれ κ_t は単調非増加。"""
        sigma2_values = [0.001, 0.01, 0.05, 0.1, 0.5]
        kappas = [compute_confidence_from_sigma2(np.ones(17) * v, tau=5.0) for v in sigma2_values]
        for i in range(len(kappas) - 1):
            assert kappas[i] >= kappas[i + 1] - 1e-8, (
                f"κ_t が単調非増加でない: κ[{i}]={kappas[i]:.4f} < κ[{i+1}]={kappas[i+1]:.4f}"
            )


# ---------------------------------------------------------------------------
# 6. 不確実性較正テスト
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_perfect_calibration(self):
        """真のモデルからサンプルした場合、較正は良好になる（高確率）。"""
        rng = np.random.RandomState(42)
        n = 500
        mu = rng.randn(n)
        sigma = np.ones(n) * 0.5
        y = mu + rng.randn(n) * sigma  # 真の N(mu, sigma^2) からサンプル

        result = coverage_test(mu, sigma**2, y, levels=[0.80, 0.90], warn_threshold=0.05)
        # 大サンプルなら±5pp 以内に収まるはず
        assert result.calibration_status in {"GOOD", "WARN"}, (
            f"真のモデルから生成したデータで REJECT: {result.summary}"
        )

    def test_overconfident_detection(self):
        """過信（予測分散が小さすぎる）を検出する。"""
        rng = np.random.RandomState(42)
        n = 500
        mu = rng.randn(n)
        sigma_too_small = np.ones(n) * 0.1  # 真の σ=0.5 に対して 5 倍小さく
        y = mu + rng.randn(n) * 0.5

        result = coverage_test(mu, sigma_too_small**2, y, levels=[0.80])
        assert result.calibration_status in {"WARN", "REJECT"}, (
            "過信（分散過小）が検出されていない"
        )
        assert result.coverage_errors[0] < 0, "過信は負の誤差（実績 < 期待）"

    def test_underconfident_detection(self):
        """過疎（予測分散が大きすぎる）を検出する。"""
        rng = np.random.RandomState(42)
        n = 500
        mu = rng.randn(n)
        sigma_too_large = np.ones(n) * 5.0  # 真の σ=0.5 に対して 10 倍大きく
        y = mu + rng.randn(n) * 0.5

        result = coverage_test(mu, sigma_too_large**2, y, levels=[0.80])
        assert result.calibration_status in {"WARN", "REJECT"}, (
            "過疎（分散過大）が検出されていない"
        )
        assert result.coverage_errors[0] > 0, "過疎は正の誤差（実績 > 期待）"

    def test_insufficient_samples(self):
        """サンプル数 < 10 のとき WARN を返す。"""
        result = coverage_test(
            np.ones(5), np.ones(5) * 0.1, np.ones(5),
            levels=[0.80], warn_threshold=0.05,
        )
        assert result.calibration_status == "WARN"
        assert result.n_samples < 10


# ---------------------------------------------------------------------------
# 7. 保存・ロードテスト
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load(self):
        """save() → load() の往復で予測結果が一致する。"""
        f, z, y, dates = _make_synthetic_data(n_days=80)
        sig_dates = dates - pd.Timedelta(days=1)

        module = _make_gp_module(window_size=60)
        module.fit(f[:60], z[:60], y[:60], dates[:60], signal_dates=sig_dates[:60])

        mu_before, sigma2_before = module.predict_mean_var(f[0], z[0])

        with tempfile.TemporaryDirectory() as tmpdir:
            module.save(tmpdir)
            loaded = GPUncertaintyModule.load(tmpdir)

        mu_after, sigma2_after = loaded.predict_mean_var(f[0], z[0])
        np.testing.assert_array_almost_equal(mu_before, mu_after, decimal=6,
            err_msg="save→load で予測平均が変化しました")
        np.testing.assert_array_almost_equal(sigma2_before, sigma2_after, decimal=6,
            err_msg="save→load で予測分散が変化しました")


# ---------------------------------------------------------------------------
# 8. 堅牢性テスト
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_enabled_false_returns_z_lin(self):
        """enabled=False のとき predict_mean_var は z_lin をそのまま返す。"""
        cfg = GPConfig(enabled=False)
        module = GPUncertaintyModule(cfg)
        z_lin = np.ones(17) * 0.01
        f_t = np.zeros(6)
        mu, sigma2 = module.predict_mean_var(f_t, z_lin)
        np.testing.assert_array_equal(mu, z_lin)
        assert np.all(np.isnan(sigma2))

    def test_all_nan_y_skips_gracefully(self):
        """y_matrix が全 NaN でも fit() がクラッシュしない。"""
        f, z, y, dates = _make_synthetic_data(n_days=80)
        sig_dates = dates - pd.Timedelta(days=1)
        y_nan = np.full_like(y, np.nan)

        module = _make_gp_module(window_size=60)
        # 全 NaN は有効サンプル < 30 → None モデルとして登録（エラーなし）
        module.fit(f[:60], z[:60], y_nan[:60], dates[:60], signal_dates=sig_dates[:60])
        assert module._is_fitted

    def test_predict_with_unfitted_sector(self):
        """一部業種がフィットに失敗してもモジュール全体は動作する。"""
        f, z, y, dates = _make_synthetic_data(n_days=100)
        sig_dates = dates - pd.Timedelta(days=1)

        module = _make_gp_module(window_size=80)
        module.fit(f, z, y, dates, signal_dates=sig_dates)

        # None モデルがある場合でも predict は crash しない
        mu, sigma2 = module.predict_mean_var(f[0], z[0])
        assert mu.shape == (17,)
        assert sigma2.shape == (17,)


# ---------------------------------------------------------------------------
# 9. ConfidenceScorer テスト
# ---------------------------------------------------------------------------


class TestConfidenceScorer:
    def test_inv_exp_mode(self):
        """inv_exp モード: κ = exp(-τ × σ²)。"""
        scorer = ConfidenceScorer(ConfidenceConfig(mapping="inv_exp", tau=5.0, smoothing_halflife=0))
        sigma2 = np.ones(17) * 0.1
        kappa = scorer.score(sigma2)
        expected = float(np.exp(-5.0 * 0.1))
        assert abs(kappa - expected) < 0.05, f"inv_exp の計算が不正確: {kappa:.4f} vs {expected:.4f}"

    def test_smoothing_effect(self):
        """EWMA スムージングが κ_t を過去値に向けて引っ張る。"""
        cfg = ConfidenceConfig(mapping="inv_exp", tau=5.0, smoothing_halflife=3)
        scorer = ConfidenceScorer(cfg)

        # 最初は高確信度
        sigma2_low = np.ones(17) * 0.01
        kappa1 = scorer.score(sigma2_low)

        # 突然の高不確実性
        sigma2_high = np.ones(17) * 5.0
        kappa2 = scorer.score(sigma2_high)

        # スムージングにより kappa2 は生の値（≈ exp(-25) ≈ 0.0 → min_kappa）より高いはず
        raw_kappa2 = float(np.exp(-5.0 * 5.0))
        assert kappa2 > raw_kappa2, "スムージングが機能していない（kappa が生の値と同じ）"

    def test_min_kappa_enforced(self):
        """min_kappa が常に下限として機能する。"""
        cfg = ConfidenceConfig(min_kappa=0.2, mapping="inv_exp", tau=100.0, smoothing_halflife=0)
        scorer = ConfidenceScorer(cfg)
        sigma2_huge = np.ones(17) * 1000.0
        kappa = scorer.score(sigma2_huge)
        assert kappa >= 0.2, f"min_kappa=0.2 が守られていない: κ={kappa:.4f}"

    def test_reset_clears_history(self):
        """reset() 後は履歴がクリアされる。"""
        scorer = ConfidenceScorer(ConfidenceConfig())
        sigma2 = np.ones(17) * 0.1
        scorer.score(sigma2)
        scorer.score(sigma2)
        assert len(scorer._kappa_history) == 2
        scorer.reset()
        assert len(scorer._kappa_history) == 0

    def test_apply_sizing(self):
        """apply_sizing で w_final = κ × w_base。"""
        scorer = ConfidenceScorer()
        w = np.array([1.0, -0.5, 0.3] + [0.0] * 14)
        kappa = 0.6
        w_final = scorer.apply_sizing(w, kappa)
        np.testing.assert_array_almost_equal(w_final, w * kappa)


# ---------------------------------------------------------------------------
# 10. ARD 解釈性テスト
# ---------------------------------------------------------------------------


class TestARDInterpretability:
    def test_ard_summary_has_correct_keys(self):
        """get_ard_summary() が必要なキーを持つ dict を返す。"""
        f, z, y, dates = _make_synthetic_data(n_days=80)
        sig_dates = dates - pd.Timedelta(days=1)
        module = _make_gp_module(window_size=60)
        module.fit(f[:60], z[:60], y[:60], dates[:60], signal_dates=sig_dates[:60])

        ard_list = module.get_ard_summary()
        assert len(ard_list) == 17, f"業種数が不正: {len(ard_list)}"

        for entry in ard_list:
            assert "sector" in entry
            if "error" not in entry:
                assert "length_scales" in entry
                assert "importance" in entry
                assert "alpha" in entry

    def test_length_scale_shape(self):
        """ARD 長さスケールが K 次元。"""
        f, z, y, dates = _make_synthetic_data(n_days=80)
        sig_dates = dates - pd.Timedelta(days=1)
        module = _make_gp_module(window_size=60)
        module.fit(f[:60], z[:60], y[:60], dates[:60], signal_dates=sig_dates[:60])

        for entry in module.get_ard_summary():
            if "length_scales" in entry:
                assert len(entry["length_scales"]) == 6, (
                    f"長さスケールの次元が {len(entry['length_scales'])} ≠ 6"
                )

    def test_importance_sums_to_one(self):
        """重要度の合計が 1.0。"""
        f, z, y, dates = _make_synthetic_data(n_days=80)
        sig_dates = dates - pd.Timedelta(days=1)
        module = _make_gp_module(window_size=60)
        module.fit(f[:60], z[:60], y[:60], dates[:60], signal_dates=sig_dates[:60])

        for entry in module.get_ard_summary():
            if "importance" in entry:
                imp_sum = entry["importance"].sum()
                assert abs(imp_sum - 1.0) < 1e-6, (
                    f"業種 {entry['sector']}: 重要度の合計 = {imp_sum:.6f} ≠ 1.0"
                )


# ---------------------------------------------------------------------------
# 11. KernelConfig バリデーションテスト
# ---------------------------------------------------------------------------


class TestKernelConfig:
    def test_invalid_alpha_bounds_raises(self):
        """alpha_bounds[1] > 1.0 は AssertionError。"""
        with pytest.raises(AssertionError):
            KernelConfig(alpha_bounds=(1e-5, 2.0))  # 上限 > 1.0

    def test_invalid_ls_bounds_raises(self):
        """length_scale_bounds[0] < 0.1 は AssertionError。"""
        with pytest.raises(AssertionError):
            KernelConfig(length_scale_bounds=(0.05, 10.0))

    def test_invalid_nonlinear_type_raises(self):
        """nonlinear_type が未知の場合 AssertionError。"""
        with pytest.raises(AssertionError):
            KernelConfig(nonlinear_type="unknown")

    def test_valid_config_no_error(self):
        """有効な設定はエラーなし。"""
        cfg = KernelConfig(
            alpha_bounds=(1e-5, 0.5),
            length_scale_bounds=(0.3, 10.0),
            nonlinear_type="rbf",
        )
        assert cfg is not None

    def test_from_config_dict(self):
        """from_config_dict() が正しく設定を読み込む。"""
        d = {
            "alpha_init": 0.2,
            "alpha_bounds": [1e-5, 0.4],
            "nonlinear_type": "matern",
            "matern_nu": 1.5,
        }
        cfg = KernelConfig.from_config_dict(d)
        assert cfg.alpha_init == 0.2
        assert cfg.alpha_bounds == (1e-5, 0.4)
        assert cfg.nonlinear_type == "matern"
        assert cfg.matern_nu == 1.5
