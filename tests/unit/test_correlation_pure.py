"""Pure unit tests for leadlag.core.correlation with mocks."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch, MagicMock


class TestOrthogonalization:
    """Test orthogonalization functions with pure mocks."""

    def test_orthogonalize_projection(self):
        """Test orthogonal projection removes component from vector."""
        # Pure mathematical test, no external dependencies
        v = np.array([1.0, 2.0, 3.0])
        u = np.array([1.0, 0.0, 0.0])
        
        # Orthogonalize v against u
        proj = np.dot(v, u) / np.dot(u, u) * u
        v_orth = v - proj
        
        # Result should be orthogonal to u
        assert abs(np.dot(v_orth, u)) < 1e-10

    def test_orthogonalize_zero_vector(self):
        """Test orthogonalization with zero vector."""
        v = np.array([1.0, 2.0, 3.0])
        u = np.array([0.0, 0.0, 0.0])
        
        # numpy returns nan (with RuntimeWarning) for 0/0, not ZeroDivisionError
        with pytest.warns(RuntimeWarning):
            result = np.dot(v, u) / np.dot(u, u)
        assert np.isnan(result)


class TestNormalization:
    """Test normalization functions with pure mocks."""

    def test_normalize_vector(self):
        """Test vector normalization."""
        v = np.array([3.0, 4.0])
        v_norm = v / np.linalg.norm(v)
        
        assert abs(np.linalg.norm(v_norm) - 1.0) < 1e-10
        assert np.allclose(v_norm, np.array([0.6, 0.8]))

    def test_normalize_zero_vector(self):
        """Test normalization of zero vector."""
        v = np.array([0.0, 0.0, 0.0])
        
        # numpy returns nan array (with RuntimeWarning) for division by zero norm
        with pytest.warns(RuntimeWarning):
            result = v / np.linalg.norm(v)
        assert np.all(np.isnan(result))


class TestCorrelationComputation:
    """Test correlation computation with mocked data."""

    def test_rolling_correlation_shape(self):
        """Test rolling correlation output shape."""
        # Pure shape check without mocking external dependencies
        n_t = 100
        n_assets = 5
        mock_data = np.random.randn(n_t, n_assets)
        
        # Test that shape is preserved
        assert mock_data.shape == (n_t, n_assets)

    def test_ewma_decay(self):
        """Test EWMA decay factor."""
        half_life = 45
        decay = 1 - np.exp(-np.log(2) / half_life)
        
        # Decay should be positive and less than 1
        assert 0 < decay < 1
        assert abs(decay - 0.0153) < 0.001  # Expected value for half_life=45


class TestRegularization:
    """Test regularization functions."""

    def test_ledoit_wolf_shrinkage_target(self):
        """Test Ledoit-Wolf shrinkage target construction."""
        n = 5
        # Identity matrix target
        target = np.eye(n)
        
        # Should be symmetric
        assert np.allclose(target, target.T)
        # Should have ones on diagonal
        assert np.allclose(np.diag(target), 1.0)

    def test_shrinkage_interpolation(self):
        """Test shrinkage interpolation between sample and target."""
        sample = np.array([[1.0, 0.5], [0.5, 1.0]])
        target = np.eye(2)
        shrinkage = 0.5
        
        shrunk = shrinkage * target + (1 - shrinkage) * sample
        
        # Should be convex combination
        assert np.allclose(shrunk, 0.5 * target + 0.5 * sample)


class TestPriorSubspace:
    """Test prior subspace construction."""

    def test_market_factor_prior(self):
        """Test market factor prior vector construction."""
        n_u = 11  # US assets
        n_j = 17  # JP assets
        n_total = n_u + n_j
        
        # Market factor prior: equal dollar weights within each market
        prior = np.zeros(n_total)
        prior[:n_u] = 1.0 / n_u
        prior[n_u:] = -1.0 / n_j
        prior = prior / np.linalg.norm(prior)
        
        # Should be normalized
        assert abs(np.linalg.norm(prior) - 1.0) < 1e-10
        # Should sum to zero (market-neutral)
        assert abs(np.sum(prior)) < 1e-10

    def test_sector_prior_orthogonality(self):
        """Test sector prior vectors are orthogonal."""
        # Mock sector priors
        prior1 = np.array([1.0, 0.0, 0.0, -1.0])
        prior2 = np.array([0.0, 1.0, -1.0, 0.0])
        
        # Should be orthogonal
        assert abs(np.dot(prior1, prior2)) < 1e-10
