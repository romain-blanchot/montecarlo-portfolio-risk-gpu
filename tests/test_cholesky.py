import numpy as np
import pytest

from portfolio_risk_engine.domain.correlation import compute_cholesky


class TestComputeCholesky:
    def test_identity_matrix_returns_identity(self) -> None:
        identity = np.eye(3)
        L = compute_cholesky(identity)
        np.testing.assert_allclose(L, np.eye(3))

    def test_valid_2x2_reconstruction(self) -> None:
        corr = np.array([[1.0, 0.6], [0.6, 1.0]])
        L = compute_cholesky(corr)
        np.testing.assert_allclose(L @ L.T, corr, atol=1e-12)

    def test_valid_3x3_reconstruction(self) -> None:
        corr = np.array(
            [
                [1.0, 0.4, 0.2],
                [0.4, 1.0, 0.5],
                [0.2, 0.5, 1.0],
            ]
        )
        L = compute_cholesky(corr)
        np.testing.assert_allclose(L @ L.T, corr, atol=1e-12)

    def test_lower_triangular(self) -> None:
        corr = np.array([[1.0, 0.3], [0.3, 1.0]])
        L = compute_cholesky(corr)
        # Upper triangle (excluding diagonal) must be zero
        assert L[0, 1] == 0.0

    def test_non_square_raises(self) -> None:
        with pytest.raises(ValueError, match="square"):
            compute_cholesky(np.ones((2, 3)))

    def test_non_symmetric_raises(self) -> None:
        asymmetric = np.array([[1.0, 0.5], [0.3, 1.0]])
        with pytest.raises(ValueError, match="symmetric"):
            compute_cholesky(asymmetric)

    def test_diagonal_not_ones_raises(self) -> None:
        bad_diag = np.array([[2.0, 0.5], [0.5, 1.0]])
        with pytest.raises(ValueError, match="diagonal"):
            compute_cholesky(bad_diag)

    def test_non_psd_raises(self) -> None:
        # Off-diagonal > 1 forces a negative eigenvalue
        non_psd = np.array([[1.0, 1.5], [1.5, 1.0]])
        with pytest.raises(ValueError):
            compute_cholesky(non_psd)
