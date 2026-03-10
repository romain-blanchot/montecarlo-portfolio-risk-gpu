import pytest

from portfolio_risk_engine.domain.services.cholesky import cholesky


def _matmul_LLT(L: tuple[tuple[float, ...], ...]) -> list[list[float]]:
    """Compute L @ L^T for verification."""
    n = len(L)
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            result[i][j] = sum(L[i][k] * L[j][k] for k in range(n))
    return result


class TestCholesky1x1:
    def test_single_element(self):
        result = cholesky(((4.0,),))
        assert result[0][0] == pytest.approx(2.0)

    def test_reconstructs_original(self):
        matrix = ((9.0,),)
        L = cholesky(matrix)
        reconstructed = _matmul_LLT(L)
        assert reconstructed[0][0] == pytest.approx(9.0)


class TestCholesky2x2:
    def test_known_result(self):
        # [[4, 2], [2, 5]] -> L = [[2, 0], [1, 2]]
        matrix = ((4.0, 2.0), (2.0, 5.0))
        L = cholesky(matrix)
        assert L[0][0] == pytest.approx(2.0)
        assert L[0][1] == pytest.approx(0.0)
        assert L[1][0] == pytest.approx(1.0)
        assert L[1][1] == pytest.approx(2.0)

    def test_reconstructs_original(self):
        matrix = ((4.0, 2.0), (2.0, 5.0))
        L = cholesky(matrix)
        reconstructed = _matmul_LLT(L)
        for i in range(2):
            for j in range(2):
                assert reconstructed[i][j] == pytest.approx(matrix[i][j])


class TestCholesky3x3:
    def test_identity_matrix(self):
        identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        L = cholesky(identity)
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert L[i][j] == pytest.approx(expected)

    def test_reconstructs_original(self):
        matrix = (
            (25.0, 15.0, -5.0),
            (15.0, 18.0, 0.0),
            (-5.0, 0.0, 11.0),
        )
        L = cholesky(matrix)
        reconstructed = _matmul_LLT(L)
        for i in range(3):
            for j in range(3):
                assert reconstructed[i][j] == pytest.approx(matrix[i][j])

    def test_lower_triangular(self):
        matrix = (
            (25.0, 15.0, -5.0),
            (15.0, 18.0, 0.0),
            (-5.0, 0.0, 11.0),
        )
        L = cholesky(matrix)
        for i in range(3):
            for j in range(i + 1, 3):
                assert L[i][j] == pytest.approx(0.0)


class TestCholeskyValidation:
    def test_empty_matrix_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            cholesky(())

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            cholesky(((1.0, 2.0),))

    def test_not_positive_definite_raises(self):
        # Diagonal with zero
        with pytest.raises(ValueError, match="not positive definite"):
            cholesky(((0.0,),))

    def test_negative_diagonal_raises(self):
        with pytest.raises(ValueError, match="not positive definite"):
            cholesky(((-1.0,),))

    def test_indefinite_matrix_raises(self):
        # [[1, 2], [2, 1]] has eigenvalue -1
        with pytest.raises(ValueError, match="not positive definite"):
            cholesky(((1.0, 2.0), (2.0, 1.0)))


class TestCholeskyDiagonal:
    def test_diagonal_matrix(self):
        matrix = ((4.0, 0.0), (0.0, 9.0))
        L = cholesky(matrix)
        assert L[0][0] == pytest.approx(2.0)
        assert L[1][1] == pytest.approx(3.0)
        assert L[0][1] == pytest.approx(0.0)
        assert L[1][0] == pytest.approx(0.0)
