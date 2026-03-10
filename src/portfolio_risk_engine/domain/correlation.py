import numpy as np


def compute_cholesky(corr_matrix: np.ndarray) -> np.ndarray:
    """Return the lower-triangular Cholesky factor of a correlation matrix.

    Parameters
    ----------
    corr_matrix:
        Square correlation matrix of shape ``(n, n)``.

    Returns
    -------
    np.ndarray
        Lower-triangular matrix ``L`` such that ``L @ L.T == corr_matrix``,
        shape ``(n, n)``.

    Raises
    ------
    ValueError
        If the matrix is not square, not symmetric, does not have ones on
        the diagonal, or is not positive semi-definite.
    """
    if corr_matrix.ndim != 2 or corr_matrix.shape[0] != corr_matrix.shape[1]:
        raise ValueError(
            f"Correlation matrix must be square 2-D, got shape {corr_matrix.shape}"
        )

    if not np.allclose(corr_matrix, corr_matrix.T):
        raise ValueError("Correlation matrix must be symmetric")

    if not np.allclose(np.diag(corr_matrix), 1.0):
        raise ValueError("Correlation matrix must have ones on the diagonal")

    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    if np.any(eigenvalues < -1e-8):
        raise ValueError(
            f"Correlation matrix must be positive semi-definite "
            f"(smallest eigenvalue: {eigenvalues.min():.6g})"
        )

    return np.linalg.cholesky(corr_matrix)
