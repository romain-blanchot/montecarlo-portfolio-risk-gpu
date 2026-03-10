import math


def cholesky(
    matrix: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    """Pure-Python Cholesky decomposition. Returns lower-triangular L such that L @ L^T == matrix."""
    n = len(matrix)
    if n == 0:
        raise ValueError("Matrix must be non-empty.")

    for row in matrix:
        if len(row) != n:
            raise ValueError("Matrix must be square.")

    lower: list[list[float]] = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            s = sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                val = matrix[i][i] - s
                if val <= 0:
                    raise ValueError("Matrix is not positive definite.")
                lower[i][j] = math.sqrt(val)
            else:
                lower[i][j] = (matrix[i][j] - s) / lower[j][j]

    return tuple(tuple(row) for row in lower)
