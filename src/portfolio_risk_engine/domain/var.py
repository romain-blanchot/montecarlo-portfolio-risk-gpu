import numpy as np


def compute_var(losses: np.ndarray, confidence: float = 0.95) -> float:
    """Compute Value at Risk (VaR) at the given confidence level.

    VaR is the ``confidence``-th quantile of the loss distribution —
    i.e. the loss that is not exceeded with probability ``confidence``.

    Parameters
    ----------
    losses:
        1-D array of simulated portfolio losses.
    confidence:
        Confidence level in ``(0, 1)``.  Default is ``0.95``.

    Returns
    -------
    float
        VaR estimate.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return float(np.quantile(losses, confidence))
