import numpy as np

from portfolio_risk_engine.domain.var import compute_var


def compute_es(losses: np.ndarray, confidence: float = 0.95) -> float:
    """Compute Expected Shortfall (CVaR) at the given confidence level.

    ES is the mean of losses that exceed the VaR at ``confidence``.
    It answers: *given that we are in the worst ``(1 - confidence)``
    fraction of outcomes, what is the average loss?*

    Parameters
    ----------
    losses:
        1-D array of simulated portfolio losses.
    confidence:
        Confidence level in ``(0, 1)``.  Default is ``0.95``.

    Returns
    -------
    float
        ES estimate.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    var = compute_var(losses, confidence)
    tail = losses[losses >= var]
    return float(tail.mean())
