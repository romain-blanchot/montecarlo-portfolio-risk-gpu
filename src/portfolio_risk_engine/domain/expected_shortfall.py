import numpy as np

from portfolio_risk_engine.domain.var import compute_var


def compute_es(losses: np.ndarray, confidence: float = 0.95) -> float:
    """Return the Expected Shortfall (CVaR) at the given confidence level.

    ES is the mean of losses that exceed VaR(confidence). It answers:
    "given that we are in the worst (1 - confidence) of outcomes,
    what is the average loss?" More conservative than VaR alone.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    var = compute_var(losses, confidence)
    tail = losses[losses >= var]
    return float(tail.mean())
