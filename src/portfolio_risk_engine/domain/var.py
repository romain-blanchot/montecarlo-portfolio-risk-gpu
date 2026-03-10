import numpy as np


def compute_var(losses: np.ndarray, confidence: float = 0.95) -> float:
    """Return the Value at Risk at the given confidence level.

    VaR is the confidence-th quantile of the loss distribution:
    "we will not lose more than VaR with probability confidence."
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return float(np.quantile(losses, confidence))
