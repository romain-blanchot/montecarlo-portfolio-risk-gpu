from dataclasses import dataclass

import numpy as np


@dataclass
class MarketModel:
    """GBM parameters for a multi-asset simulation.

    mu      : annualised drift per asset, shape (n_assets,)
    sigma   : annualised volatility per asset, shape (n_assets,)
    dt      : length of one time-step in years (e.g. 1/252 for daily)
    n_steps : number of steps per path
    """

    mu: np.ndarray
    sigma: np.ndarray
    dt: float
    n_steps: int

    def __post_init__(self) -> None:
        if self.mu.shape != self.sigma.shape:
            raise ValueError(
                f"mu and sigma must have the same shape, "
                f"got {self.mu.shape} and {self.sigma.shape}"
            )
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {self.n_steps}")
