from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass
class Portfolio:
    """A weighted multi-asset portfolio.

    Attributes
    ----------
    S0:
        Initial asset prices, shape ``(n_assets,)``.
    weights:
        Portfolio weights per asset, shape ``(n_assets,)``.
        Must sum to 1.0 (within floating-point tolerance).
    """

    S0: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        if self.S0.shape != self.weights.shape:
            raise ValueError(
                f"S0 and weights must have the same shape, "
                f"got {self.S0.shape} and {self.weights.shape}"
            )
        if not np.isclose(self.weights.sum(), 1.0):
            raise ValueError(
                f"Portfolio weights must sum to 1.0, got {self.weights.sum():.6f}"
            )

    @cached_property
    def initial_value(self) -> float:
        """Total initial portfolio value: ``weights @ S0``."""
        return float(self.weights @ self.S0)
