from dataclasses import dataclass

from portfolio_risk_engine.domain.value_objects.ticker import Ticker


@dataclass(frozen=True)
class HestonAssetParams:
    """Per-asset Heston stochastic volatility parameters."""

    kappa: float  # mean-reversion speed
    theta: float  # long-run variance
    xi: float  # vol-of-vol
    rho: float  # leverage (return-variance correlation)
    v0: float  # initial variance

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise ValueError("kappa must be positive.")
        if self.theta <= 0:
            raise ValueError("theta must be positive.")
        if self.xi <= 0:
            raise ValueError("xi must be positive.")
        if not (-1.0 < self.rho < 1.0):
            raise ValueError("rho must be in (-1, 1).")
        if self.v0 <= 0:
            raise ValueError("v0 must be positive.")

    @property
    def feller_satisfied(self) -> bool:
        """2*kappa*theta > xi^2 ensures variance stays positive."""
        return 2 * self.kappa * self.theta > self.xi**2


@dataclass(frozen=True)
class HestonModel:
    """Multi-asset Heston stochastic volatility model."""

    tickers: tuple[Ticker, ...]
    drift_vector: tuple[float, ...]
    asset_params: tuple[HestonAssetParams, ...]
    correlation_cholesky: tuple[tuple[float, ...], ...]
    annualization_factor: int

    def __post_init__(self) -> None:
        n = len(self.tickers)

        if n == 0:
            raise ValueError("Must contain at least one ticker.")

        if len(self.drift_vector) != n:
            raise ValueError(
                f"drift_vector length ({len(self.drift_vector)}) "
                f"must match number of tickers ({n})."
            )

        if len(self.asset_params) != n:
            raise ValueError(
                f"asset_params length ({len(self.asset_params)}) "
                f"must match number of tickers ({n})."
            )

        if len(self.correlation_cholesky) != n:
            raise ValueError("correlation_cholesky must be n x n.")

        for row in self.correlation_cholesky:
            if len(row) != n:
                raise ValueError("correlation_cholesky must be n x n.")

        if self.annualization_factor <= 0:
            raise ValueError("annualization_factor must be positive.")
