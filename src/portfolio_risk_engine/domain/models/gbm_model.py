from dataclasses import dataclass

from portfolio_risk_engine.domain.models.market_parameters import MarketParameters


@dataclass(frozen=True)
class MultivariateGBM:
    """Multivariate Geometric Brownian Motion model with pre-computed Cholesky factor."""

    market_parameters: MarketParameters
    cholesky_factor: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        n = len(self.market_parameters.tickers)

        if len(self.cholesky_factor) != n:
            raise ValueError(
                f"cholesky_factor rows ({len(self.cholesky_factor)}) "
                f"must match number of tickers ({n})."
            )

        for i, row in enumerate(self.cholesky_factor):
            if len(row) != n:
                raise ValueError(
                    f"cholesky_factor row {i} has {len(row)} columns, expected {n}."
                )
