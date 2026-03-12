from dataclasses import dataclass

from portfolio_risk_engine.domain.models.market_parameters import MarketParameters


@dataclass(frozen=True)
class StudentTGBM:
    """Multivariate GBM with Student-t distributed innovations for fat tails."""

    market_parameters: MarketParameters
    cholesky_factor: tuple[tuple[float, ...], ...]
    degrees_of_freedom: float

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

        if self.degrees_of_freedom <= 2.0:
            raise ValueError("degrees_of_freedom must be > 2 for finite variance.")
