from dataclasses import dataclass

from portfolio_risk_engine.domain.value_objects.ticker import Ticker


@dataclass(frozen=True)
class MarketParameters:
    tickers: tuple[Ticker, ...]
    drift_vector: tuple[float, ...]
    covariance_matrix: tuple[tuple[float, ...], ...]
    annualization_factor: int

    def __post_init__(self) -> None:
        n = len(self.tickers)

        if n == 0:
            raise ValueError("MarketParameters must contain at least one ticker.")

        if len(self.drift_vector) != n:
            raise ValueError(
                f"drift_vector length ({len(self.drift_vector)}) "
                f"must match number of tickers ({n})."
            )

        if len(self.covariance_matrix) != n:
            raise ValueError("covariance_matrix must be n x n.")

        for row in self.covariance_matrix:
            if len(row) != n:
                raise ValueError("covariance_matrix must be n x n.")

        if self.annualization_factor <= 0:
            raise ValueError("annualization_factor must be positive.")
