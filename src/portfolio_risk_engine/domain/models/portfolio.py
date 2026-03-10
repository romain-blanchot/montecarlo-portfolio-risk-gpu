from dataclasses import dataclass

from portfolio_risk_engine.domain.models.position import Position
from portfolio_risk_engine.domain.value_objects.ticker import Ticker


@dataclass(frozen=True)
class Portfolio:
    positions: tuple[Position, ...]

    def __post_init__(self) -> None:
        if not self.positions:
            raise ValueError("Portfolio must contain at least one position.")

        tickers = [position.asset.ticker for position in self.positions]
        if len(tickers) != len(set(tickers)):
            raise ValueError("Portfolio contains duplicate tickers.")

        total_weight = sum(position.weight.value for position in self.positions)
        if abs(total_weight - 1.0) > 1e-8:
            raise ValueError("Portfolio weights must sum to 1.")

    @property
    def tickers(self) -> list[Ticker]:
        return [position.asset.ticker for position in self.positions]

    @property
    def weights(self) -> list[float]:
        return [position.weight.value for position in self.positions]
