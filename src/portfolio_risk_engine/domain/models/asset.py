from dataclasses import dataclass
from portfolio_risk_engine.domain.value_objects.ticker import Ticker
from portfolio_risk_engine.domain.value_objects.currency import Currency

@dataclass(frozen=True)
class Asset:
    ticker: Ticker
    currency: Currency
    name: str | None = None