from typing import Protocol

from portfolio_risk_engine.domain.models.asset import Asset
from portfolio_risk_engine.domain.models.historical_prices import HistoricalPrices
from portfolio_risk_engine.domain.value_objects.date_range import DateRange
from portfolio_risk_engine.domain.value_objects.ticker import Ticker


class MarketDataProvider(Protocol):
    def get_asset(self, ticker: Ticker) -> Asset: ...

    def get_historical_prices(
        self,
        tickers: tuple[Ticker, ...],
        date_range: DateRange,
    ) -> HistoricalPrices: ...
