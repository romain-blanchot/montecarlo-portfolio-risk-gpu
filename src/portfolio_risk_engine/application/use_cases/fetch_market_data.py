from portfolio_risk_engine.domain.models.historical_prices import HistoricalPrices
from portfolio_risk_engine.domain.ports.market_data_provider import MarketDataProvider
from portfolio_risk_engine.domain.value_objects.date_range import DateRange
from portfolio_risk_engine.domain.value_objects.ticker import Ticker


class FetchMarketData:
    def __init__(self, market_data_provider: MarketDataProvider) -> None:
        self._market_data_provider = market_data_provider

    def execute(
        self,
        tickers: tuple[Ticker, ...],
        date_range: DateRange,
    ) -> HistoricalPrices:
        if not tickers:
            raise ValueError("Tickers list cannot be empty.")

        return self._market_data_provider.get_historical_prices(
            tickers=tickers,
            date_range=date_range,
        )
