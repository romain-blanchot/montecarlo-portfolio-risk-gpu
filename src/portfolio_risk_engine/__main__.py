from portfolio_risk_engine.infrastructure.market_data.yahoo_finance_market_data_provider import (
    YahooFinanceMarketDataProvider,
)
from portfolio_risk_engine.domain.value_objects.ticker import Ticker


def main() -> None:
    provider = YahooFinanceMarketDataProvider()

    asset = provider.get_asset(Ticker("AAPL"))

    print("Ticker:", asset.ticker.value)
    print("Currency:", asset.currency.code)
    print("Name:", asset.name)


if __name__ == "__main__":
    main()
