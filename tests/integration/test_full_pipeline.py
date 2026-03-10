from datetime import date

import pytest

from portfolio_risk_engine.application.use_cases.compute_log_returns import (
    ComputeLogReturns,
)
from portfolio_risk_engine.application.use_cases.estimate_market_parameters import (
    EstimateMarketParameters,
)
from portfolio_risk_engine.application.use_cases.fetch_market_data import (
    FetchMarketData,
)
from portfolio_risk_engine.domain.value_objects.date_range import DateRange
from portfolio_risk_engine.domain.value_objects.ticker import Ticker
from portfolio_risk_engine.infrastructure.market_data.yahoo_finance_market_data_provider import (
    YahooFinanceMarketDataProvider,
)

AAPL = Ticker("AAPL")
MSFT = Ticker("MSFT")
DATE_RANGE = DateRange(start=date(2024, 1, 1), end=date(2024, 3, 1))


@pytest.mark.integration
class TestFullPipeline:
    def setup_method(self):
        provider = YahooFinanceMarketDataProvider()
        self.fetch = FetchMarketData(provider)
        self.compute_returns = ComputeLogReturns()
        self.estimate_params = EstimateMarketParameters()

    def test_pipeline_produces_market_parameters(self):
        prices = self.fetch.execute(tickers=(AAPL, MSFT), date_range=DATE_RANGE)
        returns = self.compute_returns.execute(prices)
        params = self.estimate_params.execute(returns)

        assert params.tickers == (AAPL, MSFT)
        assert len(params.drift_vector) == 2
        assert len(params.covariance_matrix) == 2
        assert len(params.covariance_matrix[0]) == 2

    def test_returns_have_one_less_date_than_prices(self):
        prices = self.fetch.execute(tickers=(AAPL, MSFT), date_range=DATE_RANGE)
        returns = self.compute_returns.execute(prices)

        assert len(returns.dates) == len(prices.dates) - 1

    def test_returns_tickers_match_prices(self):
        prices = self.fetch.execute(tickers=(AAPL, MSFT), date_range=DATE_RANGE)
        returns = self.compute_returns.execute(prices)

        assert returns.tickers == prices.tickers

    def test_covariance_matrix_is_symmetric(self):
        prices = self.fetch.execute(tickers=(AAPL, MSFT), date_range=DATE_RANGE)
        returns = self.compute_returns.execute(prices)
        params = self.estimate_params.execute(returns)

        assert params.covariance_matrix[0][1] == pytest.approx(
            params.covariance_matrix[1][0]
        )

    def test_variances_are_positive(self):
        prices = self.fetch.execute(tickers=(AAPL, MSFT), date_range=DATE_RANGE)
        returns = self.compute_returns.execute(prices)
        params = self.estimate_params.execute(returns)

        for i in range(len(params.tickers)):
            assert params.covariance_matrix[i][i] > 0

    def test_annualization_factor_is_daily(self):
        prices = self.fetch.execute(tickers=(AAPL,), date_range=DATE_RANGE)
        returns = self.compute_returns.execute(prices)
        params = self.estimate_params.execute(returns)

        assert params.annualization_factor == 252

    def test_pipeline_output_summary(self, capsys):
        prices = self.fetch.execute(tickers=(AAPL, MSFT), date_range=DATE_RANGE)
        returns = self.compute_returns.execute(prices)
        params = self.estimate_params.execute(returns)

        print(f"\n{'=' * 60}")
        print("PIPELINE: HistoricalPrices -> Returns -> MarketParameters")
        print(f"{'=' * 60}")
        print(f"Tickers: {[t.value for t in params.tickers]}")
        print(f"Price observations: {len(prices.dates)}")
        print(f"Return observations: {len(returns.dates)}")
        print(f"Annualization factor: {params.annualization_factor}")
        print(f"\nFirst 3 log returns (AAPL): {returns.returns_by_ticker[AAPL][:3]}")
        print(f"First 3 log returns (MSFT): {returns.returns_by_ticker[MSFT][:3]}")
        print("\nAnnualized drift vector:")
        for ticker, drift in zip(params.tickers, params.drift_vector):
            print(f"  {ticker.value}: {drift:.6f}")
        print("\nAnnualized covariance matrix:")
        header = "         " + "  ".join(f"{t.value:>10}" for t in params.tickers)
        print(header)
        for ticker, row in zip(params.tickers, params.covariance_matrix):
            vals = "  ".join(f"{v:>10.6f}" for v in row)
            print(f"  {ticker.value:>5}  {vals}")
        print(f"{'=' * 60}")

        captured = capsys.readouterr()
        assert "PIPELINE" in captured.out
