from datetime import date
from statistics import median

import numpy as np

from portfolio_risk_engine.domain.models.historical_returns import HistoricalReturns
from portfolio_risk_engine.domain.models.market_parameters import MarketParameters


class EstimateMarketParameters:
    def execute(self, returns: HistoricalReturns) -> MarketParameters:
        if len(returns.dates) < 2:
            raise ValueError("Need at least 2 return observations for estimation.")

        annualization_factor = self._estimate_annualization_factor(returns.dates)

        # Build returns matrix: shape (n_tickers, n_observations)
        returns_matrix = np.array(
            [returns.returns_by_ticker[ticker] for ticker in returns.tickers]
        )

        # Mean vector (annualized)
        mean_vector = returns_matrix.mean(axis=1) * annualization_factor

        # Covariance matrix (annualized, unbiased ddof=1)
        cov_matrix = np.cov(returns_matrix) * annualization_factor

        # np.cov returns 0-d array for single ticker
        if len(returns.tickers) == 1:
            cov_matrix = np.array([[float(cov_matrix)]])

        drift_vector = tuple(float(x) for x in mean_vector)
        covariance_matrix = tuple(tuple(float(x) for x in row) for row in cov_matrix)

        return MarketParameters(
            tickers=returns.tickers,
            drift_vector=drift_vector,
            covariance_matrix=covariance_matrix,
            annualization_factor=annualization_factor,
        )

    @staticmethod
    def _estimate_annualization_factor(dates: tuple[date, ...]) -> int:
        gaps = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
        median_gap = median(gaps)

        if median_gap <= 5:
            return 252  # daily
        if median_gap <= 10:
            return 52  # weekly
        if median_gap <= 40:
            return 12  # monthly
        if median_gap <= 100:
            return 4  # quarterly
        return 1  # annual
