import numpy as np

from portfolio_risk_engine.domain.models.historical_returns import HistoricalReturns


class EstimateStudentTParameters:
    """Estimate Student-t degrees of freedom from historical returns.

    Uses method of moments on pooled standardized returns:
    excess_kurtosis = 6 / (nu - 4)  for nu > 4
    => nu = 6 / excess_kurtosis + 4
    """

    def execute(self, returns: HistoricalReturns) -> float:
        if len(returns.dates) < 4:
            raise ValueError(
                "Need at least 4 return observations for Student-t estimation."
            )

        # Pool standardized returns across all tickers
        standardized: list[float] = []
        for ticker in returns.tickers:
            r = np.array(returns.returns_by_ticker[ticker])
            mean = np.mean(r)
            std = np.std(r, ddof=1)
            if std > 0:
                standardized.extend(((r - mean) / std).tolist())

        if len(standardized) < 4:
            return 30.0  # fallback: near-Gaussian

        z = np.array(standardized)

        # Excess kurtosis (Fisher's definition)
        m4 = np.mean(z**4)
        excess_kurt = m4 - 3.0

        if excess_kurt <= 0:
            # Platykurtic or mesokurtic: t-distribution not appropriate
            # Return high DoF (effectively Gaussian)
            return 30.0

        # nu = 6 / excess_kurtosis + 4
        nu = 6.0 / excess_kurt + 4.0

        # Clamp to reasonable range
        nu = max(nu, 2.5)
        nu = min(nu, 100.0)

        return round(nu, 2)
