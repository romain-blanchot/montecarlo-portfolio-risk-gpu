import numpy as np

from portfolio_risk_engine.domain.models.heston_model import (
    HestonAssetParams,
    HestonModel,
)
from portfolio_risk_engine.domain.models.historical_returns import HistoricalReturns
from portfolio_risk_engine.domain.models.market_parameters import MarketParameters


class EstimateHestonParameters:
    """Estimate Heston stochastic volatility parameters from historical returns.

    Uses method-of-moments on realized variance as proxy for latent variance.
    """

    def __init__(self, variance_window: int = 21) -> None:
        self._window = variance_window

    def execute(
        self,
        returns: HistoricalReturns,
        market_params: MarketParameters,
    ) -> HestonModel:
        ann = market_params.annualization_factor
        dt = 1.0 / ann
        n_tickers = len(returns.tickers)

        asset_params_list: list[HestonAssetParams] = []

        for ticker in returns.tickers:
            r = np.array(returns.returns_by_ticker[ticker])
            params = self._estimate_single_asset(r, ann, dt)
            asset_params_list.append(params)

        # Inter-asset correlation from returns
        if n_tickers > 1:
            returns_matrix = np.array(
                [returns.returns_by_ticker[t] for t in returns.tickers]
            )
            corr = np.corrcoef(returns_matrix)
            # Ensure positive definiteness
            eigvals = np.linalg.eigvalsh(corr)
            if np.min(eigvals) <= 0:
                corr = self._nearest_pd(corr)
            chol = np.linalg.cholesky(corr)
            correlation_cholesky = tuple(tuple(float(x) for x in row) for row in chol)
        else:
            correlation_cholesky = ((1.0,),)

        return HestonModel(
            tickers=returns.tickers,
            drift_vector=market_params.drift_vector,
            asset_params=tuple(asset_params_list),
            correlation_cholesky=correlation_cholesky,
            annualization_factor=ann,
        )

    def _estimate_single_asset(
        self,
        r: np.ndarray,
        ann: int,
        dt: float,
    ) -> HestonAssetParams:
        n = len(r)
        window = min(self._window, max(n // 3, 5))

        # Realized variance series (annualized)
        r_sq = r**2
        if n < window + 5:
            # Not enough data for rolling window estimation
            var = float(np.var(r, ddof=1) * ann)
            return HestonAssetParams(
                kappa=1.0,
                theta=var,
                xi=0.3,
                rho=-0.5,
                v0=var,
            )

        cumsum = np.cumsum(r_sq)
        cumsum = np.insert(cumsum, 0, 0.0)
        rolling_sum = cumsum[window:] - cumsum[:-window]
        realized_var = rolling_sum * ann / window

        # theta: long-run variance
        theta = float(np.mean(realized_var))
        theta = max(theta, 1e-6)

        # v0: recent realized variance
        v0 = float(realized_var[-1])
        v0 = max(v0, 1e-6)

        # kappa: from AR(1) on realized variance
        v_lag = realized_var[:-1]
        v_cur = realized_var[1:]
        if len(v_lag) > 2 and np.std(v_lag) > 0:
            corr_coef = np.corrcoef(v_lag, v_cur)[0, 1]
            b = corr_coef * np.std(v_cur) / np.std(v_lag)
            b = float(np.clip(b, 0.001, 0.999))
            kappa = float(-np.log(b) * ann / window)
            kappa = max(kappa, 0.01)
        else:
            kappa = 1.0

        # xi: vol-of-vol
        dv = np.diff(realized_var)
        mean_v = float(np.mean(realized_var[:-1]))
        if mean_v > 0:
            xi = float(np.std(dv) / np.sqrt(mean_v * window * dt))
            xi = max(xi, 0.01)
        else:
            xi = 0.3

        # rho: leverage effect (correlation between returns and variance changes)
        min_len = min(n - window, len(dv))
        if min_len > 2:
            r_aligned = r[window : window + min_len]
            dv_aligned = dv[:min_len]
            if np.std(r_aligned) > 0 and np.std(dv_aligned) > 0:
                rho = float(np.corrcoef(r_aligned, dv_aligned)[0, 1])
                rho = float(np.clip(rho, -0.99, 0.99))
            else:
                rho = -0.5
        else:
            rho = -0.5

        return HestonAssetParams(
            kappa=round(kappa, 4),
            theta=round(theta, 6),
            xi=round(xi, 4),
            rho=round(rho, 4),
            v0=round(v0, 6),
        )

    @staticmethod
    def _nearest_pd(matrix: np.ndarray) -> np.ndarray:
        """Compute nearest positive-definite matrix (Higham 2002)."""
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.maximum(eigvals, 1e-8)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T
