try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

from portfolio_risk_engine.domain.models.heston_model import HestonModel
from portfolio_risk_engine.domain.models.simulation_result import (
    MonteCarloSimulationResult,
)


class GpuHestonEngine:
    """CuPy-based GPU implementation for Heston stochastic volatility simulation.

    Euler-Maruyama with full truncation. Time-stepping loop on CPU,
    per-step computation fully vectorized on GPU across simulations.
    """

    def __init__(self, seed: int | None = None) -> None:
        if not _CUPY_AVAILABLE:
            raise RuntimeError(
                "CuPy is not installed. Install with: pip install cupy-cuda12x"
            )
        try:
            cp.cuda.runtime.getDeviceCount()
        except Exception as e:
            raise RuntimeError(f"No CUDA-capable GPU detected: {e}") from e
        self._seed = seed

    def simulate(
        self,
        model: HestonModel,
        initial_prices: tuple[float, ...],
        num_simulations: int,
        time_horizon_days: int,
    ) -> MonteCarloSimulationResult:
        n = len(model.tickers)
        dt = 1.0 / model.annualization_factor
        sqrt_dt = float(cp.sqrt(cp.float64(dt)))

        drift = cp.array(model.drift_vector, dtype=cp.float64)
        kappa = cp.array([p.kappa for p in model.asset_params], dtype=cp.float64)
        theta = cp.array([p.theta for p in model.asset_params], dtype=cp.float64)
        xi = cp.array([p.xi for p in model.asset_params], dtype=cp.float64)
        rho = cp.array([p.rho for p in model.asset_params], dtype=cp.float64)
        L_corr = cp.array(model.correlation_cholesky, dtype=cp.float64)

        S0_arr = cp.array(initial_prices, dtype=cp.float64)
        log_S = cp.log(cp.tile(S0_arr, (num_simulations, 1)).T)
        v = cp.tile(
            cp.array([p.v0 for p in model.asset_params], dtype=cp.float64),
            (num_simulations, 1),
        ).T

        kappa_r = kappa[:, cp.newaxis]
        theta_r = theta[:, cp.newaxis]
        xi_r = xi[:, cp.newaxis]
        rho_r = rho[:, cp.newaxis]
        sqrt_1_rho2 = cp.sqrt(1.0 - rho**2)[:, cp.newaxis]
        drift_r = drift[:, cp.newaxis]

        rng = cp.random.RandomState(self._seed)

        for _ in range(time_horizon_days):
            Z_v = rng.standard_normal((n, num_simulations), dtype=cp.float64)
            Z_indep = rng.standard_normal((n, num_simulations), dtype=cp.float64)
            Z_corr = L_corr @ Z_indep
            Z_s = rho_r * Z_v + sqrt_1_rho2 * Z_corr

            v_pos = cp.maximum(v, 0.0)
            sqrt_v = cp.sqrt(v_pos)

            log_S += (drift_r - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * Z_s
            v = v + kappa_r * (theta_r - v) * dt + xi_r * sqrt_v * sqrt_dt * Z_v
            v = cp.maximum(v, 0.0)

        terminal_prices_cpu = cp.asnumpy(cp.exp(log_S))

        terminal_prices = {}
        for i, ticker in enumerate(model.tickers):
            terminal_prices[ticker] = tuple(terminal_prices_cpu[i].tolist())

        return MonteCarloSimulationResult(
            tickers=model.tickers,
            initial_prices=initial_prices,
            terminal_prices=terminal_prices,
            num_simulations=num_simulations,
            time_horizon_days=time_horizon_days,
        )
