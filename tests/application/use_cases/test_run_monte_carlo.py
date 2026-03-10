import pytest

from portfolio_risk_engine.application.use_cases.run_monte_carlo import RunMonteCarlo
from portfolio_risk_engine.domain.models.gbm_model import MultivariateGBM
from portfolio_risk_engine.domain.models.market_parameters import MarketParameters
from portfolio_risk_engine.domain.models.simulation_result import (
    MonteCarloSimulationResult,
)
from portfolio_risk_engine.domain.value_objects.ticker import Ticker
from portfolio_risk_engine.infrastructure.simulation.cpu_monte_carlo_engine import (
    CpuMonteCarloEngine,
)

AAPL = Ticker("AAPL")
MSFT = Ticker("MSFT")


def _make_params(n: int = 1) -> MarketParameters:
    if n == 1:
        return MarketParameters(
            tickers=(AAPL,),
            drift_vector=(0.10,),
            covariance_matrix=((0.04,),),
            annualization_factor=252,
        )
    return MarketParameters(
        tickers=(AAPL, MSFT),
        drift_vector=(0.10, 0.12),
        covariance_matrix=((0.04, 0.01), (0.01, 0.06)),
        annualization_factor=252,
    )


class FakeEngine:
    def __init__(self) -> None:
        self.called_with: tuple[MultivariateGBM, tuple[float, ...], int, int] | None = (
            None
        )

    def simulate(
        self,
        model: MultivariateGBM,
        initial_prices: tuple[float, ...],
        num_simulations: int,
        time_horizon_days: int,
    ) -> MonteCarloSimulationResult:
        self.called_with = (model, initial_prices, num_simulations, time_horizon_days)
        return MonteCarloSimulationResult(
            tickers=model.market_parameters.tickers,
            initial_prices=initial_prices,
            terminal_prices={
                t: tuple(100.0 for _ in range(num_simulations))
                for t in model.market_parameters.tickers
            },
            num_simulations=num_simulations,
            time_horizon_days=time_horizon_days,
        )


class TestRunMonteCarloValidation:
    def test_negative_num_simulations_raises(self):
        use_case = RunMonteCarlo(engine=FakeEngine())
        with pytest.raises(ValueError, match="num_simulations must be positive"):
            use_case.execute(
                market_params=_make_params(),
                initial_prices=(100.0,),
                num_simulations=-1,
                time_horizon_days=21,
            )

    def test_zero_num_simulations_raises(self):
        use_case = RunMonteCarlo(engine=FakeEngine())
        with pytest.raises(ValueError, match="num_simulations must be positive"):
            use_case.execute(
                market_params=_make_params(),
                initial_prices=(100.0,),
                num_simulations=0,
                time_horizon_days=21,
            )

    def test_negative_time_horizon_raises(self):
        use_case = RunMonteCarlo(engine=FakeEngine())
        with pytest.raises(ValueError, match="time_horizon_days must be positive"):
            use_case.execute(
                market_params=_make_params(),
                initial_prices=(100.0,),
                num_simulations=100,
                time_horizon_days=-1,
            )

    def test_initial_prices_length_mismatch_raises(self):
        use_case = RunMonteCarlo(engine=FakeEngine())
        with pytest.raises(ValueError, match="initial_prices length"):
            use_case.execute(
                market_params=_make_params(1),
                initial_prices=(100.0, 200.0),
                num_simulations=100,
                time_horizon_days=21,
            )

    def test_zero_initial_price_raises(self):
        use_case = RunMonteCarlo(engine=FakeEngine())
        with pytest.raises(ValueError, match="initial prices must be positive"):
            use_case.execute(
                market_params=_make_params(1),
                initial_prices=(0.0,),
                num_simulations=100,
                time_horizon_days=21,
            )

    def test_negative_initial_price_raises(self):
        use_case = RunMonteCarlo(engine=FakeEngine())
        with pytest.raises(ValueError, match="initial prices must be positive"):
            use_case.execute(
                market_params=_make_params(1),
                initial_prices=(-50.0,),
                num_simulations=100,
                time_horizon_days=21,
            )


class TestRunMonteCarloDelegation:
    def test_delegates_to_engine(self):
        engine = FakeEngine()
        use_case = RunMonteCarlo(engine=engine)
        use_case.execute(
            market_params=_make_params(1),
            initial_prices=(150.0,),
            num_simulations=100,
            time_horizon_days=21,
        )
        assert engine.called_with is not None
        model, prices, n_sims, t_days = engine.called_with
        assert isinstance(model, MultivariateGBM)
        assert prices == (150.0,)
        assert n_sims == 100
        assert t_days == 21

    def test_builds_cholesky_factor(self):
        engine = FakeEngine()
        use_case = RunMonteCarlo(engine=engine)
        use_case.execute(
            market_params=_make_params(1),
            initial_prices=(150.0,),
            num_simulations=100,
            time_horizon_days=21,
        )
        model = engine.called_with[0]
        # Cholesky of ((0.04,)) should be ((0.2,))
        assert model.cholesky_factor[0][0] == pytest.approx(0.2)

    def test_returns_simulation_result(self):
        use_case = RunMonteCarlo(engine=FakeEngine())
        result = use_case.execute(
            market_params=_make_params(1),
            initial_prices=(150.0,),
            num_simulations=100,
            time_horizon_days=21,
        )
        assert isinstance(result, MonteCarloSimulationResult)
        assert result.num_simulations == 100


class TestRunMonteCarloWithCpuEngine:
    def test_end_to_end(self):
        engine = CpuMonteCarloEngine()
        use_case = RunMonteCarlo(engine=engine)
        result = use_case.execute(
            market_params=_make_params(2),
            initial_prices=(150.0, 300.0),
            num_simulations=1000,
            time_horizon_days=21,
        )
        assert result.num_simulations == 1000
        assert len(result.terminal_prices[AAPL]) == 1000
        assert len(result.terminal_prices[MSFT]) == 1000
