from dataclasses import FrozenInstanceError

import pytest

from portfolio_risk_engine.domain.models.simulation_result import (
    MonteCarloSimulationResult,
)
from portfolio_risk_engine.domain.value_objects.ticker import Ticker

AAPL = Ticker("AAPL")
MSFT = Ticker("MSFT")


def _make_result(**kwargs: object) -> MonteCarloSimulationResult:
    defaults: dict[str, object] = {
        "tickers": (AAPL,),
        "initial_prices": (150.0,),
        "terminal_prices": {AAPL: (155.0, 148.0, 160.0)},
        "num_simulations": 3,
        "time_horizon_days": 21,
    }
    defaults.update(kwargs)
    return MonteCarloSimulationResult(**defaults)  # type: ignore[arg-type]


class TestSimulationResultCreation:
    def test_single_ticker(self):
        result = _make_result()
        assert result.tickers == (AAPL,)
        assert result.num_simulations == 3
        assert result.time_horizon_days == 21

    def test_multiple_tickers(self):
        result = _make_result(
            tickers=(AAPL, MSFT),
            initial_prices=(150.0, 300.0),
            terminal_prices={
                AAPL: (155.0, 148.0, 160.0),
                MSFT: (310.0, 295.0, 320.0),
            },
        )
        assert len(result.tickers) == 2

    def test_initial_prices_preserved(self):
        result = _make_result()
        assert result.initial_prices == (150.0,)


class TestSimulationResultValidation:
    def test_empty_tickers_raises(self):
        with pytest.raises(ValueError, match="at least one ticker"):
            _make_result(
                tickers=(),
                initial_prices=(),
                terminal_prices={},
            )

    def test_initial_prices_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="initial_prices length"):
            _make_result(initial_prices=(150.0, 300.0))

    def test_negative_initial_price_raises(self):
        with pytest.raises(ValueError, match="initial prices must be positive"):
            _make_result(initial_prices=(-150.0,))

    def test_zero_initial_price_raises(self):
        with pytest.raises(ValueError, match="initial prices must be positive"):
            _make_result(initial_prices=(0.0,))

    def test_terminal_prices_keys_mismatch_raises(self):
        with pytest.raises(ValueError, match="keys must match tickers"):
            _make_result(terminal_prices={MSFT: (155.0, 148.0, 160.0)})

    def test_terminal_prices_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="has 2 values, expected 3"):
            _make_result(terminal_prices={AAPL: (155.0, 148.0)})

    def test_zero_simulations_raises(self):
        with pytest.raises(ValueError, match="num_simulations must be positive"):
            _make_result(
                terminal_prices={AAPL: ()},
                num_simulations=0,
            )

    def test_negative_time_horizon_raises(self):
        with pytest.raises(ValueError, match="time_horizon_days must be positive"):
            _make_result(time_horizon_days=-1)


class TestSimulationResultImmutability:
    def test_frozen(self):
        result = _make_result()
        with pytest.raises(FrozenInstanceError):
            result.num_simulations = 100  # type: ignore[misc]
