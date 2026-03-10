import numpy as np
import pytest

from portfolio_risk_engine.domain.portfolio import Portfolio


class TestPortfolio:
    def test_initial_value_two_assets(self) -> None:
        s0 = np.array([100.0, 200.0])
        weights = np.array([0.6, 0.4])
        p = Portfolio(S0=s0, weights=weights)
        # 0.6*100 + 0.4*200 = 60 + 80 = 140
        assert np.isclose(p.initial_value, 140.0)

    def test_initial_value_equal_weights(self) -> None:
        s0 = np.array([50.0, 100.0, 150.0])
        weights = np.ones(3) / 3
        p = Portfolio(S0=s0, weights=weights)
        expected = (50.0 + 100.0 + 150.0) / 3
        assert np.isclose(p.initial_value, expected)

    def test_initial_value_single_asset(self) -> None:
        p = Portfolio(S0=np.array([75.0]), weights=np.array([1.0]))
        assert np.isclose(p.initial_value, 75.0)

    def test_weights_not_summing_to_one_raises(self) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            Portfolio(
                S0=np.array([100.0, 200.0]),
                weights=np.array([0.6, 0.5]),  # sum = 1.1
            )

    def test_weights_summing_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            Portfolio(
                S0=np.array([100.0, 200.0]),
                weights=np.array([0.3, 0.3]),  # sum = 0.6
            )

    def test_mismatched_shapes_raise(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            Portfolio(
                S0=np.array([100.0, 200.0, 300.0]),
                weights=np.array([0.5, 0.5]),
            )
