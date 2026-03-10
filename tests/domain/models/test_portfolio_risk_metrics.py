from dataclasses import FrozenInstanceError

import pytest

from portfolio_risk_engine.domain.models.portfolio_risk_metrics import (
    PortfolioRiskMetrics,
)


class TestPortfolioRiskMetricsCreation:
    def test_create(self):
        metrics = PortfolioRiskMetrics(
            mean_return=0.05,
            volatility=0.20,
            var_95=0.10,
            var_99=0.15,
            es_95=0.12,
            es_99=0.18,
        )
        assert metrics.mean_return == pytest.approx(0.05)
        assert metrics.volatility == pytest.approx(0.20)
        assert metrics.var_95 == pytest.approx(0.10)
        assert metrics.var_99 == pytest.approx(0.15)
        assert metrics.es_95 == pytest.approx(0.12)
        assert metrics.es_99 == pytest.approx(0.18)

    def test_negative_mean_return_allowed(self):
        metrics = PortfolioRiskMetrics(
            mean_return=-0.05,
            volatility=0.20,
            var_95=0.10,
            var_99=0.15,
            es_95=0.12,
            es_99=0.18,
        )
        assert metrics.mean_return == -0.05


class TestPortfolioRiskMetricsImmutability:
    def test_frozen(self):
        metrics = PortfolioRiskMetrics(
            mean_return=0.05,
            volatility=0.20,
            var_95=0.10,
            var_99=0.15,
            es_95=0.12,
            es_99=0.18,
        )
        with pytest.raises(FrozenInstanceError):
            metrics.var_95 = 0.0  # type: ignore[misc]
