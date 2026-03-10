from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioRiskMetrics:
    """Risk metrics computed from Monte Carlo simulation. VaR and ES use loss-positive convention."""

    mean_return: float
    volatility: float
    var_95: float
    var_99: float
    es_95: float
    es_99: float
