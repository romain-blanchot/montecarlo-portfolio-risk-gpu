from dataclasses import dataclass

from portfolio_risk_engine.domain.models.asset import Asset
from portfolio_risk_engine.domain.value_objects.weight import Weight


@dataclass(frozen=True)
class Position:
    asset: Asset
    weight: Weight
