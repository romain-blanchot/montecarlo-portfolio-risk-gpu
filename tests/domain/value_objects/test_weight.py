from dataclasses import FrozenInstanceError

import pytest

from portfolio_risk_engine.domain.value_objects.weight import Weight


class TestWeightCreation:
    def test_valid_weight(self):
        w = Weight(0.5)
        assert w.value == pytest.approx(0.5)

    def test_zero(self):
        w = Weight(0.0)
        assert w.value == pytest.approx(0.0)

    def test_one(self):
        w = Weight(1.0)
        assert w.value == pytest.approx(1.0)

    def test_small_weight(self):
        w = Weight(0.001)
        assert w.value == pytest.approx(0.001)


class TestWeightValidation:
    def test_negative(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            Weight(-0.1)

    def test_greater_than_one(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            Weight(1.01)

    def test_large_negative(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            Weight(-100.0)

    def test_large_positive(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            Weight(100.0)


class TestWeightImmutability:
    def test_frozen(self):
        w = Weight(0.5)
        with pytest.raises(FrozenInstanceError):
            w.value = 0.6  # type: ignore[misc]


class TestWeightEquality:
    def test_equal_weights(self):
        assert Weight(0.5) == Weight(0.5)

    def test_different_weights(self):
        assert Weight(0.3) != Weight(0.7)
