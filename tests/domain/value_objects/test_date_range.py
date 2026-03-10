from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from portfolio_risk_engine.domain.value_objects.date_range import DateRange


class TestDateRangeCreation:
    def test_valid_range(self):
        dr = DateRange(start=date(2024, 1, 1), end=date(2024, 12, 31))
        assert dr.start == date(2024, 1, 1)
        assert dr.end == date(2024, 12, 31)

    def test_single_day_apart(self):
        dr = DateRange(start=date(2024, 1, 1), end=date(2024, 1, 2))
        assert dr.start < dr.end


class TestDateRangeValidation:
    def test_start_equals_end(self):
        with pytest.raises(ValueError, match="start date must be before end date"):
            DateRange(start=date(2024, 1, 1), end=date(2024, 1, 1))

    def test_start_after_end(self):
        with pytest.raises(ValueError, match="start date must be before end date"):
            DateRange(start=date(2024, 12, 31), end=date(2024, 1, 1))


class TestDateRangeImmutability:
    def test_frozen(self):
        dr = DateRange(start=date(2024, 1, 1), end=date(2024, 12, 31))
        with pytest.raises(FrozenInstanceError):
            dr.start = date(2024, 6, 1)  # type: ignore[misc]


class TestDateRangeEquality:
    def test_equal(self):
        d1 = DateRange(start=date(2024, 1, 1), end=date(2024, 12, 31))
        d2 = DateRange(start=date(2024, 1, 1), end=date(2024, 12, 31))
        assert d1 == d2

    def test_different(self):
        d1 = DateRange(start=date(2024, 1, 1), end=date(2024, 12, 31))
        d2 = DateRange(start=date(2024, 1, 1), end=date(2024, 6, 30))
        assert d1 != d2
