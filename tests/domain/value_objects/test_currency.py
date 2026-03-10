from dataclasses import FrozenInstanceError

import pytest

from portfolio_risk_engine.domain.value_objects.currency import Currency


class TestCurrencyCreation:
    def test_valid_currency(self):
        c = Currency("USD")
        assert c.code == "USD"

    def test_strips_whitespace(self):
        c = Currency("  usd  ")
        assert c.code == "USD"

    def test_uppercases(self):
        c = Currency("eur")
        assert c.code == "EUR"

    def test_strips_and_uppercases(self):
        c = Currency(" gbp ")
        assert c.code == "GBP"


class TestCurrencyValidation:
    def test_too_short(self):
        with pytest.raises(ValueError, match="3-letter ISO"):
            Currency("US")

    def test_too_long(self):
        with pytest.raises(ValueError, match="3-letter ISO"):
            Currency("USDX")

    def test_numeric(self):
        with pytest.raises(ValueError, match="3-letter ISO"):
            Currency("123")

    def test_contains_digit(self):
        with pytest.raises(ValueError, match="3-letter ISO"):
            Currency("U2D")

    def test_empty_string(self):
        with pytest.raises(ValueError, match="3-letter ISO"):
            Currency("")

    def test_whitespace_only(self):
        with pytest.raises(ValueError, match="3-letter ISO"):
            Currency("   ")

    def test_special_characters(self):
        with pytest.raises(ValueError, match="3-letter ISO"):
            Currency("U$D")


class TestCurrencyImmutability:
    def test_frozen(self):
        c = Currency("USD")
        with pytest.raises(FrozenInstanceError):
            c.code = "EUR"  # type: ignore[misc]


class TestCurrencyEquality:
    def test_equal_currencies(self):
        assert Currency("USD") == Currency("USD")

    def test_same_after_normalization(self):
        assert Currency("usd") == Currency(" USD ")

    def test_different_currencies(self):
        assert Currency("USD") != Currency("EUR")
