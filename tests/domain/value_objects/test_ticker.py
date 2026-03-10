from dataclasses import FrozenInstanceError

import pytest

from portfolio_risk_engine.domain.value_objects.ticker import Ticker


class TestTickerCreation:
    def test_valid_ticker(self):
        t = Ticker("AAPL")
        assert t.value == "AAPL"

    def test_strips_whitespace(self):
        t = Ticker("  aapl  ")
        assert t.value == "AAPL"

    def test_uppercases(self):
        t = Ticker("msft")
        assert t.value == "MSFT"

    def test_with_dot(self):
        t = Ticker("BRK.B")
        assert t.value == "BRK.B"

    def test_with_hyphen(self):
        t = Ticker("BF-B")
        assert t.value == "BF-B"

    def test_with_digits(self):
        t = Ticker("3M")
        assert t.value == "3M"

    def test_single_character(self):
        t = Ticker("X")
        assert t.value == "X"

    def test_max_length(self):
        t = Ticker("ABCDEFGHIJ")
        assert t.value == "ABCDEFGHIJ"


class TestTickerValidation:
    def test_empty_string(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            Ticker("")

    def test_whitespace_only(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            Ticker("   ")

    def test_too_long(self):
        with pytest.raises(ValueError, match="Invalid ticker"):
            Ticker("ABCDEFGHIJK")

    def test_special_characters(self):
        with pytest.raises(ValueError, match="Invalid ticker"):
            Ticker("AA$L")

    def test_space_in_middle(self):
        with pytest.raises(ValueError, match="Invalid ticker"):
            Ticker("AA PL")

    def test_underscore(self):
        with pytest.raises(ValueError, match="Invalid ticker"):
            Ticker("AA_PL")


class TestTickerImmutability:
    def test_frozen(self):
        t = Ticker("AAPL")
        with pytest.raises(FrozenInstanceError):
            t.value = "MSFT"  # type: ignore[misc]


class TestTickerEquality:
    def test_equal_tickers(self):
        assert Ticker("AAPL") == Ticker("AAPL")

    def test_same_after_normalization(self):
        assert Ticker("aapl") == Ticker(" AAPL ")

    def test_different_tickers(self):
        assert Ticker("AAPL") != Ticker("MSFT")
