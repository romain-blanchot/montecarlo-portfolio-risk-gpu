from dataclasses import FrozenInstanceError

import pytest

from portfolio_risk_engine.domain.models.market_parameters import MarketParameters
from portfolio_risk_engine.domain.value_objects.ticker import Ticker

AAPL = Ticker("AAPL")
MSFT = Ticker("MSFT")


class TestMarketParametersCreation:
    def test_single_ticker(self):
        mp = MarketParameters(
            tickers=(AAPL,),
            drift_vector=(0.1,),
            covariance_matrix=((0.04,),),
            annualization_factor=252,
        )
        assert mp.tickers == (AAPL,)
        assert mp.annualization_factor == 252

    def test_multiple_tickers(self):
        mp = MarketParameters(
            tickers=(AAPL, MSFT),
            drift_vector=(0.1, 0.12),
            covariance_matrix=((0.04, 0.01), (0.01, 0.05)),
            annualization_factor=252,
        )
        assert len(mp.tickers) == 2
        assert len(mp.covariance_matrix) == 2
        assert len(mp.covariance_matrix[0]) == 2


class TestMarketParametersValidation:
    def test_empty_tickers(self):
        with pytest.raises(ValueError, match="at least one ticker"):
            MarketParameters(
                tickers=(),
                drift_vector=(),
                covariance_matrix=(),
                annualization_factor=252,
            )

    def test_drift_vector_length_mismatch(self):
        with pytest.raises(ValueError, match="drift_vector length"):
            MarketParameters(
                tickers=(AAPL, MSFT),
                drift_vector=(0.1,),
                covariance_matrix=((0.04, 0.01), (0.01, 0.05)),
                annualization_factor=252,
            )

    def test_covariance_matrix_wrong_rows(self):
        with pytest.raises(ValueError, match="n x n"):
            MarketParameters(
                tickers=(AAPL, MSFT),
                drift_vector=(0.1, 0.12),
                covariance_matrix=((0.04, 0.01),),
                annualization_factor=252,
            )

    def test_covariance_matrix_wrong_cols(self):
        with pytest.raises(ValueError, match="n x n"):
            MarketParameters(
                tickers=(AAPL, MSFT),
                drift_vector=(0.1, 0.12),
                covariance_matrix=((0.04,), (0.01,)),
                annualization_factor=252,
            )

    def test_annualization_factor_zero(self):
        with pytest.raises(ValueError, match="annualization_factor must be positive"):
            MarketParameters(
                tickers=(AAPL,),
                drift_vector=(0.1,),
                covariance_matrix=((0.04,),),
                annualization_factor=0,
            )

    def test_annualization_factor_negative(self):
        with pytest.raises(ValueError, match="annualization_factor must be positive"):
            MarketParameters(
                tickers=(AAPL,),
                drift_vector=(0.1,),
                covariance_matrix=((0.04,),),
                annualization_factor=-1,
            )


class TestMarketParametersImmutability:
    def test_frozen(self):
        mp = MarketParameters(
            tickers=(AAPL,),
            drift_vector=(0.1,),
            covariance_matrix=((0.04,),),
            annualization_factor=252,
        )
        with pytest.raises(FrozenInstanceError):
            mp.tickers = ()  # type: ignore[misc]
