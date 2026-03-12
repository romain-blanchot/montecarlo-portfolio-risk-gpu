import pytest

from portfolio_risk_engine.domain.models.market_parameters import MarketParameters
from portfolio_risk_engine.domain.models.student_t_gbm import StudentTGBM
from portfolio_risk_engine.domain.value_objects.ticker import Ticker


def _make_params() -> MarketParameters:
    return MarketParameters(
        tickers=(Ticker("AAPL"), Ticker("MSFT")),
        drift_vector=(0.10, 0.12),
        covariance_matrix=((0.04, 0.01), (0.01, 0.05)),
        annualization_factor=252,
    )


class TestStudentTGBM:
    def test_valid_construction(self) -> None:
        model = StudentTGBM(
            market_parameters=_make_params(),
            cholesky_factor=((1.0, 0.0), (0.25, 0.97)),
            degrees_of_freedom=5.0,
        )
        assert model.degrees_of_freedom == 5.0

    def test_df_must_be_above_2(self) -> None:
        with pytest.raises(ValueError, match="degrees_of_freedom must be > 2"):
            StudentTGBM(
                market_parameters=_make_params(),
                cholesky_factor=((1.0, 0.0), (0.25, 0.97)),
                degrees_of_freedom=2.0,
            )

    def test_df_just_above_2(self) -> None:
        model = StudentTGBM(
            market_parameters=_make_params(),
            cholesky_factor=((1.0, 0.0), (0.25, 0.97)),
            degrees_of_freedom=2.01,
        )
        assert model.degrees_of_freedom == 2.01

    def test_cholesky_wrong_rows(self) -> None:
        with pytest.raises(ValueError, match="cholesky_factor rows"):
            StudentTGBM(
                market_parameters=_make_params(),
                cholesky_factor=((1.0, 0.0),),
                degrees_of_freedom=5.0,
            )

    def test_cholesky_wrong_cols(self) -> None:
        with pytest.raises(ValueError, match="columns, expected 2"):
            StudentTGBM(
                market_parameters=_make_params(),
                cholesky_factor=((1.0,), (0.25, 0.97)),
                degrees_of_freedom=5.0,
            )

    def test_frozen(self) -> None:
        model = StudentTGBM(
            market_parameters=_make_params(),
            cholesky_factor=((1.0, 0.0), (0.25, 0.97)),
            degrees_of_freedom=5.0,
        )
        with pytest.raises(AttributeError):
            model.degrees_of_freedom = 10.0  # type: ignore[misc]
