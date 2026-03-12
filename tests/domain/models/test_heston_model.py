import pytest

from portfolio_risk_engine.domain.models.heston_model import (
    HestonAssetParams,
    HestonModel,
)
from portfolio_risk_engine.domain.value_objects.ticker import Ticker


class TestHestonAssetParams:
    def test_valid_construction(self) -> None:
        p = HestonAssetParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
        assert p.kappa == 2.0
        assert p.theta == 0.04

    def test_feller_satisfied(self) -> None:
        # 2*2.0*0.04 = 0.16 > 0.3^2 = 0.09
        p = HestonAssetParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
        assert p.feller_satisfied is True

    def test_feller_not_satisfied(self) -> None:
        # 2*0.5*0.01 = 0.01 < 0.5^2 = 0.25
        p = HestonAssetParams(kappa=0.5, theta=0.01, xi=0.5, rho=-0.5, v0=0.01)
        assert p.feller_satisfied is False

    def test_kappa_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="kappa must be positive"):
            HestonAssetParams(kappa=0, theta=0.04, xi=0.3, rho=-0.5, v0=0.04)

    def test_theta_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="theta must be positive"):
            HestonAssetParams(kappa=1.0, theta=0, xi=0.3, rho=-0.5, v0=0.04)

    def test_xi_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="xi must be positive"):
            HestonAssetParams(kappa=1.0, theta=0.04, xi=0, rho=-0.5, v0=0.04)

    def test_rho_must_be_in_range(self) -> None:
        with pytest.raises(ValueError, match="rho must be in"):
            HestonAssetParams(kappa=1.0, theta=0.04, xi=0.3, rho=-1.0, v0=0.04)
        with pytest.raises(ValueError, match="rho must be in"):
            HestonAssetParams(kappa=1.0, theta=0.04, xi=0.3, rho=1.0, v0=0.04)

    def test_v0_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="v0 must be positive"):
            HestonAssetParams(kappa=1.0, theta=0.04, xi=0.3, rho=-0.5, v0=0)


class TestHestonModel:
    def _make_params(self) -> HestonAssetParams:
        return HestonAssetParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)

    def test_valid_construction(self) -> None:
        model = HestonModel(
            tickers=(Ticker("AAPL"),),
            drift_vector=(0.1,),
            asset_params=(self._make_params(),),
            correlation_cholesky=((1.0,),),
            annualization_factor=252,
        )
        assert len(model.tickers) == 1

    def test_empty_tickers_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one ticker"):
            HestonModel(
                tickers=(),
                drift_vector=(),
                asset_params=(),
                correlation_cholesky=(),
                annualization_factor=252,
            )

    def test_drift_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="drift_vector length"):
            HestonModel(
                tickers=(Ticker("AAPL"),),
                drift_vector=(0.1, 0.2),
                asset_params=(self._make_params(),),
                correlation_cholesky=((1.0,),),
                annualization_factor=252,
            )

    def test_asset_params_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="asset_params length"):
            HestonModel(
                tickers=(Ticker("AAPL"), Ticker("MSFT")),
                drift_vector=(0.1, 0.2),
                asset_params=(self._make_params(),),
                correlation_cholesky=((1.0, 0.0), (0.0, 1.0)),
                annualization_factor=252,
            )

    def test_correlation_cholesky_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="n x n"):
            HestonModel(
                tickers=(Ticker("AAPL"),),
                drift_vector=(0.1,),
                asset_params=(self._make_params(),),
                correlation_cholesky=((1.0, 0.0),),
                annualization_factor=252,
            )

    def test_annualization_factor_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="annualization_factor must be positive"):
            HestonModel(
                tickers=(Ticker("AAPL"),),
                drift_vector=(0.1,),
                asset_params=(self._make_params(),),
                correlation_cholesky=((1.0,),),
                annualization_factor=0,
            )
