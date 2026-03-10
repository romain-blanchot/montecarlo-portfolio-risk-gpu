from dataclasses import dataclass
import re
# Par default :
# @dataclass(
#     init=True,
#     repr=True,
#     eq=True,
#     order=False,
#     unsafe_hash=False,
#     frozen=False,
#     match_args=True,
#     kw_only=False,
#     slots=False
# )



@dataclass(frozen=True)
class Asset:
    ticker : str
    currency: str
    name : str | None = None

    def __post_init__(self) -> None:
        cleaned_ticker = self._clean_text(self.ticker, "ticker")
        cleaned_currency = self._clean_text(self.currency, "currency")

        self._validate_ticker(cleaned_ticker)
        self._validate_currency(cleaned_currency)

        object.__setattr__(self, "ticker", cleaned_ticker)
        object.__setattr__(self, "currency", cleaned_currency)

    @staticmethod
    def _clean_text(value: str, field_name: str) -> str:
        v = value.strip()
        if not v:
            raise ValueError(f"Asset {field_name} cannot be empty.")
        return v.upper()

    @staticmethod
    def _validate_ticker(value: str) -> None:
        if not re.fullmatch(r"[A-Z0-9\.\-]{1,10}", value):
            raise ValueError(f"Invalid ticker: {value}")

    @staticmethod
    def _validate_currency(value: str) -> None:
        if len(value) != 3 or not value.isalpha():
            raise ValueError("Currency must be a 3-letter code like USD or EUR.")
