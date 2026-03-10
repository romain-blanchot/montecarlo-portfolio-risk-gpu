from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Ticker:
    """
    Financial asset ticker symbol.

    Represents the identifier used on exchanges
    (e.g., AAPL, MSFT, BRK.B).

    The ticker is normalized to uppercase and validated
    using a simple regex.
    """

    value: str

    def __post_init__(self) -> None:
        v = self.value.strip().upper()

        if not v:
            raise ValueError("Ticker cannot be empty.")

        if not re.fullmatch(r"[A-Z0-9\.\-]{1,10}", v):
            raise ValueError(f"Invalid ticker: {v}")

        object.__setattr__(self, "value", v)
