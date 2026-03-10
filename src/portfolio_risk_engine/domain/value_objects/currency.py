from dataclasses import dataclass


@dataclass(frozen=True)
class Currency:
    """
    Value object representing a currency using a 3-letter ISO code.

    The code is normalized to uppercase and validated to ensure it
    follows the ISO 4217 format (e.g., USD, EUR, JPY).
    """

    code: str

    def __post_init__(self) -> None:
        """
        Normalize and validate the currency code after initialization.
        """
        v = self.code.strip().upper()

        if len(v) != 3 or not v.isalpha():
            raise ValueError("Currency must be a 3-letter ISO code.")

        object.__setattr__(self, "code", v)
