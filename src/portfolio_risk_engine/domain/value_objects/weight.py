from dataclasses import dataclass


@dataclass(frozen=True)
class Weight:
    """
    Portfolio weight value object.

    Represents the allocation of an asset in a portfolio.

    The value must be between 0 and 1 inclusive.
    """

    value: float

    def __post_init__(self) -> None:
        """
        Validate the weight value.
        """
        if not (0.0 <= self.value <= 1.0):
            raise ValueError("Weight must be between 0 and 1.")
