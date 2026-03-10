from dataclasses import dataclass

@dataclass(frozen=True)
class Weight:
    value: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            raise ValueError("Weight must be between 0 and 1.")