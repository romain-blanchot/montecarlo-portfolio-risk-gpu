from dataclasses import dataclass

@dataclass(frozen=True)
class Currency:
    code: str

    def __post_init__(self) -> None:
        v = self.code.strip().upper()

        if len(v) != 3 or not v.isalpha():
            raise ValueError("Currency must be a 3-letter ISO code.")

        object.__setattr__(self, "code", v)