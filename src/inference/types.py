from dataclasses import dataclass


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float

    x1: int
    y1: int
    x2: int
    y2: int
    model_name: str = ""

    @property
    def center(self) -> tuple[int, int]:
        return (
            (self.x1 + self.x2) // 2,
            (self.y1 + self.y2) // 2,
        )

    @property
    def bottom_center(self) -> tuple[int, int]:
        return (
            (self.x1 + self.x2) // 2,
            self.y2,
        )