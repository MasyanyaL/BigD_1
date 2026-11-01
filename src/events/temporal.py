from __future__ import annotations

from collections import deque


class TemporalConfirm:
    """
    Скользящее окно m кадров. Возвращает True, если за последние m кадров
    объект был внутри зоны не менее n раз и окно уже заполнено.
    """

    def __init__(self, n: int = 3, m: int = 5) -> None:
        if m < 1:
            raise ValueError(f"m должно быть >= 1, получено {m}")
        if n < 1:
            raise ValueError(f"n должно быть >= 1, получено {n}")
        if n > m:
            raise ValueError(f"n ({n}) не может быть больше m ({m})")

        self.n = n
        self.m = m
        self.window: deque[bool] = deque(maxlen=m)

    def update(self, detected: bool) -> bool:
        self.window.append(bool(detected))
        return len(self.window) == self.m and sum(self.window) >= self.n

    def reset(self) -> None:
        self.window.clear()