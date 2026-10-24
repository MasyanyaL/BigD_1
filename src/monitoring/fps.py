
import time
from collections import deque


class FPSMeter:
    """
    измеряет среднюю частоту обработки кадров
    """

    def __init__(self, window: int = 30) -> None:
        if window < 2:
            raise ValueError("Размер окна FPS должен быть не меньше 2")

        self.times = deque(maxlen=window)

    def tick(self) -> float:
        #регистрирует обработанный кадр и возвращает FPS
        now = time.perf_counter()
        self.times.append(now)

        if len(self.times) < 2:
            return 0.0

        dt = self.times[-1] - self.times[0]

        if dt <= 0:
            return 0.0

        return (len(self.times) - 1) / dt