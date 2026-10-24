from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


class VideoSource:
    def __init__(self, source: str | int, reconnect_sec: float = 2.0) -> None:
        if reconnect_sec < 0:
            raise ValueError("reconnect_sec не может быть отрицательным")

        self.source = source
        self.reconnect_sec = reconnect_sec
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        #открываем источник видео
        self.close()

        # если это путь к файлу (строка), проверяем его наличие
        # если это RTSP (начинается с rtsp) или индекс камеры (int) - пропускаем проверку
        if isinstance(self.source, str) and not self.source.lower().startswith(("rtsp://", "rtsps://")):
            if not Path(self.source).is_file():
                raise FileNotFoundError(f"Видеофайл не найден: {self.source}")

        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            self.close()
            raise RuntimeError(f"Не удалось открыть источник видео: {self.source}")

        logger.info("Источник видео открыт: %s", self.source)

    def read(self):
        #dозвращает кадр
        if self.cap is None or not self.cap.isOpened():
            self.open()

        assert self.cap is not None
        ok, frame = self.cap.read()

        if ok and frame is not None:
            return frame

        # если файл - закончился
        if isinstance(self.source, str) and not self.source.lower().startswith(("rtsp://", "rtsps://")):
            raise EOFError(f"Воспроизведение завершено: {self.source}")

        # для камер- пытаемся переподключиться
        logger.warning("Не удалось получить кадр. Переподключение к %s", self.source)
        time.sleep(self.reconnect_sec)
        self.open()

        assert self.cap is not None
        ok, frame = self.cap.read()

        if not ok or frame is None:
            raise RuntimeError(f"Видеопоток недоступен после переподключения: {self.source}")

        return frame

    def close(self) -> None:
        """Освобождает ресурсы OpenCV."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Источник видео закрыт: %s", self.source)
