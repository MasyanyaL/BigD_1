from __future__ import annotations

import logging

from ultralytics import YOLO

from src.inference.types import Detection

logger = logging.getLogger(__name__)


class Detector:
    """
    Обёртка над одной YOLO-моделью.

    - use_tracking=False → predict (этап 3, baseline)
    - use_tracking=True  → track    (этап 5, нужен Track ID)
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.4,
        model_name: str = "unknown",
        allowed_classes: set[str] | None = None,
        use_tracking: bool = False,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        self.confidence = confidence
        self.model_name = model_name
        self.allowed_classes = allowed_classes
        self.use_tracking = use_tracking
        self.tracker = tracker

        try:
            self.model = YOLO(model_path)
        except Exception as exc:
            logger.exception(
                "Не удалось загрузить модель '%s' из '%s'", model_name, model_path
            )
            raise RuntimeError(
                f"Ошибка загрузки модели '{model_name}' ({model_path}): {exc}"
            ) from exc

        logger.info(
            "Модель загружена: %s (%s), tracking=%s",
            model_name, model_path, use_tracking,
        )

    def predict(self, frame) -> list[Detection]:
        """Возвращает список унифицированных детекций."""
        try:
            if self.use_tracking:
                result = self.model.track(
                    frame,
                    conf=self.confidence,
                    verbose=False,
                    persist=True,
                    tracker=self.tracker,
                )[0]
            else:
                result = self.model.predict(
                    frame, conf=self.confidence, verbose=False
                )[0]
        except Exception:
            logger.exception("Ошибка инференса модели '%s'", self.model_name)
            return []

        if result is None or result.boxes is None or len(result.boxes) == 0:
            return []

        detections: list[Detection] = []
        names = result.names

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            class_name = str(names[class_id])

            if self.allowed_classes is not None and class_name not in self.allowed_classes:
                continue

            track_id: int | None = None
            if self.use_tracking and box.id is not None:
                track_id = int(box.id.item())

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    model_name=self.model_name,
                    track_id=track_id,
                )
            )

        return detections