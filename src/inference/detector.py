
from ultralytics import YOLO

from src.inference.types import Detection


class Detector:
    """
    Загружает одну модель, выполняет детекцию
    и возвращает унифицированный список Detection.
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.5,
        model_name: str = "unknown",
        allowed_classes: set[str] | None = None,
    ) -> None:
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.model_name = model_name
        self.allowed_classes = allowed_classes

    def predict(self, frame) -> list[Detection]:
        result = self.model.predict(
            frame,
            conf=self.confidence,
            verbose=False,
        )[0]

        detections: list[Detection] = []

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            class_name = str(result.names[class_id])

            # Фильтрация классов, если она задана
            if (
                self.allowed_classes is not None
                and class_name not in self.allowed_classes
            ):
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist(),
            )

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    model_name=self.model_name,
                )
            )

        return detections
