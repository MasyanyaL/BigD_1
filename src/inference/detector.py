from ultralytics import YOLO

from src.inference.types import Detection


class Detector:
    """
    Возвращает список объектов Detection.
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.5,
    ) -> None:
        self.model = YOLO(model_path)
        self.confidence = confidence

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

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist(),
            )

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=str(
                        result.names[class_id]
                    ),
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        return detections