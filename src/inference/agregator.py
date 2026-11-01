from src.inference.types import Detection


class DetectionAggregator:
    """
    Объединяет результаты нескольких специализированных
    моделей детекции.
    """

    def __init__(
        self,
        detectors: dict[str, object],
    ) -> None:
        self.detectors = detectors

    def predict(self, frame) -> list[Detection]:
        all_detections: list[Detection] = []

        for model_name, detector in self.detectors.items():
            detections = detector.predict(frame)

            # Указываем, какая модель обнаружила объект
            for detection in detections:
                detection.model_name = model_name

            all_detections.extend(detections)

        return all_detections

