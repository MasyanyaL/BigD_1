import cv2

from src.inference.types import Detection


def draw_detections(
    frame,
    detections: list[Detection],
):
    output = frame.copy()

    for det in detections:
        cv2.rectangle(
            output,
            (det.x1, det.y1),
            (det.x2, det.y2),
            (0, 255, 0),
            2,
        )

        label = (
            f"{det.class_name}: "
            f"{det.confidence:.2f}"
        )

        cv2.putText(
            output,
            label,
            (det.x1, max(20, det.y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return output