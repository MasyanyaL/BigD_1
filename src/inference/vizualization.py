
import cv2

from src.inference.types import Detection


COLORS = {
    # General
    "person": (255, 255, 0),
    "car": (255, 0, 255),
    "truck": (255, 0, 255),
    "bus": (255, 0, 255),
    "motorcycle": (255, 0, 255),
    "bicycle": (255, 0, 255),

    # PPE
    "Gloves": (255, 165, 0),
    "Hard_hat": (0, 255, 0),
    "Mask": (255, 255, 0),
    "Safety_boots": (255, 0, 255),
    "Vest": (255, 0, 0),

    # Fire and smoke
    "fire": (0, 0, 255),
    "smoke": (128, 128, 128),

    # Spill
    "spill": (0, 255, 255),
}


def draw_detections(
    frame,
    detections: list[Detection],
):
    output = frame.copy()

    for det in detections:
        color = COLORS.get(
            det.class_name,
            (255, 255, 255),
        )

        cv2.rectangle(
            output,
            (det.x1, det.y1),
            (det.x2, det.y2),
            color,
            2,
        )

        label = (
            f"{det.class_name}: {det.confidence:.2f}"
            f" [{det.model_name}]"
        )

        cv2.putText(
            output,
            label,
            (det.x1, max(20, det.y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return output
