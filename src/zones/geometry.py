import cv2
import numpy as np
from src.inference.types import Detection

def bottom_center(det: Detection) -> tuple[int, int]:
    """Возвращает нижнюю центральную точку рамки (ноги человека/колеса машины)."""
    return ((det.x1 + det.x2) // 2, det.y2)

def normalize_to_pixels(polygon: list[list[float]], width: int, height: int) -> np.ndarray:
    """Переводит координаты [0..1] из YAML в пиксели."""
    return np.array(
        [[int(pt[0] * width), int(pt[1] * height)] for pt in polygon],
        dtype=np.int32
    )

def point_in_polygon(point: tuple[int, int], polygon_px: np.ndarray) -> bool:
    """Проверяет, находится ли точка внутри полигона."""
    return cv2.pointPolygonTest(polygon_px, point, False) >= 0

def draw_polygon(frame, polygon_px: np.ndarray, color=(0, 0, 255), thickness=2):
    """Рисует зону на экране."""
    cv2.polylines(frame, [polygon_px], isClosed=True, color=color, thickness=thickness)