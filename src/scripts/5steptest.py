import logging
import time
import yaml

import cv2

from src.capture.video_capture import VideoSource
from src.monitoring.fps import FPSMeter
from src.inference.detector import Detector
from src.inference.agregator import DetectionAggregator
from src.inference.vizualization import draw_detections
from src.zones.geometry import normalize_to_pixels, draw_polygon
from src.events.engine import EventEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    # 1. Конфигурация
    with open("configs/configuration.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. Детекторы
    #    general — с tracking (нужен Track ID для EventEngine)
    general_detector = Detector(
        model_path="models/yolo11n.pt", confidence=0.4,
        model_name="general", allowed_classes={"person"},
        use_tracking=True,
    )
    ppe_detector = Detector(model_path="models/best2.pt", confidence=0.4, model_name="ppe")
    fire_detector = Detector(model_path="models/firesmokebest.pt", confidence=0.4, model_name="fire_smoke")
    spill_detector = Detector(model_path="models/spill.pt", confidence=0.15, model_name="spill")

    aggregator = DetectionAggregator(
        detectors={"general": general_detector, "ppe": ppe_detector, "fire_smoke": fire_detector, "spill": spill_detector}
    )

    # 3. EventEngine (читает events/models/storage прямо из config)
    event_engine = EventEngine(config)

    restricted_zone_norm = config.get("zones", {}).get("restricted_01", [])

    source = VideoSource(source="datav/skater.mp4", reconnect_sec=2.0)
    fps_meter = FPSMeter(window=30)

    try:
        source.open()
        fps_original = source.cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps_original if fps_original > 0 else 0.0

        window_name = "Industrial Vision - Stage 5"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 450)

        while True:
            start_time = time.perf_counter()
            try:
                frame = source.read()
            except EOFError:
                logging.info("Воспроизведение видеофайла завершено")
                break

            if frame is None:
                continue

            height, width = frame.shape[:2]

            if width > 800:
                frame = cv2.resize(frame, (800, int(height * (800 / width))))
                height, width = frame.shape[:2]

            polygon_px = None
            if restricted_zone_norm:
                polygon_px = normalize_to_pixels(restricted_zone_norm, width, height)

            detections = aggregator.predict(frame)
            frame = draw_detections(frame, detections)

            if polygon_px is not None and len(polygon_px) >= 3:
                draw_polygon(frame, polygon_px, color=(0, 0, 255), thickness=2)
                event_engine.process(detections, polygon_px, frame)

            fps = fps_meter.tick()
            cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow(window_name, frame)

            elapsed_time = time.perf_counter() - start_time
            wait_ms = max(1, int(frame_delay * 1000) - int(elapsed_time * 1000)) if frame_delay > 0 else 1
            if (cv2.waitKey(wait_ms) & 0xFF) == ord("q"):
                break

    except Exception:
        logging.exception("Ошибка при работе конвейера")
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()