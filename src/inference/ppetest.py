import logging
import time
import cv2

from src.capture.video_capture import VideoSource
from src.monitoring.fps import FPSMeter
from src.inference.detector import Detector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def main() -> None:
    video_path = "datav/people.mp4"
    model_path = "models/best.pt"

    source = VideoSource(source=video_path, reconnect_sec=2.0)

    # Инициализируем детектор с нашей новой моделью
    detector = Detector(model_path=model_path, confidence=0.4)
    fps_meter = FPSMeter(window=30)

    try:
        source.open()

        # Настройка для правильной скорости воспроизведения
        fps_original = source.cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 0.0
        if isinstance(source.source, str) and fps_original > 0:
            frame_delay = 1.0 / fps_original

        window_name = "PPE Detection - Trained Model"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1000, 600)

        while True:
            start_time = time.perf_counter()

            try:
                frame = source.read()
            except EOFError:
                logging.info("Видео закончилось")
                break

            if frame is None:
                continue

            # Уменьшаем кадр
            height, width = frame.shape[:2]
            if width > 800:
                new_width = 800
                new_height = int(height * (new_width / width))
                frame = cv2.resize(frame, (new_width, new_height))

            # ДЕТЕКЦИЯ
            detections = detector.predict(frame)

            # ОТРИСОВКА
            for det in detections:
                # Разные цвета для разных классов (опционально)
                # Например, каска - зеленая, жилет - синяя
                color = (0, 255, 0) if det.class_name == "helmet" else (255, 0, 0)

                cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, 2)

                label = f"{det.class_name} {det.confidence:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                # Фон для текста
                cv2.rectangle(frame, (det.x1, det.y1 - 20), (det.x1 + w, det.y1), color, -1)
                # Сам текст (черный)
                cv2.putText(frame, label, (det.x1, det.y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # FPS
            fps = fps_meter.tick()
            cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            cv2.imshow(window_name, frame)

            # Задержка и обработка 'q'
            elapsed_time = time.perf_counter() - start_time
            wait_ms = 1
            if frame_delay > 0:
                wait_ms = max(1, int(frame_delay * 1000) - int(elapsed_time * 1000))

            if (cv2.waitKey(wait_ms) & 0xFF) == ord("q"):
                break

    except Exception:
        logging.exception("Ошибка при тестировании видео")
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()