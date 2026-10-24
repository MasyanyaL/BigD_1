import logging
import time

import cv2

from src.capture.video_capture import VideoSource
from src.monitoring.fps import FPSMeter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def main() -> None:
    source = VideoSource(
        source= "datav/turkey-istanbul.mp4",
        reconnect_sec=2.0,
    )

    fps_meter = FPSMeter(window=30)

    try:
        source.open()

        #  оригинальный FPS видеофайла
        fps_original = source.cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 0.0

        # считаем нужную паузу
        if isinstance(source.source, str) and fps_original > 0:
            frame_delay = 1.0 / fps_original
            logging.info(f"Оригинальный FPS видео: {fps_original:.2f}. Пауза: {frame_delay:.3f} сек.")

        window_name = "Industrial Vision - Video"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        while True:
            # Засекаем время начала обработки кадра
            start_time = time.perf_counter()

            try:
                frame = source.read()
            except EOFError:
                logging.info("Воспроизведение видеофайла завершено")
                break

            fps = fps_meter.tick()

            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

            cv2.imshow(window_name, frame)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

            # для синхронизации времени
            if frame_delay > 0:
                elapsed_time = time.perf_counter() - start_time
                sleep_time = frame_delay - elapsed_time
                # обработка заняла меньше времени, чем длится кадр, спим
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except Exception:
        logging.exception("Ошибка при работе с видеопотоком")
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
