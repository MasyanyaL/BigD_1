# import logging
# import time
#
# import cv2
#
# from src.capture.video_capture import VideoSource
# from src.monitoring.fps import FPSMeter
#
# from src.inference.detector import Detector
# from src.inference.agregator import DetectionAggregator
# from src.inference.vizualization import draw_detections
#
#
# # --------------------------------------------------
# # 1. Создание общей модели
# # --------------------------------------------------
#
# general_detector = Detector(
#     model_path="models/yolo11n.pt",
#     confidence=0.4,
#     model_name="general",
#     allowed_classes={
#         "person",
#         "car",
#         "truck",
#         "bus",
#         "motorcycle",
#         "bicycle",
#     },
# )
#
#
# # --------------------------------------------------
# # 2. Создание специализированной PPE-модели
# # --------------------------------------------------
#
# ppe_detector = Detector(
#     model_path="models/best2.pt",
#     confidence=0.4,
#     model_name="ppe",
#     allowed_classes={
#         "Gloves",
#         "Hard_hat",
#         "Mask",
#         "Safety_boots",
#         "Vest",
#     },
# )
#
# fire_smoke_detector = Detector(
#     model_path="models/firesmokebest.pt",
#     confidence=0.4,
#     model_name="fire_smoke",
#     allowed_classes={
#         "fire",
#         "smoke",
#     },
# )
#
# # spill_detector = Detector(
# #     model_path="models/spill.pt",
# #     confidence=0.4,
# #     model_name="spill",
# #     allowed_classes={
# #         "spill",
# #     },
# # )
#
# # --------------------------------------------------
# # 3. Создание агрегатора
# # --------------------------------------------------
#
# aggregator = DetectionAggregator(
#     detectors={
#         "general": general_detector,
#         "ppe": ppe_detector,
#         "fire_smoke": fire_smoke_detector,
#         #"spill": spill_detector,
#     }
# )
#
#
# # --------------------------------------------------
# # 4. Настройка логирования
# # --------------------------------------------------
#
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)s %(message)s",
# )
#
#
# # --------------------------------------------------
# # 5. Основная функция
# # --------------------------------------------------
#
# def main() -> None:
#
#     source = VideoSource(
#         source="datav/fireandpeople2.mp4",
#         reconnect_sec=2.0,
#     )
#
#     fps_meter = FPSMeter(window=30)
#
#     try:
#         source.open()
#
#         # Получаем FPS исходного видео
#         fps_original = source.cap.get(cv2.CAP_PROP_FPS)
#
#         frame_delay = 0.0
#
#         if isinstance(source.source, str) and fps_original > 0:
#             frame_delay = 1.0 / fps_original
#
#         window_name = "Industrial Vision - Multi Model Detection"
#
#         cv2.namedWindow(
#             window_name,
#             cv2.WINDOW_NORMAL,
#         )
#
#         cv2.resizeWindow(
#             window_name,
#             800,
#             450,
#         )
#
#         while True:
#
#             start_time = time.perf_counter()
#
#             # --------------------------------------
#             # Получение кадра
#             # --------------------------------------
#
#             try:
#                 frame = source.read()
#
#             except EOFError:
#                 logging.info(
#                     "Воспроизведение видеофайла завершено"
#                 )
#                 break
#
#             if frame is None:
#                 continue
#
#             # --------------------------------------
#             # Изменение размера кадра
#             # --------------------------------------
#
#             height, width = frame.shape[:2]
#
#             if width > 800:
#
#                 new_width = 800
#
#                 new_height = int(
#                     height * (new_width / width)
#                 )
#
#                 frame = cv2.resize(
#                     frame,
#                     (new_width, new_height),
#                 )
#
#             # --------------------------------------
#             # Детекция несколькими моделями
#             # --------------------------------------
#
#             detections = aggregator.predict(frame)
#
#             # --------------------------------------
#             # Отрисовка всех обнаруженных объектов
#             # --------------------------------------
#
#             frame = draw_detections(
#                 frame,
#                 detections,
#             )
#
#             # --------------------------------------
#             # Подсчёт FPS
#             # --------------------------------------
#
#             fps = fps_meter.tick()
#
#             cv2.putText(
#                 frame,
#                 f"FPS: {fps:.2f}",
#                 (20, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1.0,
#                 (0, 0, 255),
#                 2,
#             )
#
#             # --------------------------------------
#             # Отображение кадра
#             # --------------------------------------
#
#             cv2.imshow(
#                 window_name,
#                 frame,
#             )
#
#             # --------------------------------------
#             # Сохранение скорости воспроизведения
#             # --------------------------------------
#
#             elapsed_time = time.perf_counter() - start_time
#
#             wait_ms = 1
#
#             if frame_delay > 0:
#
#                 wait_ms = max(
#                     1,
#                     int(frame_delay * 1000)
#                     - int(elapsed_time * 1000),
#                 )
#
#             key = cv2.waitKey(wait_ms) & 0xFF
#
#             if key == ord("q"):
#
#                 logging.info(
#                     "Выход по нажатию клавиши Q"
#                 )
#
#                 break
#
#     except Exception:
#
#         logging.exception(
#             "Ошибка при работе конвейера"
#         )
#
#     finally:
#
#         source.close()
#
#         cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     main()
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
    source = VideoSource(
        source="datav/dogcat.mp4",
        reconnect_sec=2.0,
    )

    detector = Detector(model_path="yolo11n.pt", confidence=0.4)

    fps_meter = FPSMeter(window=30)

    try:
        source.open()

        # FPS видео для задержки
        fps_original = source.cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 0.0
        if isinstance(source.source, str) and fps_original > 0:
            frame_delay = 1.0 / fps_original

        window_name = "Industrial Vision - Video + Detector"
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
                new_width = 800
                new_height = int(height * (new_width / width))
                frame = cv2.resize(frame, (new_width, new_height))

            # detection
            # jтправляем кадр в нейронку
            detections = detector.predict(frame)

            for det in detections:
                #прямоугольник (зеленый)
                cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)

                #текст: "person 0.85"
                label = f"{det.class_name} {det.confidence:.2f}"

                #фон для текста
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (det.x1, det.y1 - 20), (det.x1 + w, det.y1), (0, 255, 0), -1)

                #сам текст
                cv2.putText(frame, label, (det.x1, det.y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            fps = fps_meter.tick()

            #FPS на экран
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

            cv2.imshow(window_name, frame)

            elapsed_time = time.perf_counter() - start_time
            wait_ms = 1

            if frame_delay > 0:
                #секунды -> миллисекунды
                wait_ms = max(1, int(frame_delay * 1000) - int(elapsed_time * 1000))

            # waitKey заменяет time.sleep()
            key = cv2.waitKey(wait_ms) & 0xFF

            if key == ord("q"):
                logging.info("Выход по нажатию клавиши Q")
                break

    except Exception:
        logging.exception("Ошибка при работе конвейера")
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()