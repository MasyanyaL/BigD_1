from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

from src.events.storage import EventStorage
from src.events.temporal import TemporalConfirm
from src.inference.types import Detection
from src.zones.geometry import bottom_center, point_in_polygon

logger = logging.getLogger(__name__)


class EventEngine:
    """
    Аналитический слой: превращает детекции в события.

    Логика (этап 5 + 8):
      - интересуют person и vehicle с track_id и confidence >= порога;
      - временное подтверждение через TemporalConfirm (n из m);
      - cooldown между событиями для одного track_id;
      - сброс temporal при выходе из зоны (независимо от active);
      - отсутствие детекции трека в кадре = False в temporal (баг 2);
      - событие закрывается при выходе из зоны или после
        missing_frames_threshold кадров без детекции (баг 3);
      - запись события в EventStorage в формате events/YYYY/MM/DD/<uuid>/.
    """

    VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}

    def __init__(self, config: dict) -> None:
        events_cfg = config.get("events", {})
        self.n = int(events_cfg.get("confirmation_n", 3))
        self.m = int(events_cfg.get("confirmation_m", 5))
        self.cooldown = float(events_cfg.get("cooldown_sec", 10))
        self.track_ttl = float(events_cfg.get("track_ttl_sec", 60))
        self.missing_frames_threshold = int(
            events_cfg.get("missing_frames_threshold", 15)
        )

        if self.n > self.m:
            raise ValueError(f"confirmation_n ({self.n}) > confirmation_m ({self.m})")
        if self.n < 1 or self.m < 1:
            raise ValueError("confirmation_n и confirmation_m должны быть >= 1")

        models_cfg = config.get("models", {})
        self.person_confidence = float(models_cfg.get("person_confidence", 0.5))

        camera_cfg = config.get("camera", {})
        self.camera_id = camera_cfg.get("id", "cam_unknown")

        storage_cfg = config.get("storage", {})
        self.storage = EventStorage(
            root=storage_cfg.get("root", "events"),
            retention_days=int(storage_cfg.get("retention_days", 30)),
        )

        self.confirmers: dict[int, TemporalConfirm] = {}
        self.active_events: set[int] = set()
        self.last_event_time: dict[int, float] = {}
        self.last_seen: dict[int, float] = {}
        self.missing_frames: dict[int, int] = {}

        self._last_cleanup = time.monotonic()

    def process(
        self,
        detections: list[Detection],
        polygon_px: np.ndarray | None,
        frame=None,
    ) -> None:
        if polygon_px is None or len(polygon_px) < 3:
            return

        now = time.monotonic()
        seen_this_frame: set[int] = set()

        for det in detections:
            if det.track_id is None:
                continue
            if det.class_name == "person":
                if det.confidence < self.person_confidence:
                    continue
                event_type = "person_in_restricted_zone"
            elif det.class_name in self.VEHICLE_CLASSES:
                event_type = "vehicle_in_restricted_zone"
            else:
                continue

            tid = det.track_id
            seen_this_frame.add(tid)
            self.last_seen[tid] = now
            self.missing_frames[tid] = 0

            is_inside = point_in_polygon(bottom_center(det), polygon_px)

            if tid not in self.confirmers:
                self.confirmers[tid] = TemporalConfirm(n=self.n, m=self.m)

            # Баг 1: сброс при выходе, независимо от active
            if not is_inside:
                if tid in self.active_events:
                    self.active_events.discard(tid)
                    logger.info("track_id=%s вышел из зоны, событие закрыто", tid)
                self.confirmers[tid].reset()
                continue

            confirmed = self.confirmers[tid].update(True)

            if confirmed and tid not in self.active_events:
                last_ts = self.last_event_time.get(tid, 0.0)
                if now - last_ts >= self.cooldown:
                    self.active_events.add(tid)
                    self.last_event_time[tid] = now
                    self.emit_event(det, event_type, polygon_px, frame)

        # Баг 2: отсутствие детекции = False в temporal
        # Баг 3: закрытие события при потере трека
        for tid in list(self.confirmers.keys()):
            if tid in seen_this_frame:
                continue

            self.confirmers[tid].update(False)
            self.missing_frames[tid] = self.missing_frames.get(tid, 0) + 1

            if self.missing_frames[tid] >= self.missing_frames_threshold:
                if tid in self.active_events:
                    self.active_events.discard(tid)
                    logger.info(
                        "track_id=%s потерян (%d кадров), событие закрыто",
                        tid, self.missing_frames[tid],
                    )
                self.confirmers.pop(tid, None)
                self.missing_frames.pop(tid, None)
                self.last_event_time.pop(tid, None)
                self.last_seen.pop(tid, None)

        # очистка совсем старых треков (fallback)
        if now - self._last_cleanup > self.track_ttl:
            self._cleanup(now)
            self._last_cleanup = now

    def _cleanup(self, now: float) -> None:
        stale = [
            tid for tid, ts in self.last_seen.items()
            if now - ts > self.track_ttl
        ]
        for tid in stale:
            self.confirmers.pop(tid, None)
            self.active_events.discard(tid)
            self.last_event_time.pop(tid, None)
            self.last_seen.pop(tid, None)
            self.missing_frames.pop(tid, None)
        if stale:
            logger.debug("Очищено устаревших треков: %d", len(stale))

    def emit_event(
        self,
        det: Detection,
        event_type: str,
        polygon_px: np.ndarray,
        frame=None,
    ) -> None:
        # формат по 13.3 методички
        event = {
            "event_id": str(uuid4()),
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "camera_id": self.camera_id,
            "confidence": round(det.confidence, 4),
            "track_id": det.track_id,
            "coordinates": {
                "x": (det.x1 + det.x2) / 2,
                "y": det.y2,
                "bbox": [det.x1, det.y1, det.x2, det.y2],
                "polygon": polygon_px.tolist(),
            },
            "snapshot": None,
            "status": "new",
        }

        logger.warning(
            "СОБЫТИЕ: %s, track_id=%s, conf=%.2f",
            event_type, det.track_id, det.confidence,
        )

        try:
            self.storage.save(event, frame=frame)
        except Exception:
            logger.exception("Не удалось сохранить событие")