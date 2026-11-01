from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


class EventStorage:
    """
    Сохраняет events/YYYY/MM/DD/<event_uuid>/{event.json, snapshot.jpg}.
    При инициализации удаляет папки старше retention_days.
    """

    def __init__(self, root: str | Path, retention_days: int = 30) -> None:
        self.root = Path(root)
        self.retention_days = int(retention_days)
        self.root.mkdir(parents=True, exist_ok=True)
        self._purge_old()

    def save(self, event: dict, frame=None) -> Path:
        ts = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
        event_dir = self.root / f"{ts:%Y}" / f"{ts:%m}" / f"{ts:%d}" / event["event_id"]
        event_dir.mkdir(parents=True, exist_ok=True)

        if frame is not None:
            snap = event_dir / "snapshot.jpg"
            cv2.imwrite(str(snap), frame)
            event["snapshot"] = str(snap.relative_to(self.root))

        with open(event_dir / "event.json", "w", encoding="utf-8") as f:
            json.dump(event, f, ensure_ascii=False, indent=2)

        return event_dir

    def _purge_old(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        for year_dir in self.root.glob("*"):
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue
            for month_dir in year_dir.glob("*"):
                if not month_dir.is_dir() or not month_dir.name.isdigit():
                    continue
                for day_dir in month_dir.glob("*"):
                    if not day_dir.is_dir() or not day_dir.name.isdigit():
                        continue
                    try:
                        d = datetime(
                            int(year_dir.name), int(month_dir.name),
                            int(day_dir.name), tzinfo=timezone.utc,
                        )
                    except ValueError:
                        continue
                    if d < cutoff:
                        try:
                            shutil.rmtree(day_dir)
                            logger.info("Удалён устаревший день: %s", day_dir)
                        except OSError:
                            logger.exception("Не удалось удалить %s", day_dir)