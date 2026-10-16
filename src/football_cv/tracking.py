from dataclasses import asdict, replace
from types import SimpleNamespace
import numpy as np
from .config import TrackingConfig
from .types import Detection

class PersonTracker:
    def __init__(self, config: TrackingConfig, fps: float):
        from ultralytics.trackers.byte_tracker import BYTETracker
        settings = asdict(config)
        # Current ByteTrack API accepts a frame count, not a source frame rate.
        settings['track_buffer'] = max(1, round(config.track_buffer * fps / 30))
        self.tracker = BYTETracker(SimpleNamespace(**settings))
        self.scene_id = 0

    def reset(self) -> None:
        self.tracker.reset()
        self.scene_id += 1

    def update(self, detections: list[Detection], image) -> list[Detection]:
        from ultralytics.engine.results import Boxes
        data = np.array([[*d.bbox, d.confidence, d.class_id] for d in detections], dtype=np.float32).reshape(-1, 6)
        tracks = self.tracker.update(Boxes(data, image.shape[:2]), image)
        ids = {int(row[7]): int(row[4]) for row in tracks}
        return [replace(d, track_id=ids.get(i), scene_id=self.scene_id) for i, d in enumerate(detections)]
