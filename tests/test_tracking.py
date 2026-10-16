"""Exercise actual tracker state on synthetic detections, with no detector weights."""
import numpy as np
from football_cv.config import TrackingConfig, CutConfig
from football_cv.tracking import PersonTracker
from football_cv.types import Detection
from football_cv.scenes import CutDetector

def test_real_bytetrack_loss_and_reset():
    tracker = PersonTracker(TrackingConfig(), 30)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    d = Detection((10, 10, 30, 60), .9)
    first = tracker.update([d], image)[0]
    assert first.track_id is not None
    # One missing observation should be recoverable within the configured lost buffer.
    tracker.update([], image)
    recovered = tracker.update([d], image)[0]
    assert recovered.track_id == first.track_id
    tracker.reset()
    fresh = tracker.update([d], image)[0]
    # Numeric IDs can repeat; the scene namespace must distinguish the new identity.
    assert fresh.scene_id == first.scene_id + 1

def test_cut():
    # Identical frames stay in-scene; a black-to-white jump changes both cut signals.
    cuts = CutDetector(CutConfig())
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    assert not cuts.update(black)
    assert not cuts.update(black)
    assert cuts.update(np.full_like(black, 255))
