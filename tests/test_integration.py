"""Opt-in API smoke test on user-supplied footage and local pretrained weights."""
import os
from pathlib import Path
import pytest
from football_cv.config import Config, TrackingConfig
from football_cv.detection import PersonDetector
from football_cv.tracking import PersonTracker
from football_cv.video import VideoReader

@pytest.mark.integration
def test_local_video_and_weights():
    # Explicit local paths keep ordinary test runs offline and independent of video assets.
    video, weights = os.getenv('FOOTBALL_TEST_VIDEO'), os.getenv('FOOTBALL_TEST_MODEL')
    if not video or not weights:
        pytest.skip('Set FOOTBALL_TEST_VIDEO and FOOTBALL_TEST_MODEL to local files')
    assert Path(video).is_file() and Path(weights).is_file()
    reader = VideoReader(video, 640)
    try:
        detector = PersonDetector(Config(input_path=video, model=weights, device='cpu'))
        tracker = PersonTracker(TrackingConfig(), reader.fps)
        # Consecutive frames exercise detector-to-tracker compatibility, not accuracy.
        for _ in range(3):
            frame = reader.read()
            assert frame is not None
            people = tracker.update(detector.detect(frame.image), frame.image)
            assert all(d.class_id == 0 and 0 <= d.confidence <= 1 for d in people)
    finally:
        reader.close()
