"""Detector boundary tests without weights, downloads, or GPU access."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from football_cv.config import Config, load_config
from football_cv.detection import PersonDetector


def test_hd_preset_retains_full_resolution():
    config = load_config('configs/all22_hd.yaml')
    assert config.processing_width == 1920
    assert config.imgsz == 1280
    assert config.nms_iou == .45


@pytest.mark.parametrize('value', [-.1, 0, 1.1, float('nan'), True, '0.45'])
def test_invalid_suppression(value):
    with pytest.raises(ValueError):
        Config(nms_iou=value)


def test_detection_preserves_tracking_candidates_and_filters_nonpeople():
    detector = PersonDetector.__new__(PersonDetector)
    detector.config = Config(imgsz=1280, nms_iou=.45)
    detector.device = 'cpu'
    detector.model = MagicMock()
    boxes = MagicMock()
    boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[1, 2, 10, 20], [20, 2, 30, 20], [40, 2, 50, 20]])
    boxes.conf.cpu.return_value.numpy.return_value = np.array([.9, .15, .8])
    boxes.cls.cpu.return_value.numpy.return_value = np.array([0, 0, 2])
    detector.model.predict.return_value = [MagicMock(boxes=boxes)]
    people = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert len(people) == 2
    assert people[1].confidence == .15  # Needed for ByteTrack low-score recovery.
    assert people[0].bbox == (1, 2, 10, 20)
    options = detector.model.predict.call_args.kwargs
    assert options['iou'] == .45
    assert options['imgsz'] == 1280
    assert options['conf'] == .1
    assert options['classes'] == [0]


def test_no_detection_boxes():
    detector = PersonDetector.__new__(PersonDetector)
    detector.config, detector.device, detector.model = Config(), 'cpu', MagicMock()
    detector.model.predict.return_value = [MagicMock(boxes=None)]
    assert detector.detect(np.zeros((100, 100, 3), dtype=np.uint8)) == []
