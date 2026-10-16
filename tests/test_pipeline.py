"""Check orchestration/resource ownership without loading weights or opening a GUI."""
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from football_cv.app import run
from football_cv.config import Config
from football_cv.detection import resolve_device
from football_cv.types import Frame
from football_cv.performance import PerformanceMeter

def test_device_selection():
    # Mock CUDA capabilities so CPU-only machines cover both hardware selection paths.
    with patch('torch.cuda.is_available', return_value=False):
        assert resolve_device('auto') == 'cpu'
        assert resolve_device('cpu') == 'cpu'
        with pytest.raises(ValueError, match='unavailable'):
            resolve_device('0')
    with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=1):
        assert resolve_device('auto') == '0'
        assert resolve_device('cuda:0') == '0'
    with pytest.raises(ValueError, match='device must'):
        resolve_device('bogus')

def test_headless_pipeline_cleanup():
    # One frame followed by EOF must yield one output write and release both resources.
    frame = Frame(0, 0, np.zeros((64, 64, 3), dtype=np.uint8), (64, 64))
    with patch('football_cv.app.VideoReader') as reader, patch('football_cv.app.PersonDetector') as detector, patch('football_cv.app.PersonTracker') as tracker, patch('football_cv.app.VideoSink') as sink:
        reader.return_value.fps = 30
        reader.return_value.read.side_effect = [frame, None]
        detector.return_value.detect.return_value = []
        tracker.return_value.update.return_value = []
        assert run(Config(display=False, save_output=True, output_path='outputs/unit-new.mp4')) == 1
        sink.return_value.write.assert_called_once()
        sink.return_value.close.assert_called_once()
        reader.return_value.close.assert_called_once()

def test_initialization_failure_cleanup():
    # Capture is already open when model loading fails, so the finally block must close it.
    with patch('football_cv.app.VideoReader') as reader, patch('football_cv.app.PersonDetector', side_effect=RuntimeError('model failed')):
        with pytest.raises(RuntimeError):
            run(Config(display=False))
        reader.return_value.close.assert_called_once()

def test_meter():
    # A two-frame window evicts old durations rather than averaging individual FPS values.
    meter = PerformanceMeter(2)
    assert meter.update(.1) == pytest.approx(10)
    assert meter.update(.3) == pytest.approx(5)
    assert meter.update(.1) == pytest.approx(5)
