"""Exercise real MP4 codecs for timing and mock capture for deterministic failures."""
from unittest.mock import patch
import cv2
import numpy as np
import pytest
from football_cv.video import VideoReader, VideoSink

def test_missing(tmp_path):
    with pytest.raises(ValueError, match='does not exist'):
        VideoReader(str(tmp_path / 'absent.mp4'), 64)

def test_synthetic_roundtrip(tmp_path):
    # Distinct constant frames provide a tiny, redistributable video with known timing.
    path = tmp_path / 'test.mp4'
    sink = VideoSink(str(path), 10, (128, 64))
    for i in range(5):
        sink.write(np.full((64, 128, 3), i * 40, dtype=np.uint8))
    sink.close()
    reader = VideoReader(str(path), 64)
    frames = []
    try:
        while (frame := reader.read()) is not None:
            frames.append(frame)
    finally:
        reader.close()
    assert [f.index for f in frames] == list(range(5))
    assert frames[0].source_size == (128, 64)
    assert frames[0].image.shape == (32, 64, 3)
    assert [f.timestamp for f in frames] == pytest.approx([0, .1, .2, .3, .4], abs=.01)
    with pytest.raises(ValueError, match='already exists'):
        VideoSink(str(path), 10, (128, 64))

def test_corrupt(tmp_path):
    path = tmp_path / 'bad.mp4'
    path.write_bytes(b'not a video')
    with pytest.raises(ValueError, match='Cannot open'):
        VideoReader(str(path), 64)

def test_decode_failure_and_release(tmp_path):
    # A mocked backend distinguishes zero decoded frames from an early mid-file failure.
    path = tmp_path / 'test.mp4'
    path.touch()
    with patch('football_cv.video.cv2.VideoCapture') as factory:
        cap = factory.return_value
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: 10}.get(prop, 0)
        cap.read.return_value = (False, None)
        reader = VideoReader(str(path), 64)
        with pytest.raises(ValueError, match='no frames'):
            reader.read()
        reader.index = 2
        with pytest.raises(ValueError, match='decode failed'):
            reader.read()
        reader.close()
        cap.release.assert_called_once()
