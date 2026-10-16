import pytest
from football_cv.config import Config, TrackingConfig, load_config

def test_defaults():
    assert load_config('configs/default.yaml').tracking.track_buffer == 30

@pytest.mark.parametrize('kwargs', [{'confidence': -1}, {'confidence': float('nan')}, {'imgsz': 0}, {'display': 'false'}, {'save_output': True, 'input_path': 'same.mp4', 'output_path': 'same.mp4'}])
def test_invalid(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)

@pytest.mark.parametrize('text', ['[]', 'unknown: 1', 'tracking: []', 'display: nope', 'a: ['])
def test_bad_yaml(tmp_path, text):
    path = tmp_path / 'config.yaml'
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)

def test_tracking_thresholds():
    with pytest.raises(ValueError):
        TrackingConfig(track_low_thresh=.5)
