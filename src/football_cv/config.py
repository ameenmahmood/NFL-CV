"""Load strict YAML settings without importing the model or video pipeline."""
from dataclasses import dataclass, field, fields
from pathlib import Path
import math
import yaml

def probability(value: float, name: str) -> None:
    """Reject invalid thresholds, including bool (an int subclass) and NaN."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number between 0 and 1")

@dataclass
class TrackingConfig:
    """ByteTrack association settings; buffer is expressed in frames at 30 FPS."""
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    fuse_score: bool = True

    def __post_init__(self) -> None:
        for name in ('track_high_thresh', 'track_low_thresh', 'new_track_thresh', 'match_thresh'):
            probability(getattr(self, name), name)
        # Recovery may use weak detections; starting an identity needs a stronger one.
        if not 0 < self.track_low_thresh < self.track_high_thresh <= self.new_track_thresh:
            raise ValueError('Require 0 < low < high <= new track thresholds')
        if type(self.track_buffer) is not int or self.track_buffer < 1:
            raise ValueError('track_buffer must be a positive integer')
        if type(self.fuse_score) is not bool:
            raise ValueError('fuse_score must be boolean')

@dataclass
class CutConfig:
    """Independent histogram and pixel-change gates for the hard-cut heuristic."""
    enabled: bool = True
    histogram_threshold: float = 0.65
    pixel_threshold: float = 0.25

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError('cuts.enabled must be boolean')
        probability(self.histogram_threshold, 'histogram_threshold')
        probability(self.pixel_threshold, 'pixel_threshold')

@dataclass
class Config:
    """Runtime settings; relative paths resolve against the working directory."""
    input_path: str = 'data/test.mp4'
    output_path: str = 'outputs/annotated.mp4'
    save_output: bool = False
    display: bool = True
    model: str = 'yolo11n.pt'
    confidence: float = 0.25
    device: str = 'auto'
    processing_width: int = 960
    imgsz: int = 640
    nms_iou: float = 0.7
    show_trails: bool = False
    track_output: str | None = None
    focus_annotations: str | None = None
    # Each Config owns its nested settings; instances must not share mutable state.
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    cuts: CutConfig = field(default_factory=CutConfig)

    def __post_init__(self) -> None:
        probability(self.confidence, 'confidence')
        probability(self.nms_iou, 'nms_iou')
        if self.nms_iou == 0:
            raise ValueError('nms_iou must be greater than 0')
        for name in ('input_path', 'output_path', 'model', 'device'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f'{name} must be a nonempty string')
        for name in ('processing_width', 'imgsz'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 32:
                raise ValueError(f'{name} must be an integer >= 32')
        for name in ('save_output', 'display', 'show_trails'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be boolean')
        for name in ('track_output', 'focus_annotations'):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f'{name} must be a nonempty path or null')
        if self.track_output and Path(self.track_output).suffix.lower() != '.jsonl':
            raise ValueError('track_output must have .jsonl extension')
        if self.save_output and Path(self.input_path).resolve() == Path(self.output_path).resolve():
            raise ValueError('Output must not overwrite input')
        if self.save_output and Path(self.output_path).suffix.lower() != '.mp4':
            raise ValueError('Output must have .mp4 extension')

def construct(cls, data):
    """Reject misspelled YAML keys instead of silently applying default values."""
    if not isinstance(data, dict):
        raise ValueError(f'{cls.__name__} must be a YAML mapping')
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f'Unknown {cls.__name__} settings: {sorted(unknown)}')
    return cls(**data)

def load_config(path: str | Path) -> Config:
    """Parse safe YAML and validate nested settings before building the root config."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f'Cannot load configuration {path}: {exc}') from exc
    if not isinstance(data, dict):
        raise ValueError('Configuration must be a YAML mapping')
    data['tracking'] = construct(TrackingConfig, data.get('tracking', {}))
    data['cuts'] = construct(CutConfig, data.get('cuts', {}))
    return construct(Config, data)
