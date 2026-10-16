"""OpenCV capture/output with explicit timing, dimensions, and resource ownership."""
import logging
import math
from pathlib import Path
import cv2
from .types import Frame

log = logging.getLogger(__name__)

class VideoReader:
    """Read every source frame, shrinking only when it exceeds the width limit."""
    def __init__(self, path: str, width: int):
        if not Path(path).is_file():
            raise ValueError(f'Video file does not exist: {path}')
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            self.cap.release()
            raise ValueError(f'Cannot open video (corrupted or unsupported codec): {path}')
        self.width = width
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(self.fps) or self.fps <= 0:
            self.cap.release()
            raise ValueError('Video has no usable frame rate; cannot preserve timing')
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.index = 0
        self.last_timestamp = -1.0

    def read(self) -> Frame | None:
        """Return a timestamped frame, None at EOF, or raise on detectable decode loss."""
        ok, image = self.cap.read()
        if not ok or image is None:
            if self.index == 0:
                raise ValueError('Video opened but no frames could be decoded')
            # With a known frame count, an early failure is not treated as clean EOF.
            if self.count > 0 and self.index < self.count:
                raise ValueError(f'Video decode failed at frame {self.index} of {self.count}')
            return None
        h, w = image.shape[:2]
        timestamp = float(self.cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
        # Some backends return missing/repeated timestamps; keep fallback time monotonic.
        if not math.isfinite(timestamp) or timestamp < 0 or timestamp <= self.last_timestamp:
            timestamp = max(self.index / self.fps, self.last_timestamp + 1 / self.fps)
        self.last_timestamp = timestamp
        if w > self.width:
            image = cv2.resize(image, (self.width, max(1, round(h * self.width / w))), interpolation=cv2.INTER_AREA)
        # Retain source size so processed pixel coordinates can be rescaled later.
        frame = Frame(self.index, timestamp, image, (w, h))
        self.index += 1
        return frame

    def close(self) -> None:
        self.cap.release()

class VideoSink:
    """Write a new constant-rate, video-only MP4 with fixed frame dimensions."""
    def __init__(self, path: str, fps: float, size: tuple[int, int]):
        target = Path(path)
        if target.exists():
            raise ValueError(f'Output already exists; choose a new path: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        if not self.writer.isOpened():
            self.writer.release()
            raise ValueError(f'Cannot create MP4 output: {target}')
        self.size = size

    def write(self, image) -> None:
        if (image.shape[1], image.shape[0]) != self.size:
            raise ValueError('Video frame dimensions changed during output')
        self.writer.write(image)

    def close(self) -> None:
        # Finalize the MP4 container even when processing stopped before source EOF.
        self.writer.release()
