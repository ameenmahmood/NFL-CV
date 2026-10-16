"""Coordinates are pixels in the processed frame; timestamps are seconds."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass(frozen=True)
class Frame:
    """Zero-based source index, seconds, BGR pixels, and source (width, height)."""
    index: int
    timestamp: float
    image: NDArray[np.uint8]
    source_size: tuple[int, int]

@dataclass(frozen=True)
class Detection:
    """An xyxy pixel box and person score, with an optional scene-scoped track ID."""
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int = 0
    track_id: int | None = None
    scene_id: int = 0

    @property
    def footpoint(self) -> tuple[float, float]:
        """Bottom center for future calibrated field mapping, not field coordinates."""
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)
