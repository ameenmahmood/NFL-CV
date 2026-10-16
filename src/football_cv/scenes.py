"""Conservative hard-cut heuristic; manual resets cover undetected transitions."""
import cv2
import numpy as np
from .config import CutConfig

class CutDetector:
    """Compare adjacent frames; this is a heuristic, not guaranteed shot detection."""
    def __init__(self, config: CutConfig):
        self.config = config
        self.previous = None
        self.histogram = None

    def update(self, image) -> bool:
        # A fixed thumbnail bounds per-frame cost and makes input sizes comparable.
        small = cv2.resize(image, (96, 54))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        cut = False
        if self.previous is not None and self.config.enabled:
            distance = cv2.compareHist(self.histogram, hist, cv2.HISTCMP_BHATTACHARYYA)
            # Float subtraction avoids uint8 wraparound; normalize change to [0, 1].
            change = np.abs(small.astype(float) - self.previous.astype(float)).mean() / 255
            # Requiring both signals limits false resets from motion alone.
            cut = distance > self.config.histogram_threshold and change > self.config.pixel_threshold
        self.previous, self.histogram = small, hist
        return bool(cut)
