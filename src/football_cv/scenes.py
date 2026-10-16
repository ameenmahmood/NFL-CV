"""Conservative hard-cut heuristic; manual resets cover undetected transitions."""
import cv2
import numpy as np
from .config import CutConfig

class CutDetector:
    def __init__(self, config: CutConfig):
        self.config = config
        self.previous = None
        self.histogram = None

    def update(self, image) -> bool:
        small = cv2.resize(image, (96, 54))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        cut = False
        if self.previous is not None and self.config.enabled:
            distance = cv2.compareHist(self.histogram, hist, cv2.HISTCMP_BHATTACHARYYA)
            change = np.abs(small.astype(float) - self.previous.astype(float)).mean() / 255
            cut = distance > self.config.histogram_threshold and change > self.config.pixel_threshold
        self.previous, self.histogram = small, hist
        return bool(cut)
