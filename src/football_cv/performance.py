"""Rolling throughput for the processing intervals supplied by the caller."""
from collections import deque

class PerformanceMeter:
    """Use frames/total time rather than averaging unstable instantaneous FPS."""
    def __init__(self, window: int = 30):
        self.durations = deque(maxlen=window)

    def update(self, seconds: float) -> float:
        # Guard against timer granularity producing a zero-duration interval.
        self.durations.append(max(seconds, 1e-9))
        return len(self.durations) / sum(self.durations)
