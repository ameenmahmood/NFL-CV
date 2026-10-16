from collections import deque

class PerformanceMeter:
    def __init__(self, window: int = 30):
        self.durations = deque(maxlen=window)

    def update(self, seconds: float) -> float:
        self.durations.append(max(seconds, 1e-9))
        return len(self.durations) / sum(self.durations)
