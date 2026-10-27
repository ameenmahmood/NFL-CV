"""Measured image-space paths and conservative focus from reviewed role references.

Role bindings follow ByteTrack IDs, so undetected identity swaps remain a limitation.
This is not a role classifier, camera-motion compensation, or field-speed estimate.
"""
from collections import deque

from .evaluation import score_frame, source_detections
from .types import Detection, Frame

FOCUS_GROUPS = {'defensive_priority', 'offensive_skill', 'quarterback'}


class PlayerMonitor:
    """Bound memory, expire missing identities, and never interpolate missing players."""
    def __init__(self, reference: dict | None = None, max_idle_frames: int = 30, history_size: int = 60):
        self.reference = reference
        self.anchors = {f['index']: f for f in reference['frames']} if reference else {}
        self.max_idle_frames = max_idle_frames
        self.history_size = history_size
        self.scene_id = None
        self.groups: dict[tuple[int, int], str] = {}
        self.history: dict[tuple[int, int], deque] = {}
        self.last_seen: dict[tuple[int, int], int] = {}
        self.records: list[dict] = []

    def update(self, frame: Frame, detections: list[Detection], scene_id: int,
               confidence: float) -> list[Detection]:
        """Bind roles only at exact reviewed frames; return current measured focus boxes."""
        if self.scene_id != scene_id:
            self.groups.clear()
            self.history.clear()
            self.last_seen.clear()
            self.scene_id = scene_id
        for key in list(self.last_seen):
            if frame.index - self.last_seen[key] > self.max_idle_frames:
                self.groups.pop(key, None)
                self.history.pop(key, None)
                self.last_seen.pop(key)
        measured = [d for d in detections if d.track_id is not None and d.confidence >= confidence]
        source = source_detections(measured, frame.source_size, (frame.image.shape[1], frame.image.shape[0]))
        if self.reference:
            if list(frame.source_size) != self.reference['source_size']:
                raise ValueError('Focus reference dimensions do not match source video')
            if frame.index in self.anchors:
                anchor = self.anchors[frame.index]
                # A new reviewed anchor replaces old bindings instead of accumulating stale roles.
                self.groups.clear()
                focused_labels = [label for label in anchor['labels'] if label['group'] in FOCUS_GROUPS]
                scored = score_frame(focused_labels, source, False, confidence)
                for match in scored['matches']:
                    self.groups[(match['scene_id'], match['track_id'])] = match['group']
        selected = []
        self.records = []
        width, height = frame.source_size
        for original, observation in zip(measured, source):
            key = (observation.scene_id, observation.track_id)
            if self.reference and key not in self.groups:
                continue
            selected.append(original)
            self.last_seen[key] = frame.index
            point = {'frame': frame.index, 'timestamp': frame.timestamp,
                     'footpoint': list(observation.footpoint),
                     'normalized_image': [observation.footpoint[0] / width, observation.footpoint[1] / height]}
            self.history.setdefault(key, deque(maxlen=self.history_size)).append(point)
            self.records.append({'scene_id': observation.scene_id, 'track_id': observation.track_id,
                                 'group': self.groups.get(key), 'group_source': 'reviewed_reference' if key in self.groups else None,
                                 'bbox_source_pixels': list(observation.bbox), 'confidence': observation.confidence,
                                 **point})
        # Generic monitoring still shows untracked detections; focus mode cannot
        # assign them a role without a reliable identity association.
        if not self.reference:
            selected.extend(d for d in detections if d.track_id is None and d.confidence >= confidence)
        return selected

    def trail_segments(self, key: tuple[int, int], size: tuple[int, int]) -> list[list[tuple[int, int]]]:
        """Break paths at detection gaps; scale normalized image points for display."""
        segments: list[list[tuple[int, int]]] = []
        previous = None
        for point in self.history.get(key, []):
            if previous is None or point['frame'] != previous + 1:
                segments.append([])
            x, y = point['normalized_image']
            segments[-1].append((round(x * size[0]), round(y * size[1])))
            previous = point['frame']
        return [segment for segment in segments if len(segment) >= 2]
