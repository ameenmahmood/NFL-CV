"""Render person observations and processing metadata without changing input pixels."""
import cv2
from .types import Frame, Detection

def annotate(frame: Frame, detections: list[Detection], fps: float, threshold: float, monitor=None, show_trails: bool = False):
    """Draw only display-qualified observations; confidence is not formation certainty."""
    image = frame.image.copy()
    for d in detections:
        # Low-score observations may aid tracking without cluttering the overlay.
        if d.confidence < threshold:
            continue
        x1, y1, x2, y2 = map(round, d.bbox)
        label = f'person {d.confidence:.2f}'
        if d.track_id is not None:
            key = (d.scene_id, d.track_id)
            if monitor is not None:
                if show_trails:
                    # Segments contain consecutive measurements only, never filled gaps.
                    for segment in monitor.trail_segments(key, (image.shape[1], image.shape[0])):
                        for a, b in zip(segment, segment[1:]):
                            cv2.line(image, a, b, (0, 200, 255), 2)
                if key in monitor.groups:
                    label = f'{monitor.groups[key]} [ref] {d.confidence:.2f}'
            # Numeric track IDs can recur after scene resets.
            label = f'{d.scene_id}:{d.track_id} ' + label
        else:
            label = 'untracked ' + label
        cv2.rectangle(image, (x1, y1), (x2, y2), (40, 220, 60), 2)
        cv2.putText(image, label, (max(0, x1), max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (40, 220, 60), 1)
    cv2.rectangle(image, (0, 0), (image.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(image, f'Frame {frame.index} | {frame.timestamp:.3f}s | Processing {fps:.1f} FPS',
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    if monitor is not None and monitor.reference:
        cv2.putText(image, f'Reviewed-group focus: {len(detections)} current tracks', (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 200, 255), 2)
    return image
