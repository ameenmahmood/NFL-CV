"""Render person observations and processing metadata without changing input pixels."""
import cv2
from .types import Frame, Detection

def annotate(frame: Frame, detections: list[Detection], fps: float, threshold: float):
    """Draw only display-qualified observations; confidence is not formation certainty."""
    image = frame.image.copy()
    for d in detections:
        # Low-score observations may aid tracking without cluttering the overlay.
        if d.confidence < threshold:
            continue
        x1, y1, x2, y2 = map(round, d.bbox)
        label = f'person {d.confidence:.2f}'
        if d.track_id is not None:
            # Numeric track IDs can recur after scene resets.
            label = f'{d.scene_id}:{d.track_id} ' + label
        else:
            label = 'untracked ' + label
        cv2.rectangle(image, (x1, y1), (x2, y2), (40, 220, 60), 2)
        cv2.putText(image, label, (max(0, x1), max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (40, 220, 60), 1)
    cv2.rectangle(image, (0, 0), (image.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(image, f'Frame {frame.index} | {frame.timestamp:.3f}s | Processing {fps:.1f} FPS',
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    return image
