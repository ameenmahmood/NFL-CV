"""Fixed-threshold person localization evaluation against reviewed role-group boxes.

Role groups belong to ground truth. Matching a person box does not mean the model
recognized that person's football position. This module does not compute AP/mAP.
"""
import json
import math
from pathlib import Path
from dataclasses import replace

from .types import Detection

GROUPS = ('defensive_priority', 'defensive_line', 'offensive_line',
          'offensive_skill', 'quarterback', 'other_person')


def box_iou(a: tuple | list, b: tuple | list) -> float:
    """Intersection over union for xyxy boxes; touching edges have zero area."""
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def load_annotations(path: str | Path) -> dict:
    """Reject drafts, ambiguous units, and malformed boxes before reporting metrics."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict) or type(data.get('version')) is not int or data['version'] != 1:
        raise ValueError('Annotation version must be 1')
    digest = data.get('video_sha256')
    if digest is not None and (not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('video_sha256 must be a lowercase SHA-256 hex digest')
    size = data.get('source_size')
    if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        raise ValueError('source_size must be positive integer [width, height]')
    if not isinstance(data.get('frames'), list) or not data['frames']:
        raise ValueError('At least one reviewed frame is required')
    seen = set()
    for frame in data['frames']:
        if not isinstance(frame, dict):
            raise ValueError('Each frame must be a mapping')
        index = frame.get('index')
        if type(index) is not int or index < 0 or index in seen:
            raise ValueError('Frame indices must be unique nonnegative integers')
        seen.add(index)
        if frame.get('review_status') != 'reviewed' or type(frame.get('exhaustive')) is not bool:
            raise ValueError('Frames require review_status=reviewed and boolean exhaustive')
        if not isinstance(frame.get('labels'), list):
            raise ValueError('Frame labels must be a list')
        ids = set()
        for label in frame['labels']:
            if not isinstance(label, dict) or label.get('group') not in GROUPS:
                raise ValueError('Unknown annotation role group')
            identifier, box = label.get('id'), label.get('bbox')
            if not isinstance(identifier, str) or not identifier or identifier in ids:
                raise ValueError('Label IDs must be nonempty and unique within a frame')
            ids.add(identifier)
            if (not isinstance(box, list) or len(box) != 4 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in box)):
                raise ValueError('bbox must contain four finite pixel coordinates')
            if not (0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1]):
                raise ValueError('bbox must have positive area within source_size')
    return data


def source_detections(predictions: list[Detection], source_size: tuple[int, int],
                      processed_size: tuple[int, int]) -> list[Detection]:
    """Undo resizing with separate axis scales to account for rounded frame heights."""
    if min(*source_size, *processed_size) <= 0:
        raise ValueError('Frame dimensions must be positive')
    sx, sy = source_size[0] / processed_size[0], source_size[1] / processed_size[1]
    return [replace(d, bbox=(d.bbox[0] * sx, d.bbox[1] * sy, d.bbox[2] * sx, d.bbox[3] * sy)) for d in predictions]


def score_frame(labels: list[dict], predictions: list[Detection], exhaustive: bool,
                confidence: float = .25, iou_threshold: float = .5) -> dict:
    """Greedily match predictions in score order, with each target used at most once.

    On partially labeled frames, unmatched predictions are unscored rather than
    false positives. Precision is therefore unavailable, while annotated-box recall
    remains measurable. The selected box sample may not represent all visible people.
    """
    if not 0 <= confidence <= 1 or not 0 < iou_threshold <= 1:
        raise ValueError('Invalid confidence or IoU threshold')
    selected = sorted((p for p in predictions if p.class_id == 0 and p.confidence >= confidence),
                      key=lambda p: p.confidence, reverse=True)
    available = set(range(len(labels)))
    matches, unmatched = [], []
    for prediction in selected:
        # Stable index order makes ties reproducible across runs/platforms.
        target = max(sorted(available), key=lambda i: box_iou(prediction.bbox, labels[i]['bbox']), default=None)
        overlap = box_iou(prediction.bbox, labels[target]['bbox']) if target is not None else 0
        if target is not None and overlap >= iou_threshold:
            available.remove(target)
            matches.append({'label_id': labels[target]['id'], 'group': labels[target]['group'],
                            'iou': overlap, 'prediction': list(prediction.bbox), 'confidence': prediction.confidence})
        else:
            unmatched.append(list(prediction.bbox))
    tp, fn = len(matches), len(available)
    groups = {}
    for group in GROUPS:
        total = sum(label['group'] == group for label in labels)
        found = sum(match['group'] == group for match in matches)
        groups[group] = {'annotated': total, 'matched': found, 'missed': total - found,
                         'recall': found / total if total else None}
    return {'true_positives': tp, 'false_negatives': fn,
            'false_positives': len(unmatched) if exhaustive else None,
            'unscored_predictions': 0 if exhaustive else len(unmatched),
            'precision': tp / len(selected) if exhaustive and selected else None,
            'recall': tp / len(labels) if labels else None, 'groups': groups,
            'matches': matches, 'missed_label_ids': [labels[i]['id'] for i in sorted(available)],
            'unmatched_boxes': unmatched}


def summarize(results: list[dict]) -> dict:
    """Aggregate counts before division; partial labeling disables overall precision."""
    groups = {}
    for group in GROUPS:
        total = sum(r['groups'][group]['annotated'] for r in results)
        found = sum(r['groups'][group]['matched'] for r in results)
        groups[group] = {'annotated': total, 'matched': found, 'missed': total - found,
                         'recall': found / total if total else None}
    tp = sum(r['true_positives'] for r in results)
    fn = sum(r['false_negatives'] for r in results)
    fp = sum(r['false_positives'] for r in results) if results and all(r['false_positives'] is not None for r in results) else None
    return {'true_positives': tp, 'false_negatives': fn, 'false_positives': fp,
            'precision': tp / (tp + fp) if fp is not None and tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None, 'groups': groups}
