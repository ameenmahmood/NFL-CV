"""Compare full-frame inference with overlapping crops on reviewed clean frames.

This is a diagnostic experiment, not the application's default detector. Partial
reference labels can measure annotated-box recall but cannot establish precision.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
from ultralytics import YOLO

from football_cv.evaluation import box_iou, load_annotations, score_frame, summarize
from football_cv.types import Detection


def tile_rectangles(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Cover the frame with four 60%-size crops (20% frame overlap per axis)."""
    if width < 1 or height < 1:
        raise ValueError('Frame dimensions must be positive')
    tw, th = max(1, round(width * .6)), max(1, round(height * .6))
    return [(x, y, x + tw, y + th) for y in sorted({0, height - th}) for x in sorted({0, width - tw})]


def merge_boxes(detections: list[Detection], threshold: float = .45) -> list[Detection]:
    """Score-ordered single-class NMS prevents crop/full-frame duplicates multiplying."""
    kept = []
    for candidate in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if all(box_iou(candidate.bbox, other.bbox) <= threshold for other in kept):
            kept.append(candidate)
    return kept


def predict(model, image) -> list[Detection]:
    """Keep native model scores; low-score observations remain available for review."""
    result = model.predict(image, classes=[0], conf=.1, imgsz=1280, iou=.45, device='cpu', verbose=False)[0]
    return [Detection(tuple(map(float, b)), float(s)) for b, s in
            zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())]


def tiled_predictions(model, image, full: list[Detection]) -> list[Detection]:
    """Add crop observations in source pixels, rejecting boxes cut by internal edges."""
    height, width = image.shape[:2]
    candidates = list(full)
    for x1, y1, x2, y2 in tile_rectangles(width, height):
        for detection in predict(model, image[y1:y2, x1:x2]):
            a, b, c, d = detection.bbox
            # Internal crop boundaries can truncate people. Another overlapping crop
            # or the full-frame pass should supply the complete observation instead.
            if ((x1 and a < 5) or (y1 and b < 5) or
                (x2 < width and c > x2 - x1 - 5) or (y2 < height and d > y2 - y1 - 5)):
                continue
            candidates.append(replace(detection, bbox=(a + x1, b + y1, c + x1, d + y1)))
    return merge_boxes(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', nargs='+', required=True)
    parser.add_argument('--videos', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if len(args.annotations) != len(args.videos):
        parser.error('Supply one video per annotation file, in the same order')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model = YOLO('yolo11n.pt')
    reports = []
    for annotation_path, video_path in zip(args.annotations, args.videos):
        data = load_annotations(annotation_path)
        with Path(video_path).open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        if not data.get('video_sha256') or digest != data['video_sha256']:
            raise ValueError('Video must match reviewed annotation checksum')
        cap = cv2.VideoCapture(video_path)
        rows = {'full': [], 'tiled': []}
        try:
            for frame in data['frames']:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame['index'])
                ok, image = cap.read()
                if not ok or [image.shape[1], image.shape[0]] != data['source_size']:
                    raise ValueError('Cannot read reviewed source frame at expected dimensions')
                tick = perf_counter()
                full = predict(model, image)
                full_seconds = perf_counter() - tick
                tick = perf_counter()
                tiled = tiled_predictions(model, image, full)
                tiled_seconds = full_seconds + perf_counter() - tick
                for name, predictions, seconds in [('full', full, full_seconds), ('tiled', tiled, tiled_seconds)]:
                    result = score_frame(frame['labels'], predictions, frame['exhaustive'])
                    result.update(frame=frame['index'], seconds=seconds,
                                  displayed_boxes=sum(d.confidence >= .25 for d in predictions))
                    rows[name].append(result)
                    preview = image.copy()
                    for label in frame['labels']:
                        color = (0, 0, 255) if label['id'] in result['missed_label_ids'] else (0, 200, 0)
                        a, b, c, d = map(round, label['bbox'])
                        cv2.rectangle(preview, (a, b), (c, d), color, 2)
                        cv2.putText(preview, label['id'], (a, max(20, b - 5)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
                    cv2.imwrite(str(output / f'{Path(annotation_path).stem}_{frame["index"]}_{name}.jpg'), preview)
            report = {'annotations': annotation_path, 'source_sha256': digest,
                      'full': summarize(rows['full']), 'tiled': summarize(rows['tiled']), 'frames': rows}
            reports.append(report)
            print(json.dumps({k: v for k, v in report.items() if k != 'frames'}, indent=2), flush=True)
        finally:
            cap.release()
    (output / 'tiling_benchmark.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
