"""Evaluate a detector on reviewed boxes in clean source-video coordinates."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2

from football_cv.config import load_config
from football_cv.detection import PersonDetector
from football_cv.evaluation import load_annotations, score_frame, source_detections, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('annotations')
    parser.add_argument('--video', required=True)
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--match-iou', type=float, default=.5)
    args = parser.parse_args()
    dataset = load_annotations(args.annotations)
    if not 0 < args.match_iou <= 1:
        parser.error('--match-iou must be greater than zero and at most one')
    # Bind labels to the exact clean asset, never an annotated copy or another encode.
    with Path(args.video).open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    if dataset.get('video_sha256') and digest != dataset['video_sha256']:
        raise ValueError('Video checksum does not match annotation source')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    detector = PersonDetector(config)
    cap = cv2.VideoCapture(args.video)
    records = []
    try:
        if not cap.isOpened() or [int(cap.get(3)), int(cap.get(4))] != dataset['source_size']:
            raise ValueError('Video cannot be opened or dimensions do not match annotations')
        for frame in dataset['frames']:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame['index'])
            ok, original = cap.read()
            if not ok:
                raise ValueError(f'Cannot decode annotated frame {frame["index"]}')
            h, w = original.shape[:2]
            image = original
            if w > config.processing_width:
                image = cv2.resize(original, (config.processing_width, max(1, round(h * config.processing_width / w))), interpolation=cv2.INTER_AREA)
            tick = perf_counter()
            detections = detector.detect(image)
            elapsed = perf_counter() - tick
            # Scale back before IoU: annotations always refer to the clean source pixels.
            predictions = source_detections(detections, (w, h), (image.shape[1], image.shape[0]))
            scored = score_frame(frame['labels'], predictions, frame['exhaustive'], config.confidence, args.match_iou)
            scored.update(index=frame['index'], inference_seconds=elapsed)
            records.append(scored)
            preview = original.copy()
            for label in frame['labels']:
                # Green/red represent found/missed ground-truth boxes, not predicted roles.
                color = (0, 0, 255) if label['id'] in scored['missed_label_ids'] else (0, 200, 0)
                x1, y1, x2, y2 = map(round, label['bbox'])
                cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
                cv2.putText(preview, label['id'], (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
            if not cv2.imwrite(str(output / f'frame_{frame["index"]}_review.jpg'), preview):
                raise OSError('Could not save evaluation preview')
    finally:
        cap.release()
    report = {'annotations': args.annotations,
              'annotations_sha256': hashlib.sha256(Path(args.annotations).read_bytes()).hexdigest(),
              'source_sha256': digest, 'config': args.config,
              'model': config.model, 'device': detector.device, 'confidence': config.confidence,
              'imgsz': config.imgsz, 'nms_iou': config.nms_iou, 'match_iou': args.match_iou,
              'note': 'Person localization on reviewed boxes, not role classification or mAP. First inference includes warmup.',
              'summary': summarize(records), 'frames': records}
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
