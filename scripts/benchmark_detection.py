"""Compare detector presets on identical, evenly spaced real-video frames.

Counts and overlap diagnostics are not accuracy metrics; review saved images.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import cv2
from ultralytics import YOLO

from football_cv.types import Detection, Frame
from football_cv.visualization import annotate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f'Cannot open {args.video}')
    total, fps = int(cap.get(7)), cap.get(5)
    indices = [int(total * f) for f in (.05, .2, .35, .5, .7, .9)]
    frames = []
    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = cap.read()
            if not ok:
                raise ValueError(f'Cannot decode frame {index}')
            frames.append((index, image))
    finally:
        cap.release()
    records = []
    for model_name, width, imgsz, iou in (
        ('yolo11n.pt', 960, 640, .7),
        ('yolo11n.pt', 1920, 1280, .7),
        ('yolo11n.pt', 1920, 1280, .45),
        ('yolo11s.pt', 1920, 1280, .45),
    ):
        model = YOLO(model_name)
        label = f'{Path(model_name).stem}_{width}_{imgsz}_iou{iou}'
        # Warm up independently from timed inference for each preset.
        model.predict(frames[0][1], imgsz=imgsz, device='cpu', verbose=False)
        for index, original in frames:
            h, w = original.shape[:2]
            image = cv2.resize(original, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
            tick = perf_counter()
            result = model.predict(image, classes=[0], conf=.1, imgsz=imgsz,
                                   iou=iou, device='cpu', verbose=False)[0]
            seconds = perf_counter() - tick
            detections = [Detection(tuple(map(float, b)), float(s)) for b, s in
                          zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())]
            frame = Frame(index, index / fps, image, (w, h))
            annotated = annotate(frame, detections, 1 / seconds, .25)
            if not cv2.imwrite(str(output / f'{label}_frame{index}.jpg'), annotated):
                raise OSError('Could not save annotated comparison')
            record = {'preset': label, 'frame': index, 'seconds': seconds,
                      'count_at_025': sum(d.confidence >= .25 for d in detections),
                      'detections': [asdict(d) for d in detections]}
            records.append(record)
            print(f'{label} frame {index}: {record["count_at_025"]} people, {seconds:.3f}s', flush=True)
    (output / 'benchmark.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
