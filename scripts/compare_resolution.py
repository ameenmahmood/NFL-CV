"""Compare detection resolution on identical frames from a supplied video."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from football_cv.config import Config
from football_cv.detection import PersonDetector
from football_cv.types import Frame
from football_cv.visualization import annotate

parser = argparse.ArgumentParser()
parser.add_argument('video')
parser.add_argument('--output', default='outputs/all22_validation')
args = parser.parse_args()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
detector = PersonDetector(Config(input_path=args.video, device='cpu'))
cap = cv2.VideoCapture(args.video)
comparisons, rows = [], []
try:
    for index in (0, 2657, 5090):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, original = cap.read()
        if not ok:
            raise ValueError(f'Cannot decode frame {index}')
        row = []
        for width, imgsz in ((960, 640), (1920, 1280)):
            detector.config.imgsz = imgsz
            image = cv2.resize(original, (width, round(original.shape[0] * width / original.shape[1])))
            start = perf_counter()
            detections = detector.detect(image)
            seconds = perf_counter() - start
            shown = sum(d.confidence >= .25 for d in detections)
            comparisons.append({'frame': index, 'processing_width': width, 'imgsz': imgsz,
                                'detections_at_025': shown, 'inference_seconds': seconds})
            annotated = annotate(Frame(index, index / 30, image, (1920, 1080)), detections, 1 / seconds, .25)
            cv2.imwrite(str(output / f'comparison_{index}_{imgsz}.jpg'), annotated)
            preview = cv2.resize(annotated, (960, 540))
            cv2.putText(preview, f'Input {width}, YOLO {imgsz}: {shown} people', (10, 530), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
            row.append(preview)
        rows.append(np.hstack(row))
finally:
    cap.release()
cv2.imwrite(str(output / 'resolution_comparison.jpg'), np.vstack(rows))
(output / 'resolution_comparison.json').write_text(json.dumps(comparisons, indent=2))
print(json.dumps(comparisons, indent=2))
