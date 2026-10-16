"""Run reproducible, bounded real-video checks without opening a GUI."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from football_cv.config import Config
from football_cv.detection import PersonDetector
from football_cv.scenes import CutDetector
from football_cv.tracking import PersonTracker
from football_cv.types import Frame
from football_cv.video import VideoSink
from football_cv.visualization import annotate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    parser.add_argument('--output', default='outputs/all22_validation')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError('Cannot open video')
    fps, expected = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(3)), int(cap.get(4))
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    report = {'source': args.video, 'fps': fps, 'source_size': [width, height],
              'reported_frames': expected, 'decoded_frames': count, 'segments': []}
    print(f'Decoded {count}/{expected} frames ({count / fps:.2f}s)', flush=True)
    config = Config(input_path=args.video, device='cpu', display=False)
    detector = PersonDetector(config)
    samples = []
    # Each segment is consecutive; fresh tracker state prevents linking across seeks.
    for start in (0, count // 2, max(0, count - 150)):
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        tracker = PersonTracker(config.tracking, fps)
        cuts = CutDetector(config.cuts)
        sink = None
        visible, tracked, resets = [], [], []
        elapsed = []
        try:
            for offset in range(min(150, count - start)):
                tick = perf_counter()
                ok, image = cap.read()
                if not ok:
                    raise ValueError(f'Decode failed in segment at {start + offset}')
                h, w = image.shape[:2]
                image = cv2.resize(image, (960, round(h * 960 / w)), interpolation=cv2.INTER_AREA)
                frame = Frame(start + offset, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, image, (w, h))
                if cuts.update(image):
                    tracker.reset()
                    resets.append(frame.index)
                people = tracker.update(detector.detect(image), image)
                elapsed.append(perf_counter() - tick)
                shown = [d for d in people if d.confidence >= config.confidence]
                visible.append(len(shown))
                tracked.append(sum(d.track_id is not None for d in shown))
                annotated = annotate(frame, people, 1 / elapsed[-1], config.confidence)
                if sink is None:
                    sink = VideoSink(str(output / f'segment_{start}.mp4'), fps, (960, image.shape[0]))
                sink.write(annotated)
                if offset in (0, 75, 149):
                    cv2.imwrite(str(output / f'frame_{frame.index}.jpg'), annotated)
                    samples.append(cv2.resize(annotated, (640, 360)))
        finally:
            cap.release()
            if sink:
                sink.close()
        stats = {'start_frame': start, 'frames': len(visible), 'processing_fps': len(elapsed) / sum(elapsed),
                 'visible_detections_min': min(visible), 'visible_detections_max': max(visible),
                 'visible_detections_mean': float(np.mean(visible)),
                 'tracked_detections_mean': float(np.mean(tracked)), 'cut_reset_frames': resets}
        report['segments'].append(stats)
        print(json.dumps(stats), flush=True)
    if samples:
        cv2.imwrite(str(output / 'contact_sheet.jpg'), np.vstack([np.hstack(samples[i:i+3]) for i in range(0, len(samples), 3)]))
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
