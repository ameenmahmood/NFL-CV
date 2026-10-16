"""Run reproducible, bounded real-video checks without opening a GUI."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from football_cv.config import Config, load_config
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
    parser.add_argument('--config', help='Detector/tracker YAML preset; defaults to CPU baseline')
    parser.add_argument('--starts', nargs='+', type=int, help='Segment start frame indices')
    parser.add_argument('--segment-frames', type=int, default=150)
    args = parser.parse_args()
    output = Path(args.output)
    if args.segment_frames < 1 or (args.starts and min(args.starts) < 0):
        parser.error('Segment length must be positive and start indices nonnegative')
    output.mkdir(parents=True, exist_ok=False)
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
    if fps <= 0 or count == 0 or count != expected:
        raise ValueError(f'Invalid or incomplete video: decoded {count}/{expected} frames, FPS={fps}')
    starts = args.starts if args.starts is not None else [0, count // 2, max(0, count - args.segment_frames)]
    if any(start >= count for start in starts) or len(set(starts)) != len(starts):
        raise ValueError('Segment starts must be unique and within the video')
    config = load_config(args.config) if args.config else Config(device='cpu', display=False)
    report = {'source': args.video, 'fps': fps, 'source_size': [width, height],
              'reported_frames': expected, 'decoded_frames': count,
              'model': config.model, 'imgsz': config.imgsz, 'processing_width': config.processing_width,
              'nms_iou': config.nms_iou, 'segments': []}
    print(f'Decoded {count}/{expected} frames ({count / fps:.2f}s)', flush=True)
    detector = PersonDetector(config)
    report['device'] = detector.device
    samples = []
    # Each segment is consecutive; fresh tracker state prevents linking across seeks.
    for start in starts:
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        tracker = PersonTracker(config.tracking, fps)
        cuts = CutDetector(config.cuts)
        sink = None
        visible, tracked, resets = [], [], []
        elapsed = []
        try:
            segment_length = min(args.segment_frames, count - start)
            for offset in range(segment_length):
                tick = perf_counter()
                ok, image = cap.read()
                if not ok:
                    raise ValueError(f'Decode failed in segment at {start + offset}')
                h, w = image.shape[:2]
                if w > config.processing_width:
                    image = cv2.resize(image, (config.processing_width, max(1, round(h * config.processing_width / w))), interpolation=cv2.INTER_AREA)
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
                    sink = VideoSink(str(output / f'segment_{start}.mp4'), fps, (image.shape[1], image.shape[0]))
                sink.write(annotated)
                if offset in {0, segment_length // 2, segment_length - 1}:
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
        while len(samples) % 3:
            samples.append(np.zeros_like(samples[-1]))
        cv2.imwrite(str(output / 'contact_sheet.jpg'), np.vstack([np.hstack(samples[i:i+3]) for i in range(0, len(samples), 3)]))
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
