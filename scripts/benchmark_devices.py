"""Compare warmed CPU/CUDA video processing at identical model/resolution settings.

Includes decode, resizing, scene checks, detection, tracking, and annotation. Excludes
GUI waits, video encoding, file export, and model/warmup startup. No frames are skipped.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
import ultralytics

from football_cv.config import load_config
from football_cv.detection import PersonDetector, device_description
from football_cv.scenes import CutDetector
from football_cv.tracking import PersonTracker
from football_cv.video import VideoReader
from football_cv.visualization import annotate


def synchronize(device: str) -> None:
    """CUDA kernels are asynchronous; finish them before reading wall-clock timers."""
    if device != 'cpu':
        torch.cuda.synchronize(int(device))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video')
    parser.add_argument('--config', default='configs/all22_hd.yaml')
    parser.add_argument('--devices', nargs='+', default=['cpu', '0'])
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 1:
        parser.error('Frames and warmup must be positive')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    base = load_config(args.config)
    records = []
    # Sequential runs avoid CPU/GPU contention between benchmark variants.
    for requested in args.devices:
        config = replace(base, device=requested)
        reader = VideoReader(args.video, config.processing_width)
        try:
            detector = PersonDetector(config)
            tracker = PersonTracker(config.tracking, reader.fps)
            cuts = CutDetector(config.cuts)
            loop_seconds, detection_seconds = [], []
            for index in range(args.warmup + args.frames):
                synchronize(detector.device)
                if index == args.warmup and detector.device != 'cpu':
                    torch.cuda.reset_peak_memory_stats(int(detector.device))
                tick = perf_counter()
                frame = reader.read()
                if frame is None:
                    raise ValueError('Video is too short for the requested consecutive benchmark frames')
                if cuts.update(frame.image):
                    tracker.reset()
                start_detection = perf_counter()
                detections = detector.detect(frame.image)
                synchronize(detector.device)
                detection_time = perf_counter() - start_detection
                tracked = tracker.update(detections, frame.image)
                annotate(frame, tracked, 0, config.confidence)
                synchronize(detector.device)
                elapsed = perf_counter() - tick
                if index >= args.warmup:
                    loop_seconds.append(elapsed)
                    detection_seconds.append(detection_time)
            result = {'requested_device': requested, 'selected_device': detector.device,
                      'hardware': device_description(detector.device), 'model': config.model,
                      'imgsz': config.imgsz, 'processing_width': config.processing_width,
                      'confidence': config.confidence, 'nms_iou': config.nms_iou,
                      'frames': len(loop_seconds), 'warmup_frames': args.warmup,
                      'source_fps': reader.fps, 'loop_fps': len(loop_seconds) / sum(loop_seconds),
                      'detection_fps': len(detection_seconds) / sum(detection_seconds),
                      'median_loop_ms': float(np.median(loop_seconds) * 1000),
                      'p95_loop_ms': float(np.percentile(loop_seconds, 95) * 1000),
                      'peak_torch_allocated_mib': torch.cuda.max_memory_allocated(int(detector.device)) / 2**20 if detector.device != 'cpu' else None}
            records.append(result)
            print(json.dumps(result, indent=2), flush=True)
        finally:
            reader.close()
    report = {'video': args.video, 'torch': torch.__version__, 'cuda_build': torch.version.cuda,
              'opencv': cv2.__version__, 'ultralytics': ultralytics.__version__, 'results': records,
              'scope': 'Warm consecutive decode/detect/track/annotate loop; not GUI/export throughput or accuracy.'}
    (output / 'device_benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
