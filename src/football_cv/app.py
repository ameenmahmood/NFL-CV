"""CLI and sequential pipeline orchestration; component logic lives in adapters."""
import argparse
import logging
from pathlib import Path
from time import perf_counter
import cv2
from .config import load_config
from .detection import PersonDetector
from .tracking import PersonTracker
from .video import VideoReader, VideoSink
from .scenes import CutDetector
from .performance import PerformanceMeter
from .visualization import annotate

log = logging.getLogger(__name__)

def run(config) -> int:
    """Process frames until EOF/quit and always release capture, writer, and windows."""
    if config.save_output and Path(config.output_path).exists():
        raise ValueError(f'Output already exists; choose a new path: {config.output_path}')
    reader = VideoReader(config.input_path, config.processing_width)
    sink = None
    processed = 0
    try:
        detector = PersonDetector(config)
        tracker = PersonTracker(config.tracking, reader.fps)
        cuts, meter = CutDetector(config.cuts), PerformanceMeter()
        log.info('Device: %s; source FPS: %.2f', detector.device, reader.fps)
        paused = False
        while True:
            if paused:
                # Keep window events responsive without advancing capture or output.
                key = cv2.waitKey(50) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key in (32, ord('p')):
                    paused = False
                if key == ord('r'):
                    tracker.reset()
                if cv2.getWindowProperty('Football CV', cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue
            start = perf_counter()
            frame = reader.read()
            if frame is None:
                break
            if cuts.update(frame.image):
                # Clear old identities before associating observations in the new view.
                tracker.reset()
                log.info('Scene reset at frame %d', frame.index)
            detections = tracker.update(detector.detect(frame.image), frame.image)
            # Measure decode/inference/tracking, excluding rendering and UI waits.
            fps = meter.update(perf_counter() - start)
            image = annotate(frame, detections, fps, config.confidence)
            if config.save_output:
                if sink is None:
                    # Actual processed dimensions are only known after the first read.
                    sink = VideoSink(config.output_path, reader.fps, (image.shape[1], image.shape[0]))
                sink.write(image)
            processed += 1
            if config.display:
                cv2.imshow('Football CV', image)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key in (32, ord('p')):
                    paused = True
                if key == ord('r'):
                    tracker.reset()
                    log.info('Manual scene reset at frame %d', frame.index)
                if cv2.getWindowProperty('Football CV', cv2.WND_PROP_VISIBLE) < 1:
                    break
        log.info('Processed %d frames', processed)
        return processed
    finally:
        # Also runs after model initialization, decode, or inference errors.
        reader.close()
        if sink is not None:
            sink.close()
        if config.display:
            cv2.destroyAllWindows()

def main() -> int:
    """Apply CLI overrides to YAML and translate expected failures to exit codes."""
    parser = argparse.ArgumentParser(description='Football person detection and ByteTrack tracking')
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--input')
    parser.add_argument('--output', help='Enable saving to this new MP4 file')
    parser.add_argument('--device')
    parser.add_argument('--no-display', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    try:
        config = load_config(args.config)
        if args.input:
            config.input_path = args.input
        if args.output:
            config.output_path, config.save_output = args.output, True
        if args.device:
            config.device = args.device
        if args.no_display:
            config.display = False
        # CLI overrides may invalidate settings that were valid when YAML was loaded.
        config.__post_init__()
        run(config)
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, cv2.error) as exc:
        log.error('%s', exc)
        return 1
    except KeyboardInterrupt:
        log.info('Interrupted')
        return 130
