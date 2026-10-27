"""CLI and sequential pipeline orchestration; component logic lives in adapters."""
import argparse
import logging
import hashlib
import json
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
from .evaluation import load_annotations
from .movement import PlayerMonitor

log = logging.getLogger(__name__)

def run(config) -> int:
    """Process frames until EOF/quit and always release capture, writer, and windows."""
    if config.save_output and Path(config.output_path).exists():
        raise ValueError(f'Output already exists; choose a new path: {config.output_path}')
    if config.track_output and Path(config.track_output).exists():
        raise ValueError(f'Track output already exists: {config.track_output}')
    reference = load_annotations(config.focus_annotations) if config.focus_annotations else None
    if reference:
        with Path(config.input_path).open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        if not reference.get('video_sha256') or reference['video_sha256'] != digest:
            raise ValueError('Focus annotations require matching clean-video SHA-256')
    reader = VideoReader(config.input_path, config.processing_width)
    sink = None
    track_file = None
    processed = 0
    try:
        detector = PersonDetector(config)
        tracker = PersonTracker(config.tracking, reader.fps)
        monitor = PlayerMonitor(reference, max_idle_frames=max(1, round(config.tracking.track_buffer * reader.fps / 30))) if (reference or config.show_trails or config.track_output) else None
        if config.track_output:
            target = Path(config.track_output)
            target.parent.mkdir(parents=True, exist_ok=True)
            track_file = target.open('x', encoding='utf-8')
            track_file.write(json.dumps({'type': 'metadata', 'source': config.input_path,
                'fps': reader.fps, 'coordinates': 'source pixels and normalized image; not field coordinates',
                'focus_annotations': config.focus_annotations, 'model': config.model}) + '\n')
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
            if monitor is not None:
                detections = monitor.update(frame, detections, tracker.scene_id, config.confidence)
                if track_file:
                    track_file.write(json.dumps({'type': 'frame', 'frame': frame.index,
                        'timestamp': frame.timestamp, 'scene_id': tracker.scene_id,
                        'source_size': frame.source_size, 'observations': monitor.records}) + '\n')
            # Measure decode/inference/tracking, excluding rendering and UI waits.
            fps = meter.update(perf_counter() - start)
            image = annotate(frame, detections, fps, config.confidence, monitor=monitor, show_trails=config.show_trails)
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
        if track_file is not None:
            track_file.close()
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
    parser.add_argument('--trails', action='store_true', help='Draw measured image-space movement trails')
    parser.add_argument('--tracks-output', help='Write observations to a new JSONL file')
    parser.add_argument('--focus-annotations', help='Bind focus groups using reviewed clean-frame annotations')
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
        if args.trails:
            config.show_trails = True
        if args.tracks_output:
            config.track_output = args.tracks_output
        if args.focus_annotations:
            config.focus_annotations = args.focus_annotations
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
