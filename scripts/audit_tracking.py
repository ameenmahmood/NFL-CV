"""Run every intervening frame before comparing manually linked identity anchors."""
import argparse
import hashlib
import json
from pathlib import Path

from football_cv.config import load_config
from football_cv.detection import PersonDetector
from football_cv.evaluation import load_annotations, score_frame, source_detections
from football_cv.scenes import CutDetector
from football_cv.tracking import PersonTracker
from football_cv.tracking_metrics import anchor_continuity
from football_cv.video import VideoReader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('links')
    parser.add_argument('--video', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    links_path = Path(args.links)
    links = json.loads(links_path.read_text(encoding='utf-8'))
    data = load_annotations(links_path.parent / links['annotations'])
    frames = {f['index']: f for f in data['frames']}
    indices, identities = links['frames'], links['person_to_label_id']
    if (links.get('version') != 1 or len(indices) < 2 or len(set(indices)) != len(indices)
        or len(set(identities.values())) != len(identities) or not identities):
        raise ValueError('Require unique anchors and one-to-one reviewed identity links')
    for index in indices:
        if index not in frames or not set(identities.values()) <= {label['id'] for label in frames[index]['labels']}:
            raise ValueError('Linked labels must exist at every requested anchor')
    with Path(args.video).open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    if not data.get('video_sha256') or data['video_sha256'] != digest:
        raise ValueError('Tracking anchors require matching clean-video SHA-256')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    reader = VideoReader(args.video, config.processing_width)
    observations = {person: [] for person in identities}
    resets = []
    try:
        detector = PersonDetector(config)
        tracker = PersonTracker(config.tracking, reader.fps)
        cuts = CutDetector(config.cuts)
        while True:
            frame = reader.read()
            if frame is None:
                raise ValueError('Video ended before final anchor')
            if list(frame.source_size) != data['source_size']:
                raise ValueError('Source dimensions differ from reviewed annotations')
            if cuts.update(frame.image):
                tracker.reset()
                resets.append(frame.index)
            tracked = tracker.update(detector.detect(frame.image), frame.image)
            if frame.index in indices:
                predictions = source_detections(tracked, frame.source_size, (frame.image.shape[1], frame.image.shape[0]))
                scored = score_frame(frames[frame.index]['labels'], predictions, False, config.confidence)
                by_label = {m['label_id']: m for m in scored['matches']}
                for person, label in identities.items():
                    match = by_label.get(label)
                    key = [match['scene_id'], match['track_id']] if match and match['track_id'] is not None else None
                    observations[person].append({'frame': frame.index, 'track_key': key})
            if frame.index >= max(indices):
                break
    finally:
        reader.close()
    report = {'config': args.config, 'source_sha256': digest, 'frames_processed': frame.index + 1,
              'cut_frames': resets, 'observations': observations, 'summary': anchor_continuity(observations)}
    (output / 'tracking_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
