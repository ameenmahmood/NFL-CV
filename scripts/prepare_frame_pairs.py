"""Export candidate clean/labeled frame pairs for manual annotation review.

Thumbnail similarity is a search aid, not a synchronization or label-transfer guarantee.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def thumbnail(image):
    """Use a shared grayscale grid, excluding borders and bottom playback controls."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (240, 135))[5:115, 5:235].astype(np.float32)


def best_candidate(reference, candidates):
    """Find minimum thumbnail error; clipping limits interference from colored letters."""
    errors = np.minimum(np.abs(candidates - reference), 40).mean(axis=(1, 2))
    index = int(np.argmin(errors))
    return index, float(errors[index])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('labeled')
    parser.add_argument('clean')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    clean, labeled = cv2.VideoCapture(args.clean), cv2.VideoCapture(args.labeled)
    try:
        if not clean.isOpened() or not labeled.isOpened():
            raise ValueError('Both videos must be readable')
        clean_fps, labeled_fps = clean.get(5), labeled.get(5)
        expected_clean, expected_labeled = int(clean.get(7)), int(labeled.get(7))
        # Store only thumbnails while decoding; avoid retaining full HD videos in RAM.
        candidates = []
        while True:
            ok, image = clean.read()
            if not ok:
                break
            candidates.append(thumbnail(image))
        if not candidates or len(candidates) != expected_clean or expected_labeled < 1:
            raise ValueError('Input is empty or clean video did not decode completely')
        candidates = np.stack(candidates)
        records = []
        indices = sorted({round((expected_labeled - 1) * fraction) for fraction in (0, .2, .4, .6, .8, 1)})
        for index in indices:
            labeled.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, reference = labeled.read()
            if not ok:
                raise ValueError(f'Cannot decode labeled frame {index}')
            match, error = best_candidate(thumbnail(reference), candidates)
            clean.set(cv2.CAP_PROP_POS_FRAMES, match)
            ok, unmarked = clean.read()
            if not ok:
                raise ValueError(f'Cannot decode clean frame {match}')
            # Export originals separately: annotation boxes must use clean image pixels.
            for name, image in ((f'labeled_{index}.png', reference), (f'clean_{match}.png', unmarked)):
                if not cv2.imwrite(str(output / name), image):
                    raise OSError(f'Cannot save {name}')
            preview = np.hstack([cv2.resize(image, (960, 540)) for image in (reference, unmarked)])
            cv2.imwrite(str(output / f'pair_{index}.jpg'), preview)
            records.append({'labeled_frame': index, 'clean_frame': match,
                            'thumbnail_error': error, 'requires_manual_review': True,
                            'clean_size': [unmarked.shape[1], unmarked.shape[0]]})
        report = {'labeled_source': args.labeled, 'clean_source': args.clean,
                  'labeled_fps': labeled_fps, 'clean_fps': clean_fps,
                  'decoded_clean_frames': len(candidates), 'candidate_pairs': records}
        (output / 'candidate_pairs.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
    finally:
        clean.release()
        labeled.release()


if __name__ == '__main__':
    main()
