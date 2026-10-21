"""Check the diagnostic matcher on distinct synthetic frames, without video assets."""
from pathlib import Path
import runpy

import numpy as np

# Load helper definitions without running the script's guarded command-line entry point.
helpers = runpy.run_path(str(Path(__file__).parents[1] / 'scripts' / 'prepare_frame_pairs.py'))


def test_matching_tolerates_small_overlay():
    rng = np.random.default_rng(42)
    candidates = rng.uniform(0, 255, (3, 110, 230)).astype(np.float32)
    reference = candidates[1].copy()
    reference[20:25, 30:35] = 255
    match, error = helpers['best_candidate'](reference, candidates)
    assert match == 1
    assert error > 0


def test_thumbnail_excludes_bottom_controls():
    image = np.zeros((135, 240, 3), dtype=np.uint8)
    image[120:] = 255
    result = helpers['thumbnail'](image)
    assert result.shape == (110, 230)
    assert not result.any()
