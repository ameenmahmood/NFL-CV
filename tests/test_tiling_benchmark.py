"""Check crop geometry and duplicate handling without model weights or inference."""
from pathlib import Path
import runpy

import numpy as np

from football_cv.types import Detection

helpers = runpy.run_path(str(Path(__file__).parents[1] / 'scripts' / 'benchmark_tiling.py'))


def test_tiles_cover_frame_and_stay_within_bounds():
    coverage = np.zeros((108, 192), dtype=bool)
    rectangles = helpers['tile_rectangles'](192, 108)
    assert len(rectangles) == 4
    for x1, y1, x2, y2 in rectangles:
        assert 0 <= x1 < x2 <= 192 and 0 <= y1 < y2 <= 108
        coverage[y1:y2, x1:x2] = True
    assert coverage.all()


def test_duplicate_merge_keeps_scores_and_neighbor():
    high = Detection((0, 0, 10, 20), .9)
    duplicate = Detection((1, 0, 11, 20), .5)
    neighbor = Detection((15, 0, 25, 20), .8)
    assert helpers['merge_boxes']([duplicate, high, neighbor]) == [high, neighbor]
