"""Protect one-to-one localization counts and avoid invented precision on partial labels."""
import json

import pytest

from football_cv.evaluation import box_iou, load_annotations, score_frame, source_detections, summarize
from football_cv.types import Detection


def target(identifier='d1', group='defensive_priority', box=None):
    """Return a small independent annotation record for each scenario."""
    return {'id': identifier, 'group': group, 'bbox': box or [0, 0, 10, 10]}


def test_overlap_geometry():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert box_iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_duplicates_cannot_inflate_recall():
    # Only the highest-score box gets the target; the duplicate is a false positive.
    predictions = [Detection((0, 0, 10, 10), .5), Detection((0, 0, 10, 10), .9)]
    result = score_frame([target()], predictions, True)
    assert result['true_positives'] == 1
    assert result['false_positives'] == 1
    assert result['precision'] == .5
    assert result['recall'] == 1
    assert result['matches'][0]['confidence'] == .9


def test_partial_labels_do_not_make_other_people_false_positives():
    predictions = [Detection((0, 0, 10, 10), .9), Detection((20, 0, 30, 10), .8)]
    result = score_frame([target()], predictions, False)
    assert result['false_positives'] is None
    assert result['precision'] is None
    assert result['unscored_predictions'] == 1
    assert result['groups']['defensive_line']['recall'] is None
    assert summarize([result])['precision'] is None


def test_misses_threshold_and_nonperson_filter():
    predictions = [Detection((0, 0, 10, 10), .2), Detection((0, 0, 10, 10), .9, class_id=2)]
    result = score_frame([target()], predictions, True)
    assert result['false_negatives'] == 1
    assert result['missed_label_ids'] == ['d1']
    assert result['recall'] == 0


def test_groups_follow_ground_truth_not_predicted_roles():
    labels = [target(), target('q1', 'quarterback', [20, 0, 30, 10])]
    result = score_frame(labels, [Detection((20, 0, 30, 10), .9)], True)
    assert result['groups']['quarterback']['recall'] == 1
    assert result['groups']['defensive_priority']['missed'] == 1


def test_aggregate_counts_before_dividing():
    first = score_frame([target()], [Detection((0, 0, 10, 10), .9)], True)
    second = score_frame([target('a'), target('b', box=[20, 0, 30, 10])], [], True)
    assert summarize([first, second])['recall'] == pytest.approx(1 / 3)


@pytest.mark.parametrize('mutation', ['draft', 'bounds', 'duplicate', 'nan', 'group', 'exhaustive'])
def test_invalid_annotation_rejected(tmp_path, mutation):
    data = {'version': 1, 'source_size': [100, 100], 'frames': [
        {'index': 0, 'review_status': 'reviewed', 'exhaustive': False, 'labels': [target()]}]}
    frame = data['frames'][0]
    if mutation == 'draft':
        frame['review_status'] = 'draft'
    elif mutation == 'bounds':
        frame['labels'][0]['bbox'][2] = 101
    elif mutation == 'duplicate':
        frame['labels'].append(target())
    elif mutation == 'nan':
        frame['labels'][0]['bbox'][0] = float('nan')
    elif mutation == 'group':
        frame['labels'][0]['group'] = 'invented'
    else:
        frame['exhaustive'] = 'false'
    path = tmp_path / 'labels.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_annotations(path)


def test_empty_reference_and_predictions():
    result = score_frame([], [], True)
    assert result['recall'] is None
    assert result['precision'] is None
    assert result['false_positives'] == 0


def test_source_coordinates_restore_each_axis_and_preserve_metadata():
    detection = Detection((1, 2, 10, 20), .9, track_id=3, scene_id=2)
    restored = source_detections([detection], (200, 300), (100, 100))[0]
    assert restored.bbox == (2, 6, 20, 60)
    assert restored.track_id == 3 and restored.scene_id == 2
    assert detection.bbox == (1, 2, 10, 20)


def test_committed_priority_sample_is_valid():
    from pathlib import Path
    data = load_annotations(Path(__file__).parents[1] / 'annotations' / 'set_1_priority_sample.json')
    assert sum(len(f['labels']) for f in data['frames']) == 14
    assert all(not f['exhaustive'] for f in data['frames'])
