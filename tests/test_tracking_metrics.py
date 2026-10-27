"""Sparse anchors must not hide missing people or count cuts as identity failures."""
from football_cv.tracking_metrics import anchor_continuity


def test_anchor_outcomes():
    observations = {
        'same': [{'frame': 1, 'track_key': [0, 1]}, {'frame': 3, 'track_key': [0, 1]}],
        'changed': [{'frame': 1, 'track_key': [0, 2]}, {'frame': 3, 'track_key': [0, 3]}],
        'missing': [{'frame': 1, 'track_key': None}, {'frame': 3, 'track_key': [0, 4]}],
        'cut': [{'frame': 1, 'track_key': [0, 5]}, {'frame': 3, 'track_key': [1, 5]}]}
    result = anchor_continuity(observations)
    assert result['preserved_pairs'] == result['changed_id_pairs'] == 1
    assert result['unobserved_pairs'] == result['scene_reset_pairs'] == 1
    assert result['anchor_preservation'] == .5


def test_no_comparable_anchors():
    result = anchor_continuity({'a': [{'frame': 0, 'track_key': None}, {'frame': 1, 'track_key': None}]})
    assert result['anchor_preservation'] is None
