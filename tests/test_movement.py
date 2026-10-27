"""Movement exports must preserve observed coordinates, gaps, and manual role provenance."""
import numpy as np

from football_cv.movement import PlayerMonitor
from football_cv.types import Detection, Frame


def frame(index):
    return Frame(index, index / 30, np.zeros((100, 200, 3), dtype=np.uint8), (400, 200))


def person(track_id=1, scene_id=0):
    return Detection((20, 20, 40, 50), .9, track_id=track_id, scene_id=scene_id)


def reference():
    # Reference pixels are twice the processing dimensions on both axes.
    return {'source_size': [400, 200], 'frames': [{'index': 0, 'labels': [
        {'id': 'defender', 'group': 'defensive_priority', 'bbox': [40, 40, 80, 100]}]}]}


def test_export_scales_points_and_marks_unknown_role():
    monitor = PlayerMonitor()
    monitor.update(frame(0), [person()], 0, .25)
    record = monitor.records[0]
    assert record['footpoint'] == [60, 100]
    assert record['normalized_image'] == [.15, .5]
    assert record['group'] is None
    assert record['bbox_source_pixels'] == [40, 40, 80, 100]


def test_gap_is_not_connected_or_filled():
    monitor = PlayerMonitor()
    for index in (0, 1, 3):
        monitor.update(frame(index), [person()], 0, .25)
    assert [len(segment) for segment in monitor.trail_segments((0, 1), (200, 100))] == [2]
    monitor.update(frame(4), [], 0, .25)
    assert monitor.records == []


def test_scene_reset_clears_focus_and_paths():
    monitor = PlayerMonitor(reference())
    assert monitor.update(frame(0), [person()], 0, .25)
    assert monitor.update(frame(1), [person(scene_id=1)], 1, .25) == []
    assert monitor.groups == {} and monitor.history == {}


def test_role_does_not_jump_to_nearest_new_id():
    monitor = PlayerMonitor(reference())
    monitor.update(frame(0), [person()], 0, .25)
    assert monitor.records[0]['group_source'] == 'reviewed_reference'
    # Same position, new ByteTrack ID: no automatic role reassignment.
    assert monitor.update(frame(1), [person(track_id=2)], 0, .25) == []


def test_expired_identity_loses_manual_role():
    monitor = PlayerMonitor(reference(), max_idle_frames=2)
    monitor.update(frame(0), [person()], 0, .25)
    assert monitor.update(frame(3), [person()], 0, .25) == []
    assert monitor.groups == {}


def test_history_is_bounded_and_untracked_boxes_are_visible_without_focus():
    monitor = PlayerMonitor(history_size=2)
    for index in range(4):
        monitor.update(frame(index), [person()], 0, .25)
    assert len(monitor.history[(0, 1)]) == 2
    assert monitor.update(frame(4), [person(track_id=None)], 0, .25)
    assert monitor.records == []
