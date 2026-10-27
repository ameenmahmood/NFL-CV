"""Sparse identity-anchor diagnostics, deliberately separate from full tracking metrics."""


def anchor_continuity(observations: dict[str, list[dict]]) -> dict:
    """Compare adjacent reviewed anchors without assuming what happened between them.

    A miss is unobservable, not evidence of identity preservation. Camera resets have
    separate namespaces and are excluded from within-scene comparisons.
    """
    preserved = changed = unobserved = resets = 0
    for anchors in observations.values():
        ordered = sorted(anchors, key=lambda a: a['frame'])
        for first, second in zip(ordered, ordered[1:]):
            a, b = first['track_key'], second['track_key']
            if a is None or b is None:
                unobserved += 1
            elif a[0] != b[0]:
                resets += 1
            elif a == b:
                preserved += 1
            else:
                changed += 1
    comparable = preserved + changed
    return {'preserved_pairs': preserved, 'changed_id_pairs': changed,
            'unobserved_pairs': unobserved, 'scene_reset_pairs': resets,
            'comparable_pairs': comparable,
            'anchor_preservation': preserved / comparable if comparable else None,
            'note': 'Sparse endpoint comparison only; intervening switches, MOTA, and IDF1 are not measured.'}
