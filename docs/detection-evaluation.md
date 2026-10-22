# Reviewed player detection evaluation

The evaluator measures **person-box localization** on reviewed annotations, separately reporting the user's role groups. A matched `defensive_priority` box means the model located a labeled corner, safety, or linebacker; it does not mean the model classified their football position.

## Initial priority sample

`annotations/set_1_priority_sample.json` contains 14 manually placed visible-body boxes: seven priority defenders each in clean frames 0 and 136 of `Unlabeled set 1.mp4`. The assistant reviewed these boxes on the clean images, using the user's blue-D markings as a role reference. Boxes were not generated from detector outputs or copied from colored letter extents. The sample has not been independently audited. These two correlated frames from one play are a bootstrap diagnostic, not a held-out accuracy benchmark.

Only priority defenders are annotated. Each frame is explicitly `exhaustive: false`; unannotated linemen and offensive players are unscored, not false detections. Precision, false-positive totals, and unrepresented-group recall are unavailable (`null`). No personnel counts or formations are inferred.

### October 21, 2026 result

At person confidence >= 0.25 and box IoU >= 0.50:

| Preset | Annotated boxes matched | Missed | Annotated-box recall |
|---|---:|---:|---:|
| Baseline nano, width 960 / inference 640 / NMS .70 | 13/14 | 1 | 92.9% |
| HD nano, width 1920 / inference 1280 / NMS .45 | 14/14 | 0 | 100% |

The baseline missed the annotated near-line blue-D defender in frame 136 at the required overlap/score thresholds. HD matched that box. This supports retaining the HD preset for this sample; **100% on 14 correlated boxes does not establish general detection accuracy**. The sample was selected after viewing footage and the presets were already tuned on related footage. Do not use it as an independent test set or train on it and subsequently report it as held-out performance.

## Reproduce

Run from the repository root, using a new output directory each time:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_detection.py annotations\set_1_priority_sample.json --video "data\Unlabeled set 1.mp4" --config configs\default.yaml --output outputs\priority_baseline_new
.\.venv\Scripts\python.exe scripts\evaluate_detection.py annotations\set_1_priority_sample.json --video "data\Unlabeled set 1.mp4" --config configs\all22_hd.yaml --output outputs\priority_hd_new
```

The source SHA-256 binds labels to the exact clean video. Dimensions must also match. Source pixels are used for all boxes and IoU; processed detections are scaled back separately on each axis. Outputs include JSON per-frame matches/misses, aggregate/group counts, and clean-frame review images. Green/red rectangles in those images indicate matched/missed reference boxes, not predicted roles. First-inference timing includes model warmup and is not a throughput benchmark. Videos, weights, and outputs remain outside Git; reviewed annotation coordinates and provenance are tracked.

## Annotation contract and matching

- Version 1 JSON contains `source_size: [width, height]`, optional `video_sha256`, and a nonempty `frames` array. Each frame has a unique nonnegative `index`, `review_status: "reviewed"`, boolean `exhaustive`, and `labels`.
- Each label has a frame-unique `id`, a supported `group`, and finite positive-area `bbox: [x1, y1, x2, y2]` within source dimensions. Use visible-body boxes; do not infer hidden anatomy. Draft frames are rejected rather than silently scored.
- Supported groups are the five user groups in `docs/labeling-guide.md`, plus `other_person` for officials/sideline people when exhaustively labeling a frame. Ambiguous targets should keep a frame in draft until resolved, or be excluded with `exhaustive: false` and the selection documented. There is no ignore-region support yet.
- Predictions are filtered to person and the configured display-confidence threshold. Score-ordered greedy matching selects the highest-IoU unmatched reference box at or above the matching threshold; every prediction and reference box can match at most once. Duplicates cannot inflate recall.
- Unmatched references count as misses. On exhaustive frames, unmatched predictions count as false positives. If any scored frame is partial, aggregate precision/false positives are withheld. Aggregate recall sums counts before division; groups without reference boxes return `null`.

Tests cover overlap geometry, duplicate suppression in scoring, partial-label handling, role-group counts, thresholds, aggregate weighting, malformed/draft annotations, coordinate scaling, and the committed sample schema. Detector/tracker accuracy is separate from test pass counts.

Next expand to independently reviewed frames from distinct plays, including deep safeties, distant corners, motion, overlap, and camera changes. Fully annotate a subset of all visible people to measure precision, then evaluate identity switches on consecutive clips. No new footage is required to expand the current bootstrap sample, but additional plays will be needed for generalization claims.
