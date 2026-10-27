# Set 2 validation — October 27, 2026

## Inputs and pairing

- `data/Unlabeled set 2.mp4`: 1920×1080, 60 FPS, 446 frames, 7.43 seconds. All 446 frames decoded and were processed by the HD detector/tracker.
- `data/Labeled set 2.mp4`: 1918×1078, 30 FPS, 259 reported frames, 8.63 seconds. Six reference snapshots decoded and inspected.
- Color rectangles provide role references consistent with the five-group scheme. The clean and labeled clips differ in frame rate, duration, and framing; rectangles were not transferred by copying coordinates.
- Candidate labeled/clean indices: `0/48`, `52/139`, `103/238`, `155/337`, `206/445`, `258/445`. They are approximate image-similarity candidates, not exact synchronization. The final candidates reach the clean clip's endpoint; do not extrapolate an offset or accept them automatically.

## Reviewed priority-defender sample

`annotations/set_2_priority_sample.json` contains 14 assistant-reviewed visible-body boxes: seven blue-group defenders in each of clean frames 48 and 238. The user's colored reference identifies the groups; boxes were manually placed on clean images, independently of predictions. Near-line overlap makes some bounds less certain. This sample has not been independently audited. IDs are frame-local labels, not tracking ground truth.

At person confidence >= .25 and box IoU >= .50:

| Preset | Matches | Misses | Annotated-box recall |
|---|---:|---:|---:|
| Nano baseline: width 960 / inference 640 / NMS .70 | 8/14 | 6 | 57.1% |
| Nano HD: width 1920 / inference 1280 / NMS .45 | 10/14 | 4 | 71.4% |

HD misses: frame 48 `d_upper_near`, `d_middle_upper`, `d_middle_lower`; frame 238 `d_middle_upper`. These are score/overlap-threshold misses; some may have nearby or merged predictions rather than complete absence. They emphasize adjacent and occluded defenders as a remaining weakness. This is a tiny, correlated two-frame diagnostic, not general accuracy or role classification. Because only priority defenders are annotated, precision/false positives remain unavailable, and other groups are unscored.

The complete annotated video reopened at 1920×1080, 60 FPS, 446 frames; its middle frame decoded successfully. No identity-switch metric is claimed without track annotations. No formation classifier was added.

## Reproduce

Use new output directories for each run:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_frame_pairs.py "data\Labeled set 2.mp4" "data\Unlabeled set 2.mp4" --output outputs\set_2_pairs_new
.\.venv\Scripts\python.exe scripts\evaluate_detection.py annotations\set_2_priority_sample.json --video "data\Unlabeled set 2.mp4" --config configs\default.yaml --output outputs\set_2_baseline_new
.\.venv\Scripts\python.exe scripts\evaluate_detection.py annotations\set_2_priority_sample.json --video "data\Unlabeled set 2.mp4" --config configs\all22_hd.yaml --output outputs\set_2_hd_new
```

Ignored local artifacts: `outputs/set_2_candidates/` (source/reference snapshots and `clean_annotated.mp4`), `outputs/set_2_eval_baseline/`, and `outputs/set_2_eval_hd/` (JSON matches and reviewed-box images). Source SHA-256 in the annotation file prevents evaluation against a different encode.

Verification: 49 tests passed, 1 optional integration test skipped; both real-video evaluation runs completed separately. Both committed annotation samples pass schema checks. Source videos, weights, and outputs remain excluded from Git.
