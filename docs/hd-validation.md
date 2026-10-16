# HD All-22 validation — October 16, 2026

Input: local `data/all22_test_HD.mp4` (not committed). Metadata: 1920×1080, 30 FPS, 10,873 reported frames, 362.43 seconds. The recording includes end-zone, sideline, and post-play views.

## Identical-frame detector comparison

`scripts/benchmark_detection.py` compares six fixed fractional positions using person confidence 0.25 for display and 0.1 for detector candidates. Each preset warms up before timing. Runs use CPU; weights and generated images/JSON remain ignored.

| Frame | Nano 640 / IoU .70 | Nano 1280 / IoU .70 | Nano 1280 / IoU .45 | Small 1280 / IoU .45 |
|---|---:|---:|---:|---:|
| 543 | 15 | 31 | 27 | 25 |
| 2174 | 15 | 17 | 16 | 16 |
| 3805 | 19 | 40 | 38 | 34 |
| 5436 | 16 | 14 | 14 | 15 |
| 7611 | 18 | 18 | 18 | 17 |
| 9785 | 11 | 15 | 12 | 13 |

Columns are total **person detections**, not correct-player counts, recall, or personnel groups. The 640 preset uses frame width 960; 1280 presets retain frame width 1920. No labeled accuracy benchmark or trained football detector is claimed.

Visual inspection of frame 543 confirms higher-resolution nano inference recovers distant players missed by the baseline. Lowering NMS IoU from .70 to .45 reduces highly overlapping boxes, but does not eliminate merged players or duplicates. It can also suppress distinct overlapping players. Sideline personnel and officials remain included. The larger small model did not demonstrate a consistent visual advantage sufficient to justify its CPU cost on this sample, so it is not selected as the primary preset.

Warm inference timings across these six frames were approximately 0.030–0.040 seconds for nano 640, 0.105–0.118 seconds for nano 1280, and 0.249–0.272 seconds for small 1280. These are sparse inference-only timings, excluding video decode, tracking, display, and encoding; they do not establish real-time application speed.

Reproduce with a new output directory:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_detection.py data\all22_test_HD.mp4 --output outputs\hd_benchmark_new
```

The benchmark downloads missing trusted Ultralytics weights through its normal loader. It writes per-detection boxes/scores and annotated frames to the specified ignored output directory.

## Implemented HD preset and consecutive tracking check

`configs/all22_hd.yaml` selects nano, frame width 1920, inference size 1280, and configurable `nms_iou: 0.45`. The baseline still uses 960/640/.70. No team, official, or field filtering was added.

All 10,873 input frames decoded successfully. Using the HD preset, three separate 90-frame consecutive segments produced these results:

| Start frame (time) | Processing FPS | Mean displayed people | Mean displayed people with IDs |
|---|---:|---:|---:|
| 543 (18.10 s) | 5.71 | 28.31 | 26.42 |
| 2174 (72.47 s) | 8.03 | 14.57 | 13.96 |
| 7611 (253.70 s) | 8.00 | 18.56 | 18.04 |

Processing timing includes decode, detection, and tracking, excluding annotation/encoding. The first segment includes startup overhead. Tracking restarts at each segment seek; no cut resets fired within these short segments. This is not a full-video inference or tracking-accuracy evaluation. Encoded outputs reopened successfully at 1920×1080 with 90 frames each.

Inspected snapshots show IDs assigned to additional distant people. Occlusions still produce merged or missed players, sideline people remain, and a goalpost/flag structure was falsely detected in a close shot. Thus the preset improves useful detail but is not a reliable player-only detector. The next accuracy milestone should label on-field boxes across these views and measure precision/recall, duplicates, and identity switches; a football-specific fine-tuned detector and field-region filtering should be evaluated against those labels.

Validation artifacts: `outputs/hd_tracking/` contains three annotated clips, snapshots/contact sheet, and `report.json`. `outputs/hd_benchmark/` contains all 24 comparison images and detection records. Neither directory is committed.

```powershell
.\.venv\Scripts\python.exe scripts\validate_video.py data\all22_test_HD.mp4 --config configs\all22_hd.yaml --starts 543 2174 7611 --segment-frames 90 --output outputs\hd_tracking_new
.\.venv\Scripts\python.exe -m football_cv --config configs\all22_hd.yaml
```

Automated verification: **32 tests passed**, including real pretrained inference on the supplied HD file. New tests validate NMS settings, invalid settings, person-class filtering, empty results, and preservation of low-confidence tracking candidates. CLI help and script compilation also passed. GPU and interactive GUI behavior were not tested.
