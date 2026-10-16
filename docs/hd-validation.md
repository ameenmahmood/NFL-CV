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
