# All-22 baseline validation — October 14, 2026

Input: `all22_test.mp4` supplied locally by the user. Original file was read without modification.

- 1920×1080, 30 FPS, 5,165 frames, 172.17 seconds.
- All 5,165 frames successfully decoded with OpenCV.
- Full pytest suite: **23 passed**, including the previously optional local-video/local-weights integration test.
- Model: pretrained `yolo11n.pt`, Ultralytics 8.4.177, CPU PyTorch.
- Three consecutive 150-frame segments were processed at width 960 / inference size 640. Each segment starts with fresh tracking state. These are sample segments, not full-video inference.

| Start time | Processing FPS | Mean displayed detections | Mean tracked displayed detections |
|---|---:|---:|---:|
| 0.00 s | 10.10 | 1.23 | 0.77 |
| 86.07 s | 16.67 | 12.59 | 11.77 |
| 167.17 s | 17.00 | 14.49 | 12.95 |

Processing FPS includes detection and tracking, but excludes annotation/encoding. The first segment includes model warmup. No cut resets fired within these three segments; this does not establish that transitions elsewhere are detected correctly. Track persistence and identity correctness require labeled evaluation.

## Resolution comparison

Same frames and confidence threshold 0.25; general-purpose person detections, not player counts:

| Frame | Width 960 / inference 640 | Width 1920 / inference 1280 |
|---|---:|---:|
| 0 | 0 | 50 |
| 2657 | 10 | 25 |
| 5090 | 14 | 17 |

Visual inspection confirms the low-resolution baseline misses many distant players. Higher resolution recovers additional on-field people, but also detects sideline personnel and officials and sometimes produces duplicate boxes on the same person. Larger counts alone do not measure accuracy. Higher-resolution inference took approximately 0.13–0.15 seconds per sampled frame after startup; these sparse timings are not an end-to-end throughput benchmark. The first low-resolution comparison included about 5.9 seconds of startup overhead.

The opening frame appears to show a kickoff alignment, making this clip useful for small-person detection and non-scrimmage footage as well as later scrimmage examples. No formation predictions were generated.

## Artifacts and reproduction

Ignored local artifacts are in `outputs/all22_validation/`: three annotated MP4s, sampled JPEGs, a contact sheet, resolution comparisons, and JSON measurements.

```powershell
.\.venv\Scripts\python.exe scripts\validate_video.py "C:\path\all22_test.mp4" --output outputs\new_validation
.\.venv\Scripts\python.exe scripts\compare_resolution.py "C:\path\all22_test.mp4" --output outputs\new_validation
```

Use a new output directory for another run; annotated videos refuse overwriting. Comparison images are independent detections and intentionally have no track IDs. Default runtime configuration remains unchanged.

Next: label a small evaluation set of wide and close pre-snap frames; compare higher-resolution inference and detection models, inspect duplicate suppression, and measure on-field precision/recall before choosing defaults. Then evaluate identity switches across motion/occlusion and cuts. Field-region and team identification will be needed to exclude sideline personnel and officials. Do not use these counts to infer personnel groups or formations.
