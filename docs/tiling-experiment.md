# Overlapping-crop detection experiment — October 27, 2026

## Method

`scripts/benchmark_tiling.py` compares the nano HD full-frame pass with the union of that pass and four overlapping crops. Each crop spans 60% of source width/height; adjacent crops overlap by 20% of each frame dimension. All passes use YOLO11 nano, inference size 1280, confidence floor .1, and NMS IoU .45. Crop boxes touching internal boundaries are rejected to avoid treating partial bodies as complete observations. Global score-ordered IoU suppression merges duplicate boxes in source pixels.

Evaluation uses the same four reviewed clean frames from sets 1 and 2, with confidence .25 and matching IoU .50. No labels, thresholds, or boxes were modified after observing these results. Reference annotations are partial and cannot measure precision or false-positive rates.

## Findings

| Sample | Full-frame matches | Full + crops matches |
|---|---:|---:|
| Set 1, 14 priority-defender boxes | 14/14 | 14/14 |
| Set 2, 14 priority-defender boxes | 10/14 | 10/14 |

The same set 2 targets remain missed: frame 48 `d_upper_near`, `d_middle_upper`, `d_middle_lower`; frame 238 `d_middle_upper`. It is not merely an unchanged total with different players recovered. More displayed boxes appeared in some frames (for example, set 1 frame 136: 19 to 21), but no extra annotated priority defender was matched. Those additional boxes cannot be labeled false positives from partial annotations.

This configuration performs five model passes per frame without improving the measured target recall, so it was **not integrated into the normal detector or tracking pipeline**. It remains a reproducible diagnostic. This does not prove all tiling approaches fail; it only rejects this particular setting on these small samples. First-inference timing includes startup and is not used for a speed claim. Since target recovery did not improve, another expensive tracking run with this detector was not justified.

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_tiling.py --annotations annotations\set_1_priority_sample.json annotations\set_2_priority_sample.json --videos "data\Unlabeled set 1.mp4" "data\Unlabeled set 2.mp4" --output outputs\tiling_comparison_new
```

Use a new output directory. Local outputs in `outputs/tiling_comparison/` contain per-frame matches, missed IDs, scores/boxes for matched targets, counts, timings, and reviewed-box images. Videos, weights, and generated artifacts remain ignored.

Verification: 63 tests passed, 1 optional integration test skipped. Added synthetic checks for complete crop coverage, valid boundaries, duplicate suppression, and preservation of neighboring boxes. Both datasets were bound to clean-video SHA-256 before inference.

## What is now needed

The existing two-frame reference samples cannot establish continuous identity correctness, especially through overlapping players, and the group labels do not identify individual offensive positions. More tuning on these same few boxes would not justify confidence in the tracker.

The missing input is **reviewed identity/role annotation**, not more raw video. A ready-to-label clean clip has been extracted from the footage already supplied:

- File: `outputs/tracking_annotation_request/clean_tracking_70_to_80s.mp4`.
- Source: `data/all22_test_HD.mp4`, 70–80 seconds, source frames 2100–2399.
- Clip: 1920×1080, 30 FPS, 300 frames. Clip frame 0 corresponds to source frame 2100. This includes pre-snap alignment, the snap, player movement, and occlusion from one play; a different scene begins later and is not included.

Please return a labeled version of this clip with consistent player identities throughout: for example `CB1`, `CB2`, `S1`, `LB1`, `LB2`, and `TE1`, `RB1`, `WR1`, `WR2`, `QB1` for the visible offensive skill players. Use positions only when known; a stable `D1` or `O1` plus an unknown role is better than guessing. Keep the same ID when a player overlaps others or reappears; explicitly mark uncertain identity and out-of-view intervals. Label visible players only, preserving the clean file unchanged. Corners/safeties/linebackers remain the priority.

If the annotation tool supports export, prefer a JSON/CSV containing clip frame index, player ID, role, and bounding box in clip pixels. Identity boxes at least every 0.2–0.5 seconds, plus immediately before/after occlusions, would be substantially more useful than two isolated endpoints. A video with clearly attached persistent labels is also usable as a manual reference. Keep its original frame rate/dimensions and avoid extra playback loops or cuts where possible.

Until that reference is available, existing code/tests and sparse metrics are preserved, but no claim that tracking, individual roles, field movement, or formation/coverage recognition is solved is warranted.
