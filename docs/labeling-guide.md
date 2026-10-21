# Player labeling scheme

User-provided role groups, confirmed October 21, 2026. These are annotation categories, not capabilities of the current COCO person detector.

| Visual label | Annotation group | Meaning |
|---|---|---|
| Blue D | `defensive_priority` | Cornerbacks, safeties, and linebackers; excludes separately labeled defensive linemen |
| Green L | `defensive_line` | Defensive linemen |
| Purple L | `offensive_line` | Offensive linemen |
| Red O | `offensive_skill` | Offensive skill players other than the separately labeled quarterback, including WR, RB, and TE |
| Yellow Q | `quarterback` | Quarterback; use this category instead of also labeling the player as red O |

Corners, safeties, and linebackers are the main detection/evaluation priority. Defensive linemen still supply supporting formation context. Blue D does not distinguish a corner from a safety or linebacker. Do not infer a specific position, personnel group, or formation from these group labels alone. Green and purple L must remain separate despite sharing the same letter.

## Labeled set 1 review

- Local file: `data/Labeled set 1.mp4`; keep this footage outside Git.
- Metadata: 1918×1066 pixels, 30 FPS, 324 frames, 10.8 seconds. All 324 frames decoded successfully.
- Six snapshots inspected at zero-based indices 0, 64, 129, 193, 258, and 323. They show visible color/letter annotations over players. Saved review snapshots/contact sheet are ignored artifacts under `outputs/labeled_set_1_review/`.
- The file contains embedded letters and playback controls; it is a visual role reference, not a machine-readable box dataset. Letters vary in size and placement; their extents must not be treated as player bounding boxes.
- No detector was trained or scored on this annotated video. Model predictions on marked pixels would not provide a clean evaluation of the underlying football footage.

## Converting this reference into an evaluation set

1. Identify the original unmarked video and matching source frames/time range. For set 1 the user supplied `data/Unlabeled set 1.mp4`: 1920×1080, 30 FPS, 272 frames (9.07 seconds), all decoded successfully. The labeled recording has 324 frames and different dimensions and appears to repeat the opening near its end; do not pair by equal frame indices or assume one constant offset.
2. Use the annotated video to assign role groups, and draw each visible player's box on the corresponding clean frame. Store boxes as `(x1, y1, x2, y2)` in that clean frame's pixel coordinates with its source frame index and dimensions.
3. Mark ambiguous or obscured roles as unknown/ignore. Record whether each frame is exhaustively labeled; an unannotated person is not automatically a negative example. Keep officials and sideline personnel separate from the five target groups.
4. Manually review transferred roles/boxes, especially moving or overlapping players. Preserve original footage and labels independently.
5. Evaluate person detection separately for `defensive_priority` and the supporting groups. The current model predicts people, not these role groups; role-aware evaluation depends on reviewed annotations. Report misses and false detections without claiming role-classification accuracy.

Do not train on the colored letters themselves. Keep nearby frames from the same play together when splitting future training and evaluation sets, so almost identical frames cannot leak between splits. This short clip is useful for initial review but cannot establish generalization across games, teams, or camera angles.

## Candidate clean-frame preparation

`scripts/prepare_frame_pairs.py` decodes the clean clip and compares six labeled snapshots to grayscale thumbnails from every clean frame. It exports original PNGs, side-by-side review images, and a JSON manifest. Every match is marked `requires_manual_review: true`: the similarity score is an image difference, not a calibrated probability or proof of temporal correspondence. Static field markings can dominate it, and nearby frames differ in player positions. No role or box is transferred automatically.

```powershell
.\.venv\Scripts\python.exe scripts\prepare_frame_pairs.py "data\Labeled set 1.mp4" "data\Unlabeled set 1.mp4" --output outputs\set_1_pairs_new
```

The initial run suggested labeled/clean indices `0/89`, `65/108`, `129/167`, `194/208`, `258/271`, and `323/91`. Visual review confirms the shared play and that these are approximate, not exact player-pose matches. In particular, the final labeled snapshot returns to an earlier part of the play. These candidates are useful role references; manual box placement must use each selected clean image itself. Initial artifacts are under ignored `outputs/set_1_candidates/`.

The HD person detector and ByteTrack pipeline also processed all 272 clean frames headlessly. The annotated MP4 reopened successfully at 1920×1080, 30 FPS, 272 frames; a middle frame was decoded and inspected. Output: `outputs/set_1_candidates/clean_annotated.mp4`. This is a working detection/tracking overlay, not role predictions or a ground-truth accuracy score. Automated checks: 33 tests passed, 1 optional integration test skipped; the separate real-video CLI run supplies the full-clip smoke check.
