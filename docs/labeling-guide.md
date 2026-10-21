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

1. Identify the original unmarked video and matching source frames/time range. This remains unconfirmed for set 1. The rendered file has different dimensions from the available 1920×1080 videos, so spatial and temporal correspondence must be established before copying coordinates or frame indices.
2. Use the annotated video to assign role groups, and draw each visible player's box on the corresponding clean frame. Store boxes as `(x1, y1, x2, y2)` in that clean frame's pixel coordinates with its source frame index and dimensions.
3. Mark ambiguous or obscured roles as unknown/ignore. Record whether each frame is exhaustively labeled; an unannotated person is not automatically a negative example. Keep officials and sideline personnel separate from the five target groups.
4. Manually review transferred roles/boxes, especially moving or overlapping players. Preserve original footage and labels independently.
5. Evaluate person detection separately for `defensive_priority` and the supporting groups. The current model predicts people, not these role groups; role-aware evaluation depends on reviewed annotations. Report misses and false detections without claiming role-classification accuracy.

Do not train on the colored letters themselves. Keep nearby frames from the same play together when splitting future training and evaluation sets, so almost identical frames cannot leak between splits. This short clip is useful for initial review but cannot establish generalization across games, teams, or camera angles.
