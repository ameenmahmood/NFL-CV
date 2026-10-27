# Priority-player tracking and movement

Defensive priority is corners, safeties, and linebackers. Offensive priority is TE alignment, RB, WR, and QB. Detailed line recognition is supporting context. The current detector identifies people, not these individual football positions. The supplied role groups do not distinguish TE from WR/RB or individual defensive positions; those require additional reviewed role annotations or a validated team/role model.

Formation/coverage interpretation is not implemented. Player movement must eventually use calibrated field coordinates: camera pan/zoom affects image-space paths, and image-left is not automatically the offense's strong or weak side. TE alignment and eligible-receiver context must be validated before side labels are inferred.

## Sparse tracking audit

`annotations/set_2_track_links.json` explicitly links seven priority defenders at clean frames 48 and 238. The assistant inspected the endpoints and intermediate frame 143 for this short pre-snap sequence. These correspondences are separate from the original detection labels, whose IDs were frame-local. The links have not been independently audited.

`scripts/audit_tracking.py` processes every source frame from 0 through 238. It applies the normal detector, ByteTrack, and cut reset, then matches current tracked observations to the reviewed boxes at confidence .25 and IoU .50. It compares scene/track keys for each linked person:

- Preserved pair: both anchors observed with the same key.
- Changed ID pair: both observed in the same scene with different IDs.
- Unobserved pair: at least one anchor lacks a matched tracking ID.
- Scene reset pair: matched anchors belong to different scenes and are excluded from within-scene continuity.

This checks sparse endpoints, **not full tracking accuracy**. An identity could swap and swap back between anchors. The script does not compute MOTA, IDF1, per-frame identity switches, or movement accuracy. Missing players are not silently counted as stable tracks. More dense identity labels and motion/camera-change clips are needed before declaring tracking reliable.

```powershell
.\.venv\Scripts\python.exe scripts\audit_tracking.py annotations\set_2_track_links.json --video "data\Unlabeled set 2.mp4" --config configs\all22_hd.yaml --output outputs\tracking_nano_new
.\.venv\Scripts\python.exe scripts\audit_tracking.py annotations\set_2_track_links.json --video "data\Unlabeled set 2.mp4" --config configs\all22_hd_small.yaml --output outputs\tracking_small_new
```

Output directories must be new. Each report retains per-person observations and distinguishes unobserved endpoints from changed IDs. Video SHA-256 and frame dimensions must match the reviewed source. Detector evaluation matches now also retain the scene and tracking ID when available; ordinary detection-only evaluation has null IDs.

### October 27 sparse-anchor results

The nano HD audit processed all 239 frames through the final anchor with no detected cuts. Four of seven defender pairs were observed at both anchors and retained the same ID; three pairs were unobservable because at least one endpoint lacked a qualifying tracked box. There were no changed IDs among the four comparable pairs. Reporting this as perfect tracking would hide the three missing targets; reliable detection remains a prerequisite for continuity. Local output: `outputs/tracking_audit_nano/tracking_audit.json`.

The small HD audit also processed 239 frames with four preserved pairs, zero changed pairs, and three unobserved pairs; no cuts fired. Despite an extra detection match in the earlier two-frame detection experiment, this larger detector did not improve the number of comparable identity pairs. Local output: `outputs/tracking_audit_small/tracking_audit.json`. Both results describe only the reviewed endpoints, not every intervening identity assignment. Automated checks: 51 tests passed, 1 optional integration test skipped.
