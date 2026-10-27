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

## Measured movement export and reviewed-group focus

The application supports `--trails`, `--tracks-output FILE.jsonl`, and `--focus-annotations REVIEWED.json`. Existing defaults remain unchanged.

```powershell
.\.venv\Scripts\python.exe -m football_cv --config configs\all22_hd.yaml --input "data\Unlabeled set 2.mp4" --focus-annotations annotations\set_2_priority_sample.json --trails --tracks-output outputs\defender_tracks_new.jsonl --output outputs\defender_tracks_new.mp4 --no-display
```

With focus enabled, the video checksum and dimensions must match the annotation source. Only `defensive_priority`, `offensive_skill`, and `quarterback` reference groups are eligible; line groups are excluded from the focused display/export. Roles bind to matched tracking IDs at exact reviewed anchor frames, and the next anchor replaces the prior bindings. Before the first anchor or after a cut with no new reference, no focused people are invented. `[ref]` in the overlay means a manual reference group, not model recognition. All people still feed ByteTrack internally so filtering does not remove association context.

Bindings follow tracker IDs until they expire, are replaced at an anchor, or a scene reset clears them. They do not jump automatically to a nearest replacement ID. An undetected ByteTrack ID swap can still attach the group to the wrong person; role binding is therefore an inspection aid, not a validated role tracker. Existing priority-only samples cannot seed offensive roles; TE/WR/RB need more specific reviewed labels before individual position recognition can be evaluated.

JSONL begins with source/FPS metadata and then writes every processed frame, including frames with no observations. Each observed track records scene/track ID, timestamp, confidence, source-pixel box/footpoint, normalized-image point, and manual group provenance when available. Missing people generate no synthetic positions. The file is created exclusively and closed on normal exit, interruption, or inference errors. Without focus, measured tracked people have unknown groups; untracked detections remain visible but have no movement record.

Trails contain at most 60 measured observations per identity and break at frame gaps. Inactive history/bindings expire using the source-FPS-scaled ByteTrack buffer; scene changes clear them. No interpolation, pixel-to-yard conversion, field velocity, route, coverage, or formation inference is performed. Camera movement affects these paths, and even normalized image coordinates are not field coordinates.

### Movement verification

The complete set 2 clean clip ran with nano HD, trails, focus references, and JSONL export. Output reopened at 1920×1080/60 FPS with 446 frames; frame 300 was visually inspected. The stream contained one metadata line, 446 frame records, and 1,902 measured observations (not distinct people). It had zero focused observations before frame 48, four at frame 48, and six at the new reference at frame 238. All exported groups were `defensive_priority`, as expected from this reference file. Artifacts: `outputs/priority_movement_set2.mp4`, `outputs/priority_movement_set2.jsonl`, and `outputs/priority_movement_set2_preview.jpg`.

Final automated checks: 61 tests passed, 1 optional integration test skipped. Tests cover source-coordinate exports, unknown roles, role provenance, missing-frame gaps, scene resets, identity expiry, bounded history, refusal to assign a nearby replacement ID, application stream closure, and overwrite protection. This verifies implementation behavior; it does not establish continuous identity correctness on real footage. Next validation needs dense correspondence labels around the overlapping defenders and camera changes. Individual offensive-role evaluation will need TE/WR/RB labels beyond the current combined red-O group.

The subsequent [overlapping-crop experiment](tiling-experiment.md) did not recover additional priority targets. That document records the stopping point and the prepared clean clip for the next identity/role annotation request. More raw footage is not currently the bottleneck.
