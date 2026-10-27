# Changelog

Each meaningful change lists its implementation commit. Follow-up commits that synchronize these hashes are available through `git log -- CHANGELOG.md`; they do not reference their own hashes.

## 2026-10-27

### Set 2 priority-defender validation — hash pending

- Reviewed the new color-box reference and 1080p/60 FPS clean video, prepared candidate frame pairs, and manually annotated 14 priority-defender boxes on clean frames.
- Files: `annotations/set_2_priority_sample.json`, `tests/test_evaluation.py`, `docs/set-2-validation.md`, `docs/labeling-guide.md`, `docs/detection-evaluation.md`, `CHANGELOG.md`.
- Verification: 49 tests passed, 1 optional integration test skipped. Baseline matched 8/14 boxes and HD 10/14 at confidence .25 / IoU .50. Full HD tracking processed all 446 clean frames; output reopened at original size/rate. Small sample, partial labels, no role-classification or general accuracy claim.

## 2026-10-21

### Reviewed defensive-priority detection evaluation — `5917992`

- Added validated clean-frame annotation loading, source-coordinate scaling, one-to-one IoU matching, per-group recall, and precision safeguards for partially labeled frames.
- Manually reviewed 14 priority-defender boxes across two clean frames, preserving source checksum and annotation provenance; added a runnable evaluation CLI and documented limitations.
- Files: `src/football_cv/evaluation.py`, `scripts/evaluate_detection.py`, `tests/test_evaluation.py`, `annotations/set_1_priority_sample.json`, `docs/detection-evaluation.md`, `README.md`, `CHANGELOG.md`.
- Verification: 48 tests passed, 1 optional integration test skipped; both real-video evaluation commands reproduced baseline 13/14 and HD 14/14 reviewed-box matches at confidence .25 / IoU .50. Script compilation and whitespace checks passed. This is a two-frame bootstrap diagnostic, not general accuracy or role recognition.

### Clean source review and candidate frame preparation — `547fe57`

- Confirmed the user-supplied clean counterpart, decoded all 272 frames, and documented differing lengths and approximate frame correspondence.
- Added a commented diagnostic script exporting clean/labeled reference images and candidate match metadata, explicitly requiring review and never transferring annotation boxes automatically.
- Files: `scripts/prepare_frame_pairs.py`, `tests/test_frame_pairs.py`, `docs/labeling-guide.md`, `CHANGELOG.md`.
- Verification: six reference pairs exported and representative pairs visually inspected; 33 tests passed, 1 optional integration test skipped. The HD detector/tracker processed all 272 clean frames, and its 1080p annotated output reopened and a middle frame decoded successfully.

### Labeled player groups and first reference review — `a690e5f`

- Recorded blue D, green L, purple L, red O, and yellow Q role groups and the priority on corners, safeties, and linebackers.
- Reviewed the provided labeled video, documented burned-in label limitations, and outlined transfer to matching clean source frames with manually reviewed boxes.
- Files: `docs/labeling-guide.md`, `AGENTS.md`, `CHANGELOG.md`.
- Verification: all 324 frames decoded; six snapshots visually inspected. No application logic changed and no detector accuracy/training claims were made. Clean source/time correspondence is awaiting confirmation.

## 2026-10-16 — Establish versioned baselines

### Relevant code comments and persistent documentation standard — `8d357ab`

- Added comments/docstrings throughout source modules, validation scripts, and tests explaining data units, external API layouts, timing, tracking, cleanup, benchmark assumptions, and fixture intent.
- Recorded the requirement to maintain relevant comments for all future code in `AGENTS.md`.
- Files: `src/football_cv/*.py`, `scripts/*.py`, `tests/*.py`, `AGENTS.md`, `CHANGELOG.md`.
- Verification: executable ASTs matched HEAD across all 19 edited Python files after removing docstrings; 32 tests passed, including the local HD-video integration test. Git whitespace checks passed. Runtime logic is unchanged.

### HD detector benchmark — `2600d24`

- Added a reproducible six-frame comparison of nano/small YOLO models, two input resolutions, and NMS thresholds on the supplied 1080p video.
- Recorded visual findings and inference timings, explicitly distinguishing person counts from measured player accuracy.
- Files: `scripts/benchmark_detection.py`, `docs/hd-validation.md`.
- Verification: executed all 24 preset/frame combinations with local pretrained models; benchmark images and weights remain ignored.

### HD detection preset and configurable suppression — `8c0b746`

- Added a CPU-compatible HD preset retaining 1080p frames with 1280-pixel inference and stricter configurable NMS to improve distant-person detections and reduce duplicate boxes.
- Kept baseline resolution/suppression unchanged; preserved low-score candidates for ByteTrack.
- Extended video validation to use YAML presets and arbitrary consecutive segments with output metadata and full-decode checks.
- Files: `src/football_cv/config.py`, `src/football_cv/detection.py`, `configs/default.yaml`, `configs/all22_hd.yaml`, `scripts/validate_video.py`, `tests/test_detection.py`, `README.md`.
- Verification: 32 tests passed including the local HD-video integration test. Real-video segment results are documented in `docs/hd-validation.md`.

### Automatic GitHub synchronization — `dc22754`

- Replaced the earlier opt-in push policy with standing authorization to push completed changes to `development` or task-specific feature branches after relevant verification.
- Prohibited direct pushes to `main`/production branches, unapproved PR merges, force-pushes, remote branch deletion, and destructive recovery after failed pushes.
- Files: `AGENTS.md`, `CHANGELOG.md`.
- Created local `development` branch. No remote is configured, and GitHub CLI is not currently available on PATH; synchronization awaits repository/authentication setup.
- Verification: reviewed documentation diff and Git status; no application behavior changed, so existing test results apply.

### Milestone 1 baseline — `07b28ef`

- Retrospectively committed the implementation developed October 13, preserving the existing files without resetting or discarding work.
- Added local MP4 capture/timing, YOLO person detection, ByteTrack identities, scene resets, visualization, output saving, device selection, validated configuration, and automated tests.
- Files: `.gitignore`, `pyproject.toml`, `README.md`, `configs/default.yaml`, `src/football_cv/`, `tests/`.
- Verification on October 16: 22 tests passed, 1 optional integration test skipped.

### All-22 validation baseline — `172415b`

- Retrospectively committed the October 14 real-video validation tools and findings, including full decode checks, annotated sample segments, and resolution comparisons.
- Files: `scripts/validate_video.py`, `scripts/compare_resolution.py`, `docs/all22-validation.md`.
- Verification: scripts compile; the documented October 14 run passed 23 tests with the supplied video and weights and decoded all 5,165 frames. No new real-video accuracy benchmark was run for this Git workflow change.

### Persistent Git workflow — `cbb27e5`

- Established mandatory status/diff review, selective staging, logical conventional commits, test verification, changelog maintenance, preservation of user changes, and end-of-task Git reporting.
- Explicitly prohibited automatic pushes, force-pushes, shared-history rewrites, and branch deletions without authorization.
- Files: `AGENTS.md`, `CHANGELOG.md`.
- Verification: documentation reviewed; model weights, outputs, and virtual environment confirmed ignored by Git.
