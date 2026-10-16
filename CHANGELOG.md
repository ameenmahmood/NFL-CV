# Changelog

Each meaningful change lists its implementation commit. Follow-up commits that synchronize these hashes are available through `git log -- CHANGELOG.md`; they do not reference their own hashes.

## 2026-10-16 — Establish versioned baselines

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
