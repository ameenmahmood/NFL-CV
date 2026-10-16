# Changelog

Each meaningful change lists its implementation commit. Follow-up commits that synchronize these hashes are available through `git log -- CHANGELOG.md`; they do not reference their own hashes.

## 2026-10-16 — Establish versioned baselines

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
