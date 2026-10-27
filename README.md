# Real-Time Football Formation Recognition

Milestone 1 implements local video processing, general-purpose person detection with pretrained YOLO, and ByteTrack tracking. It does not identify teams, officials, field coordinates, pre-snap states, or formations. Detection confidence is the model's person score, not formation confidence.

## Windows PowerShell setup

Install Python 3.11 (64-bit) with the Python launcher, then run from this repository:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Activation is optional; use `.\.venv\Scripts\python.exe` in place of `python` if PowerShell blocks the activation script. Python 3.11 is the target; dependency versions are bounded in `pyproject.toml`. Installation may download large PyTorch packages. For NVIDIA acceleration install a matching CUDA-enabled PyTorch build using the [official installer](https://pytorch.org/get-started/locally/) in this environment. `device: auto` selects CUDA device 0 only when PyTorch reports it available, otherwise CPU. An explicitly requested unavailable GPU produces an error.

## Provide a video and run

Use a legally obtained MP4 broadcast or All-22 clip, ideally a short pre-snap wide-angle sequence. Put it in `data/test.mp4` or specify another path. Videos and weights are ignored by Git.

```powershell
New-Item -ItemType Directory -Force data
python -m football_cv --input "C:\Videos\football.mp4"
python -m football_cv --input "C:\Videos\football.mp4" --device cpu
python -m football_cv --input "C:\Videos\football.mp4" --output outputs\annotated.mp4
python -m football_cv --input "C:\Videos\football.mp4" --output outputs\batch.mp4 --no-display
python -m football_cv --config configs\default.yaml
```

For HD All-22 footage, use the validated higher-resolution preset:

```powershell
python -m football_cv --config configs\all22_hd.yaml --input "data\all22_test_HD.mp4"
python -m football_cv --config configs\all22_hd.yaml --input "data\all22_test_HD.mp4" --output outputs\hd_annotated.mp4 --no-display
```

This preserves up to 1920-pixel frame width, runs YOLO11 nano at inference size 1280, and uses NMS IoU 0.45 to suppress more duplicate boxes. It improves distant-person detection in inspected HD samples, with a substantial CPU speed cost. It still detects officials and sideline personnel and can miss or merge overlapping players. Lower `nms_iou` suppresses more overlap but can remove distinct neighboring players; the baseline retains 0.70. See [HD validation findings](docs/hd-validation.md) for the comparison rather than interpreting detection counts as player accuracy.

An optional `configs/all22_hd_small.yaml` preset uses the larger YOLO11 small model. It recovered one extra reviewed priority-defender box on the small set 2 sample; it is experimental and slower on CPU, not the default. See [set 2 evaluation](docs/set-2-validation.md) for the measured results and limitations. Always pass `--input` with this preset.

First execution of `yolo11n.pt` may download pretrained weights and requires internet. For offline use configure `model` with the path to a trusted local COCO detection `.pt` file. Output must be a new MP4 path; existing files are refused. A partial output is finalized if interrupted or decoding fails. MP4 output contains video only, no original audio, at the source nominal FPS. No frames are intentionally skipped; pause does not duplicate saved frames. Variable-frame-rate input is written at constant nominal FPS.

Space or **P** pauses/resumes; **Q** or Escape quits; **R** resets tracking identities manually after an unrelated scene. Closing the window quits. Headless mode runs until EOF or Ctrl+C. Display runs at processing speed rather than pacing to source FPS. The rolling processing FPS measures decode, detection, tracking, and scene analysis, excluding annotation, output encoding, display waits, and pauses.

## Configuration

`configs/default.yaml` configures input/output, saving, window display, model, display confidence threshold, inference device (`auto`, `cpu`, `0`, `cuda:0`), maximum processing width, inference image size, tracking thresholds/buffer, and cut detection. Paths resolve from the current working directory. Aspect ratio is preserved; boxes and footpoints use processed-frame pixels. Frame records retain source dimensions, zero-based index, and decoder timestamp in seconds. When decoder timestamps are missing or non-monotonic, nominal FPS provides a monotonic fallback.

The nano model and 640 inference size are conservative defaults for CPU operation. Reduce `processing_width` and `imgsz` for speed; increase them for distant players if hardware permits. No real-time speed is guaranteed. ByteTrack's low-score recovery uses detections down to `track_low_thresh`; only detections meeting `confidence` appear in the overlay. The tracking adapter scales `track_buffer` by source FPS relative to 30 FPS. IDs appear as `scene:track`; a reset changes the scene number even if ByteTrack reuses a numeric ID. Missing tracks are retained internally for recovery, but absent people are not drawn as measured detections. Ultralytics is pinned to the verified API version; upgrade it with the tracker tests.

## Structure

```text
configs/default.yaml
src/football_cv/
  app.py             orchestration and CLI
  config.py          validated YAML configuration
  types.py           frame and detection records, pixel footpoints
  video.py           capture, timing, resizing, output
  detection.py       YOLO adapter and device selection
  tracking.py        ByteTrack adapter and scene identity
  scenes.py          configurable hard-cut heuristic
  visualization.py   boxes, confidence, IDs, timestamps, FPS
  performance.py     rolling throughput
tests/               synthetic, mocked, tracker, optional integration tests
```

No empty classifier modules are added. Future team identification, field calibration, coordinate mapping, and formation classifiers can consume these records after validation. Field coordinates will require explicit calibration and visibility information rather than treating image pixels as yards.

## Tests

Reviewed-box evaluation is now available separately from pipeline smoke tests. See [detection evaluation](docs/detection-evaluation.md) for the annotation format, commands, and initial 14-box defensive-priority comparison. It reports localization recall on labeled role groups, not automatic role classification. Partial annotation deliberately disables precision reporting.

```powershell
python -m pytest -q
$env:FOOTBALL_TEST_VIDEO = "C:\Videos\football.mp4"
$env:FOOTBALL_TEST_MODEL = "C:\Models\yolo11n.pt"
python -m pytest -q -m integration
```

Normal tests do not download weights or require a GPU. They test real OpenCV synthetic MP4 encoding/decoding, configuration errors, decode failures, scene changes, and actual ByteTrack recovery/reset using synthetic detections. The integration smoke test runs three frames with supplied local weights and video; it validates API operation, not football detection accuracy. Assess accuracy on labeled football clips separately.

Development verification: 22 tests passed and the optional integration test skipped because no video/weights were supplied. Tested locally on Windows with the available Python 3.12.14 runtime, PyTorch 2.14.1, OpenCV 4.14.0, and Ultralytics 8.4.177. Python 3.11 remains the installation target but was not available for this verification. GUI playback, pretrained inference on football footage, and GPU throughput remain unverified.

Subsequent validation used local football footage and pretrained weights, including the new HD file: 32 tests passed on October 16. GUI playback and GPU performance remain unverified. Detector boundary tests check that configured NMS is applied, non-person classes are excluded, and low-confidence candidates remain available to ByteTrack.

For reproducible real-video checks (use a new output directory):

```powershell
python scripts\validate_video.py data\all22_test_HD.mp4 --config configs\all22_hd.yaml --starts 543 2174 7611 --segment-frames 90 --output outputs\hd_validation_new
```

This decodes the entire file, then runs consecutive tracking segments using the selected preset, saving annotated clips, snapshots, and a JSON report. It tests API and codec behavior, not labeled detection accuracy. `scripts/benchmark_detection.py` compares model/resolution/NMS combinations on identical frames.

## Limitations and roadmap

The COCO person model also detects officials, sideline personnel, and spectators. Small or occluded players can be missed, and tracks can swap during overlaps or camera movement. The cut detector combines histogram distance and pixel change: it is a heuristic with false positives and missed cuts, especially dissolves or visually similar angles. Detected cuts reset tracking before processing the new frame; manual resets cover missed transitions. This is not camera-motion compensation or guaranteed shot-boundary detection. Corruption at a reported early EOF produces an error; OpenCV cannot always distinguish unknown-length EOF from a decode error.

Next validate detections and tracking on labeled broadcast and All-22 clips, including cuts and occlusions; collect precision/recall, identity-switch rates, and runtime measurements. Then implement team/official identification and robust scene segmentation, field landmarks and calibrated coordinates, and pre-snap/visibility gating. Only after those are validated add offensive geometry and defensive fronts with `Unknown`/`Insufficient Visibility` results and calibrated formation confidence. Do not infer personnel groups from visible counts. Live input support is a later milestone. Coverage, blitz, and play prediction are outside scope.

APIs follow [Ultralytics prediction](https://docs.ultralytics.com/modes/predict/) and [ByteTrack reference](https://docs.ultralytics.com/reference/trackers/byte_tracker/). Review [Ultralytics licensing](https://www.ultralytics.com/license) before distributing a product.
