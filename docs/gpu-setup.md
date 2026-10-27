# NVIDIA GPU setup and reproducible speed checks

The project supports CPU execution and uses CUDA automatically only when the installed PyTorch runtime reports it available. An NVIDIA GPU can be visible to `nvidia-smi` while PyTorch is CPU-only. On this machine, the initial environment was PyTorch `2.14.1+cpu`, Torchvision `0.29.1+cpu`, and `torch.cuda.is_available() == False`, despite an RTX 4050 Laptop GPU with 6,141 MiB reported by NVIDIA driver 592.82.

## Installation

Use the project virtual environment, not another Python installation. The optional `requirements-cuda.txt` pins the matching CUDA 13.0 PyTorch/Torchvision builds from the [official PyTorch wheel index](https://download.pytorch.org/whl/cu130/). Standard CPU-capable installation remains in `pyproject.toml`. See the [official PyTorch installation guidance](https://pytorch.org/get-started/locally/) when setting up different hardware/drivers.

From **Command Prompt**, in the repository root:

```cmd
.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback')"
```

These CUDA wheels are large (the PyTorch download is approximately 2 GB). The `+cu130` version suffixes are intentional: an existing CPU-only package with the same base version must not satisfy this optional install request. No driver or system toolkit changes are made by the project instructions.

If a nonstandard/bundled environment has no pip, an existing system pip can bootstrap the project pip:

```cmd
python -m pip --python .venv\Scripts\python.exe install pip
```

Never run simultaneous application/test processes while replacing Torch packages in the environment. A normal Python-created virtual environment usually already includes pip.

## Run on the RTX GPU

```cmd
.venv\Scripts\python.exe -m football_cv --config configs\all22_hd.yaml --input "data\Unlabeled set 2.mp4" --device 0 --focus-annotations annotations\set_2_priority_sample.json --trails
```

Startup logs name the selected GPU and runtime. `--device 0` explicitly requests the first CUDA device and fails clearly if unavailable. `--device auto` uses CUDA when available and otherwise CPU; `--device cpu` remains supported. Existing presets use `auto`. Model weights, confidence, resolution, tracking logic, and precision are unchanged by selecting CUDA. This is acceleration of person inference, not improved position recognition or tracking validation.

## Compare processing speed

```cmd
.venv\Scripts\python.exe scripts\benchmark_devices.py "data\Unlabeled set 2.mp4" --config configs\all22_hd.yaml --devices cpu 0 --frames 120 --warmup 10 --output outputs\device_speed_new
```

The two devices process the same consecutive source frames with the same full-precision model/configuration. Ten startup frames are excluded, and CUDA is explicitly synchronized at timing boundaries. The report includes warmed decode/resize/cut-check/detect/track/annotate loop FPS, detector FPS, median/p95 loop latency, and peak memory allocated through Torch. It excludes GUI event waits, focus processing, telemetry export, video encoding, model loading, and warmup. Peak Torch allocation is not total GPU memory use.

This is a short local comparison, not a guaranteed sustained real-time rate. Driver/power/thermal state, operating-system caching, display, output saving, and footage content can affect results. The application still processes every frame and displays at processing speed, so a faster GPU does not itself introduce real-time playback pacing.

## Verified locally — October 27, 2026

Windows, Python 3.12.14, PyTorch `2.14.1+cu130`, Torchvision `0.29.1+cu130`, NVIDIA driver 592.82, RTX 4050 Laptop GPU. CUDA tensor multiplication and Torchvision CUDA NMS succeeded; `pip check` reported no broken requirements. Python 3.11 remains the project target but was not tested here.

The command above processed 120 consecutive set 2 frames after 10 warmup frames, with YOLO11 nano, 1280 inference size, 1920 processing width, confidence .25, and NMS IoU .45:

| Measurement | CPU | RTX 4050 |
| --- | ---: | ---: |
| Warm processing loop FPS | 8.21 | 40.21 |
| Detection FPS | 8.85 | 56.79 |
| Median loop latency (ms) | 121.69 | 23.18 |
| 95th percentile loop latency (ms) | 130.60 | 32.46 |

The measured loop speedup was 4.9× without lowering resolution or changing precision. This does not reach the source's 60 FPS at these settings. GPU peak Torch allocation was 96.2 MiB, excluding other CUDA/driver allocations. The ignored raw report is `outputs/rtx4050_benchmark/device_benchmark.json`.

The full headless GPU application also processed and saved all 446 set 2 frames, with trails and focus annotations: the output decoded as 1920×1080 at 60 FPS and the JSONL contained 446 frame records / 1,902 measured observations. Output rate metadata preserves the source playback rate and is not a processing-speed measurement. GUI interaction was not tested in this run.

GPU detection evaluation retained the previous reviewed-box results: 14/14 set 1 and 10/14 set 2 priority-defender boxes matched at IoU .50. These are small partial-label samples, not general detection or tracking accuracy. The test suite passed 64 tests, with the optional integration test skipped; the actual-video GPU runs above were performed separately. Reports, video, telemetry, weights, and the environment remain ignored by Git.
