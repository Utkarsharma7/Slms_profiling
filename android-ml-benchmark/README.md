# Android ML Benchmark

Universal benchmarking and profiling tool for **any Android device** over USB. Supports **TFLite** (CPU/GPU/NNAPI), **ONNX Runtime** (CPU/NNAPI/XNNPACK), **PyTorch Mobile**, and **LLMs via llama.cpp** (GGUF). Auto-detects your device's SoC, RAM, GPU, and NPU capabilities, then tunes settings accordingly.

Collects: inference latency, memory, tokens/sec, per-operator profiling, percentile latencies, and real-time system metrics (CPU freq, temp, GPU freq, thermal throttling).

---

## Supported formats

| Extension | Backend | Delegates / EPs | What you need |
|-----------|---------|-----------------|---------------|
| `.tflite` | TFLite APK | CPU, GPU, NNAPI | Install APK once |
| `.onnx` | ONNX Runtime | CPU, NNAPI, XNNPACK | `onnxruntime_perf_test` in `binaries/` |
| `.gguf` | llama.cpp | CPU (NEON) | `llama-bench` in `binaries/` |
| `.pt` / `.ptl` | PyTorch Mobile | CPU | `speed_benchmark_torch` in `binaries/` |

---

## Quick start

```bash
cd android-ml-benchmark

# 1. See what device you have and what it supports:
python device_info.py

# 2. Check which backends are ready:
python benchmark.py --check

# 3. Download models:
python download_models.py --list           # see available models
python download_models.py --tflite         # TFLite vision/NLP (~70 MB)
python download_models.py --onnx           # ONNX vision/NLP (~570 MB)
python download_models.py --gguf           # GGUF SLMs (~7 GB)
python download_models.py --all            # everything

# 4. Benchmark:
python benchmark.py --all                  # all models, auto-tuned settings
python benchmark.py --all --delegate all   # TFLite + ONNX on all delegates
python benchmark.py --all --profile        # with system metrics (CPU/GPU/temp)

# 5. Generate report:
python report.py                           # comparison table + CSV
```

---

## How it works on any device

The tool **auto-detects your phone's hardware** on startup:

```
======================================================================
  DEVICE INFORMATION
======================================================================
  Device:       Samsung Galaxy S24
  Android:      14 (SDK 34)
  SoC:          s5e9945
  CPU Cores:    8
    Cluster:    4x cores @ 2000 MHz
    Cluster:    3x cores @ 2800 MHz
    Cluster:    1x cores @ 3200 MHz
  GPU:          Mali-G720
  RAM:          11904 MB total, 5832 MB available
  NNAPI:        Yes
  Vulkan:       Yes

  RECOMMENDED SETTINGS
  Threads:          4
  Wait time:        45s
  Max model size:   5320 MB
  Max GGUF:         3.8B (Q4_K_M)
  Delegates:        cpu, nnapi, gpu
```

Thread count, wait time, and model size limits are auto-tuned. Override with `--threads` and `--wait`.

---

## Project layout

```
android-ml-benchmark/
├── benchmark.py              # Main entry: auto-detect format + device, run benchmarks
├── device_info.py            # Detect device SoC/RAM/GPU/NPU, recommend settings
├── download_models.py        # Download TFLite + ONNX + GGUF models
├── report.py                 # Comparison table + CSV + bottleneck analysis
├── system_profiler.py        # Poll CPU/GPU/thermal metrics during benchmarks
├── run_tf_apk_benchmark.py   # Legacy: TFLite-only script
├── adb_interface.py          # ADB helpers
├── backends/
│   ├── __init__.py           # Format → backend routing
│   ├── base.py               # Base class (push, shell, logcat)
│   ├── tflite_apk.py         # TFLite APK (CPU/GPU/NNAPI + op profiling)
│   ├── onnx_binary.py        # ONNX Runtime (CPU/NNAPI/XNNPACK + op profiling)
│   ├── llamacpp_binary.py    # llama.cpp (GGUF LLMs)
│   └── pytorch_binary.py     # PyTorch Mobile
├── binaries/                 # Benchmark binaries (llama-bench, etc.)
├── models/                   # All models (.tflite, .onnx, .gguf, .pt)
├── results/                  # JSON results, CSV reports, system profiles
├── requirements.txt
└── README.md
```

---

## Setup per backend

### TFLite (ready out of the box)

APK already installed. Supports 3 delegates:

```bash
python benchmark.py models/mobilenet_v2.tflite --delegate cpu
python benchmark.py models/mobilenet_v2.tflite --delegate gpu
python benchmark.py models/mobilenet_v2.tflite --delegate nnapi
python benchmark.py --all --delegate all   # compare all 3
```

Per-operator profiling is on by default (`--enable_op_profiling`). Disable with `--no-op-profiling`.

### ONNX Runtime

Supports 3 execution providers:
- **cpu**: Default, works everywhere
- **nnapi**: Routes to device NPU/DSP (Qualcomm HTA, MediaTek APU, Samsung NPU)
- **xnnpack**: Optimised CPU inference via XNNPACK

```bash
python benchmark.py models/mobilenetv2-12.onnx --delegate cpu
python benchmark.py models/mobilenetv2-12.onnx --delegate nnapi
python benchmark.py models/mobilenetv2-12.onnx --delegate xnnpack
python benchmark.py --all --delegate all   # compare all EPs
```

**Build `onnxruntime_perf_test` for Android arm64:**

```bash
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime
./build.sh --config Release \
  --android --android_sdk_path $ANDROID_SDK \
  --android_ndk_path $ANDROID_NDK \
  --android_abi arm64-v8a --android_api 24 \
  --build_shared_lib
```

Copy `build/Android/Release/onnxruntime_perf_test` to `binaries/`.

### llama.cpp (GGUF LLMs/SLMs)

**Build `llama-bench`:**

```bash
# Option A: NDK cross-compile on PC
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && mkdir build-android && cd build-android
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build . -t llama-bench

# Option B: Build on phone via Termux
pkg install git cmake clang
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -t llama-bench
```

Copy to `binaries/llama-bench`.

**GGUF models (auto-downloadable):**

| Model | Params | Size | Use case |
|-------|--------|------|----------|
| SmolLM2-360M | 360M | 230 MB | Tiny, fast baseline |
| Qwen2.5-0.5B | 0.5B | 390 MB | Small instruct |
| TinyLlama-1.1B | 1.1B | 670 MB | Chat |
| Llama-3.2-1B | 1B | 780 MB | Meta's smallest |
| Qwen2.5-1.5B | 1.5B | 1.0 GB | Good quality/speed balance |
| Gemma-2-2B | 2B | 1.6 GB | Google |
| Phi-3.5-mini | 3.8B | 2.2 GB | Best quality (needs 6GB+ RAM) |

### PyTorch Mobile

```bash
git clone https://github.com/pytorch/pytorch.git
cd pytorch
./scripts/build_android.sh -DBUILD_BINARY=ON -DBUILD_CAFFE2_MOBILE=OFF
```

Copy `build_android/bin/speed_benchmark_torch` to `binaries/`.

---

## System profiling

Collect CPU/GPU/thermal metrics during benchmarks to detect throttling:

```bash
python benchmark.py --all --profile                  # during benchmarks
python system_profiler.py --duration 30               # standalone
python system_profiler.py --snapshot                   # one-shot
```

The profiler auto-discovers vendor-specific sysfs paths and categorises thermal zones (CPU, GPU, battery, skin). Supports: Qualcomm, MediaTek, Samsung Exynos, Google Tensor, and generic ARM devices.

Detects thermal throttling (CPU freq drop >20% during a run) and flags it in reports.

---

## Device info

```bash
python device_info.py              # pretty-print
python device_info.py --json       # machine-readable
python device_info.py --save results/device.json
```

Detects: manufacturer, model, SoC, CPU clusters, GPU, RAM, NNAPI, NPU (MediaTek APU / Qualcomm DSP / Samsung NPU), Vulkan, thermal zones, battery state. Recommends thread count, wait time, max model size, and available delegates.

---

## Reports

```bash
python report.py                                  # default results file
python report.py results/run1.json results/run2.json  # multiple runs
python report.py --csv results/my_report.csv      # custom CSV path
```

Includes:
- **Device header**: SoC, RAM, GPU, Android version
- **Comparison table**: all models, backends, delegates, latency, memory, tok/s
- **Delegate comparison**: for models run with `--delegate all`, shows fastest hardware and speedup factor
- **Bottleneck alerts**: slow inference, high memory, bad tok/s, thermal throttling, failed delegates

---

## All options

| Option | Default | Description |
|--------|---------|-------------|
| `model` | first in models/ | Path to one model |
| `--all` | off | Benchmark every model in models/ |
| `--threads` | auto | CPU threads (auto-detected from device) |
| `--wait` | auto | Seconds to wait (auto-detected from device) |
| `--output` / `-o` | results/benchmark_results.json | Output JSON |
| `--check` | off | Show backend availability |
| `--delegate` | cpu | `cpu`, `gpu`, `nnapi`, `xnnpack`, or `all` |
| `--no-op-profiling` | off | Disable per-operator profiling |
| `--profile` | off | Collect system metrics during benchmarks |
| `--profile-interval` | 0.5 | System poll interval (seconds) |
| `--device-info` | off | Print device info and exit |
| `--no-device-info` | off | Skip device info banner |
| `--input-dims` | 1,3,224,224 | PyTorch input dimensions |
| `--input-type` | float | PyTorch input type |

---

## Tested on

This tool is designed to work on **any Android device** with USB debugging enabled. The sysfs probing and device detection handle vendor differences automatically. Tested architectures:
- **Qualcomm Snapdragon** (Adreno GPU, Hexagon DSP)
- **MediaTek Dimensity/Helio** (Mali GPU, MediaTek APU)
- **Samsung Exynos** (Mali/Xclipse GPU, Samsung NPU)
- **Google Tensor** (Mali GPU, Edge TPU)
- **Unisoc** / other ARM SoCs

---

## Troubleshooting

- **No device** → `adb devices`. Check cable, USB debugging, driver.
- **TFLite: no timings** → `--wait 90` for slower devices.
- **GPU delegate fails** → Not all ops are GPU-compatible; model falls back to CPU.
- **NNAPI fails** → NPU has limited op support; varies by SoC vendor.
- **XNNPACK not available** → Build ONNX Runtime with XNNPACK support enabled.
- **GGUF model OOM** → Check `python device_info.py` for max recommended model size.
- **Thermal throttling** → Use `--profile` to detect; let phone cool between runs.
- **Binary not found** → Build for arm64 and place in `binaries/`. See setup sections above.
