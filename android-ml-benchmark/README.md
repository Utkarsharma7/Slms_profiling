# Android llama.cpp deep profiler (per-operator)

This repo is trimmed down to one goal: **granular, per-phase + per-operator profiling
of GGUF (llama.cpp) inference on Android over USB**.

It consists of:
- **`profiler/`**: a tiny C++ Android binary (`llama-phase-profiler`) built on llama.cpp that emits a single JSON document.
- **`android-ml-benchmark/`**: a minimal Python harness that runs that binary via ADB across GGUF variants and generates a cross-variant markdown/CSV report.

---

## Quick start

```bash
cd android-ml-benchmark

# 1) Put your GGUF models in:
#   android-ml-benchmark/models/
#
# 2) Build the Android profiler binary (outputs/copies into android-ml-benchmark/binaries/):
#   - Windows PowerShell:  pwsh -File ..\profiler\build_android.ps1
#   - Bash/WSL/Git Bash:   bash ../profiler/build_android.sh
#
# 3) Run a sweep across variants (example):
python profile_llm.py --model-glob "Llama-3.2-1B-Instruct-*.gguf" --n-gen 64 --repeats 3 --cooldown 30

# 4) Generate a cross-variant report:
python phase_report.py --output results/phase_profile/report.md --csv-dir results/phase_profile/csv
```

---

## What you get

Per run (JSON from the native binary):

- **Phases (ms)**: `model_load`, `context_init`, `tokenize`, `prefill`, `decode`, `sample`, `detokenize`, `teardown` (+ `total`).
- **Memory (MB)**: RSS snapshots at phase boundaries + peak RSS.
- **Per-op time (ms)**: ggml operator aggregates for `prefill` and `decode`.
- **Per-sublayer time (ms)**: aggregates like `q_proj`, `k_proj`, `v_proj`, `o_proj`, `ffn_up`, `ffn_down`, `rms_norm`, `rope`.
- Optional **per-step decode latency series** and optional `raw_ops[]`.

Across repeats per variant, `profile_llm.py` aggregates every numeric leaf into:
`{median, p5, p95, min, max, n}`.

---

## Project layout

```
android-ml-benchmark/
├── profile_llm.py                    # run llama-phase-profiler across GGUF variants (repeats + aggregation)
├── phase_report.py                   # markdown + CSV report from all_variants.json
├── adb_interface.py                  # adb helpers
├── device_info.py                    # device detection + recommended defaults
├── system_profiler.py                # optional CPU/GPU/temp sampling (used by profile_llm.py)
├── prompts/fixed_128.txt             # fixed prompt for reproducible profiling
├── backends/
│   ├── base.py                       # ADB push/shell helpers
│   └── llamacpp_phase_profiler.py    # runs the native profiler binary and parses JSON
├── binaries/                         # should contain llama-phase-profiler (built from ../profiler/)
├── models/                           # your GGUFs live here (not tracked by git)
└── results/phase_profile/            # outputs

profiler/
├── src/llama_phase_profiler.cpp      # native profiler
├── CMakeLists.txt                    # builds llama-phase-profiler (pins llama.cpp)
├── build_android.sh                  # NDK cross-compile (Bash)
├── build_android.ps1                 # NDK cross-compile (PowerShell)
└── README.md
```

---

## Troubleshooting

- **No device**: run `adb devices` and ensure USB debugging is authorized.
- **Binary not found**: build `../profiler/` and ensure `android-ml-benchmark/binaries/llama-phase-profiler` exists.
- **Model OOM / killed**: use smaller quant or reduce model set; `profile_llm.py` can skip large models unless you pass `--skip-ram-check`.
- **Thermal throttling**: keep `SystemProfiler` enabled (default) and increase `--cooldown` between repeats.
