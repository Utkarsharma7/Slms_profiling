## Progress log

### 2026-04-22

#### Scope decision
- Repo is now focused on **deep, granular llama.cpp (GGUF) inference profiling on Android**:
  - per-phase timings
  - per-operator (ggml op) time breakdown
  - per-sublayer aggregates
  - optional system/thermal sampling during runs

#### Cleanup performed
- Deleted all non-profiler code paths:
  - removed TFLite / ONNX / PyTorch benchmarking backends and scripts
  - removed coarse benchmarking/reporting/download tooling
  - removed large unused binary artifacts (e.g. ONNX runtime library)
- Updated docs to reflect the profiler-only workflow:
  - `android-ml-benchmark/README.md` rewritten for the deep-profiler pipeline only
- Added repo-wide ignores:
  - added root `.gitignore` to ignore `profiler/build-android/`, `android-ml-benchmark/{models,binaries,results}/`, and Python caches/venvs
  - updated `android-ml-benchmark/.gitignore` for the profiler-only binary (`llama-phase-profiler`)

#### Current architecture (what remains)
- **Native profiler**: `profiler/`
  - builds `llama-phase-profiler` for Android (arm64-v8a)
- **Python harness**: `android-ml-benchmark/`
  - `profile_llm.py`: runs the profiler binary across GGUF variants + repeats + aggregation
  - `phase_report.py`: renders markdown + CSV from `results/phase_profile/all_variants.json`
  - `backends/llamacpp_phase_profiler.py`: pushes/runs binary via ADB and parses JSON
  - `backends/base.py`: shared ADB helpers (`push`, `shell_cmd`, `rm`) + `DEVICE_TMP`
  - `adb_interface.py`: subprocess wrapper for `adb`
  - `device_info.py`: device detection + recommended defaults used by the sweep
  - `system_profiler.py`: optional CPU/GPU/temp sampling and throttling heuristic

#### How to run (minimal)
1. Put GGUF models in `android-ml-benchmark/models/`.
2. Build the Android binary (copies into `android-ml-benchmark/binaries/`):
   - PowerShell: `pwsh -File profiler/build_android.ps1`
   - Bash: `bash profiler/build_android.sh`
3. Run sweep:
   - `python android-ml-benchmark/profile_llm.py --model-glob "YourModel-*.gguf" --n-gen 64 --repeats 3 --cooldown 30`
4. Generate report:
   - `python android-ml-benchmark/phase_report.py --output android-ml-benchmark/results/phase_profile/report.md --csv-dir android-ml-benchmark/results/phase_profile/csv`

