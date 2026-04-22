# llama-phase-profiler

A tiny C++ binary built on the llama.cpp C API that profiles GGUF LLM
inference on Android **per phase** (`model_load`, `context_init`,
`tokenize`, `prefill`, `decode`, `sample`, `detokenize`, `teardown`) and
emits per-op / per-sublayer aggregates captured via the ggml eval
callback.

Output is a single self-describing JSON document on stdout. See
`android-ml-benchmark/backends/llamacpp_phase_profiler.py` for the
consumer side.

---

## Build for Android (arm64-v8a)

### Prerequisites

- CMake >= 3.22
- Ninja (Android Studio ships this under `Sdk/cmake/<ver>/bin/ninja[.exe]`)
- Android NDK r23c or newer. Set `ANDROID_NDK` (or `ANDROID_NDK_HOME`)
  to the NDK root.

The first build will clone llama.cpp into `profiler/third_party/llama.cpp`
via CMake FetchContent. To build offline, check out llama.cpp there
yourself beforehand.

### Linux / macOS / WSL / Git Bash

```bash
export ANDROID_NDK=/path/to/android-ndk-r26d
bash profiler/build_android.sh
```

### Windows PowerShell

```powershell
$env:ANDROID_NDK = 'C:\Android\Sdk\ndk\26.1.10909125'
pwsh -File profiler\build_android.ps1
```

Output binary:

```
profiler/build-android/llama-phase-profiler
android-ml-benchmark/binaries/llama-phase-profiler   (auto-copied)
```

Size is ~5-15 MB depending on NDK version.

---

## llama.cpp version pin

`CMakeLists.txt` defaults `LLAMACPP_GIT_TAG` to a known-good release tag.
If you want a different one, override at configure time:

```bash
cmake -S profiler -B profiler/build-android \
  -DLLAMACPP_GIT_TAG=master \
  # ...rest of the flags from build_android.sh
```

The eval-callback API (`llama_context_params::cb_eval`) and sampler API
(`llama_sampler_chain_init`, `llama_sampler_init_greedy`,
`llama_sampler_sample`) this project relies on are stable in llama.cpp
master as of 2025; older forks will not compile.

---

## Command-line

```
llama-phase-profiler \
  -m /data/local/tmp/model.gguf \
  --prompt-file /data/local/tmp/prompt.txt \
  --n-gen 64 \
  -t 4 \
  [--seed 42] [--n-ctx 2048] [--dump-raw-ops]
```

- `-m` - path to GGUF model on device
- `--prompt-file` - UTF-8 text file; contents used as the entire prompt
- `--n-gen` - number of tokens to decode
- `-t` - CPU threads
- `--seed` - currently unused by greedy sampler but recorded in meta
- `--n-ctx` - context size (prompt + gen must fit)
- `--dump-raw-ops` - append every op event to `raw_ops[]`; output can get
  large for big prompts. Off by default.
