#!/usr/bin/env bash
# Cross-compile llama-phase-profiler for Android arm64-v8a via NDK.
# Requires: ANDROID_NDK (or ANDROID_NDK_HOME) env var pointing at NDK r23+.
#
# Output:
#   profiler/build-android/llama-phase-profiler
#   android-ml-benchmark/binaries/llama-phase-profiler   (auto-copied)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build-android"
INSTALL_DIR="${REPO_ROOT}/android-ml-benchmark/binaries"

NDK_PATH="${ANDROID_NDK:-${ANDROID_NDK_HOME:-}}"
if [[ -z "${NDK_PATH}" || ! -d "${NDK_PATH}" ]]; then
    cat >&2 <<EOF
ERROR: ANDROID_NDK / ANDROID_NDK_HOME is not set or not a directory.

Install the Android NDK (r23c or newer), then set one of:
    export ANDROID_NDK=/path/to/android-ndk-r26d
    export ANDROID_NDK_HOME=/path/to/android-ndk-r26d

On Windows use Git Bash or WSL, or run build_android.ps1 instead.
EOF
    exit 1
fi

TOOLCHAIN="${NDK_PATH}/build/cmake/android.toolchain.cmake"
if [[ ! -f "${TOOLCHAIN}" ]]; then
    echo "ERROR: NDK toolchain file not found: ${TOOLCHAIN}" >&2
    exit 1
fi

ABI="${ABI:-arm64-v8a}"
API="${API:-24}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

echo "==> Configuring (ABI=${ABI}, API=${API})"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
    -DANDROID_ABI="${ABI}" \
    -DANDROID_PLATFORM="android-${API}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_LLAMAFILE=ON \
    -DGGML_CPU_AARCH64=ON \
    -DGGML_OPENMP=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_CURL=OFF

echo "==> Building (jobs=${JOBS})"
cmake --build "${BUILD_DIR}" --target llama-phase-profiler -j"${JOBS}"

BIN="${BUILD_DIR}/llama-phase-profiler"
if [[ ! -f "${BIN}" ]]; then
    echo "ERROR: build succeeded but ${BIN} not found" >&2
    exit 1
fi

mkdir -p "${INSTALL_DIR}"
cp -f "${BIN}" "${INSTALL_DIR}/llama-phase-profiler"
chmod +x "${INSTALL_DIR}/llama-phase-profiler"

echo
echo "Built:      ${BIN}"
echo "Installed:  ${INSTALL_DIR}/llama-phase-profiler"
echo
echo "Push to device and test:"
echo "  adb push ${INSTALL_DIR}/llama-phase-profiler /data/local/tmp/"
echo "  adb shell chmod 755 /data/local/tmp/llama-phase-profiler"
