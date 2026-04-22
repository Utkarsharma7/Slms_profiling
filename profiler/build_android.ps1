# Windows PowerShell counterpart of build_android.sh
# Cross-compiles llama-phase-profiler for Android arm64-v8a via NDK.
#
# Prereqs:
#   - CMake 3.22+ on PATH
#   - Ninja on PATH (comes with Android Studio's "CMake" SDK component)
#   - $env:ANDROID_NDK or $env:ANDROID_NDK_HOME pointing at NDK r23c+
#
# Output:
#   profiler\build-android\llama-phase-profiler
#   android-ml-benchmark\binaries\llama-phase-profiler

$ErrorActionPreference = "Stop"

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot   = Resolve-Path (Join-Path $ScriptDir "..")
$BuildDir   = Join-Path $ScriptDir "build-android"
$InstallDir = Join-Path $RepoRoot "android-ml-benchmark\binaries"

$NdkPath = $env:ANDROID_NDK
if (-not $NdkPath) { $NdkPath = $env:ANDROID_NDK_HOME }
if (-not $NdkPath -or -not (Test-Path $NdkPath)) {
    Write-Error @"
ANDROID_NDK / ANDROID_NDK_HOME is not set or not a directory.

Install the Android NDK (r23c or newer) and set:
    `$env:ANDROID_NDK = 'C:\Android\Sdk\ndk\26.1.10909125'
"@
}

$Toolchain = Join-Path $NdkPath "build\cmake\android.toolchain.cmake"
if (-not (Test-Path $Toolchain)) {
    Write-Error "NDK toolchain file not found: $Toolchain"
}

$Abi = if ($env:ABI) { $env:ABI } else { "arm64-v8a" }
$Api = if ($env:API) { $env:API } else { "24" }

Write-Host "==> Configuring (ABI=$Abi, API=$Api)"
cmake -S $ScriptDir -B $BuildDir `
    -G Ninja `
    "-DCMAKE_TOOLCHAIN_FILE=$Toolchain" `
    "-DANDROID_ABI=$Abi" `
    "-DANDROID_PLATFORM=android-$Api" `
    "-DCMAKE_BUILD_TYPE=Release" `
    "-DGGML_LLAMAFILE=ON" `
    "-DGGML_CPU_AARCH64=ON" `
    "-DGGML_OPENMP=OFF" `
    "-DLLAMA_BUILD_TESTS=OFF" `
    "-DLLAMA_BUILD_EXAMPLES=OFF" `
    "-DLLAMA_BUILD_SERVER=OFF" `
    "-DLLAMA_CURL=OFF"
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

Write-Host "==> Building"
cmake --build $BuildDir --target llama-phase-profiler
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }

$Bin = Join-Path $BuildDir "llama-phase-profiler"
if (-not (Test-Path $Bin)) {
    Write-Error "Build succeeded but $Bin not found"
}

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Copy-Item -Force $Bin (Join-Path $InstallDir "llama-phase-profiler")

Write-Host ""
Write-Host "Built:      $Bin"
Write-Host "Installed:  $(Join-Path $InstallDir 'llama-phase-profiler')"
Write-Host ""
Write-Host "Push to device:"
Write-Host "  adb push `"$InstallDir\llama-phase-profiler`" /data/local/tmp/"
Write-Host "  adb shell chmod 755 /data/local/tmp/llama-phase-profiler"
