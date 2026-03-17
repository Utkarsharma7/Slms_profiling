"""
Universal Android device information collector.
Works on any Android device connected via ADB.
Detects SoC, GPU, CPU cores/frequencies, RAM, Android version, etc.

Usage standalone:
    python device_info.py              # print device info
    python device_info.py --json       # JSON output

Usage in code:
    from device_info import get_device_info
    info = get_device_info()
    print(info["soc"], info["ram_total_mb"], info["gpu"])
"""
import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from adb_interface import run_adb, ADBError


def _adb_shell(cmd: str, timeout: int = 10) -> str:
    try:
        r = run_adb("shell", cmd, check=False, timeout=timeout)
        return (r.stdout or "").strip()
    except ADBError:
        return ""


def _getprop(key: str) -> str:
    return _adb_shell(f"getprop {key}")


def get_device_info() -> dict:
    """Collect comprehensive device info from any connected Android device."""
    info = {}

    # ── Basic device identity ──
    info["manufacturer"] = _getprop("ro.product.manufacturer")
    info["model"] = _getprop("ro.product.model")
    info["device"] = _getprop("ro.product.device")
    info["brand"] = _getprop("ro.product.brand")
    info["android_version"] = _getprop("ro.build.version.release")
    info["sdk_version"] = _getprop("ro.build.version.sdk")
    info["build_fingerprint"] = _getprop("ro.build.fingerprint")
    info["kernel_version"] = _adb_shell("uname -r")

    # ── SoC / Chipset ──
    info["soc"] = (
        _getprop("ro.hardware.chipname") or
        _getprop("ro.board.platform") or
        _getprop("ro.hardware") or
        "unknown"
    )
    info["soc_model"] = _getprop("ro.soc.model") or _getprop("ro.mediatek.platform") or ""
    info["abi"] = _getprop("ro.product.cpu.abi")
    info["abi_list"] = _getprop("ro.product.cpu.abilist")

    # ── CPU info ──
    cpuinfo = _adb_shell("cat /proc/cpuinfo")
    info["cpu_cores"] = cpuinfo.count("processor\t:")
    if info["cpu_cores"] == 0:
        info["cpu_cores"] = cpuinfo.count("processor :")
    if info["cpu_cores"] == 0:
        online = _adb_shell("cat /sys/devices/system/cpu/online")
        if online:
            parts = online.split("-")
            if len(parts) == 2:
                info["cpu_cores"] = int(parts[1]) + 1

    # Extract CPU implementer/part for each unique core
    cpu_parts = set()
    for line in cpuinfo.splitlines():
        if "CPU part" in line:
            cpu_parts.add(line.split(":")[-1].strip())
    info["cpu_parts"] = list(cpu_parts)

    # Per-core max frequency
    freq_output = _adb_shell(
        "cat /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq 2>/dev/null"
    )
    max_freqs = []
    for line in freq_output.splitlines():
        line = line.strip()
        if line.isdigit():
            max_freqs.append(int(line) // 1000)
    info["cpu_max_freq_mhz"] = max_freqs

    # Cluster info: group cores by max frequency
    if max_freqs:
        clusters = {}
        for i, freq in enumerate(max_freqs):
            clusters.setdefault(freq, []).append(i)
        info["cpu_clusters"] = [
            {"cores": cores, "max_mhz": freq}
            for freq, cores in sorted(clusters.items())
        ]

    # ── GPU info ──
    gpu_renderer = _adb_shell(
        "dumpsys SurfaceFlinger 2>/dev/null | grep -i 'GLES' | head -1"
    )
    info["gpu_renderer"] = gpu_renderer.strip() if gpu_renderer else ""

    if not info["gpu_renderer"]:
        gpu_alt = _adb_shell("getprop ro.hardware.egl")
        info["gpu_renderer"] = gpu_alt

    gpu_freq = _adb_shell("cat /sys/class/devfreq/*/max_freq 2>/dev/null | head -1")
    if gpu_freq and gpu_freq.strip().isdigit():
        freq_val = int(gpu_freq.strip())
        if freq_val > 1_000_000:
            info["gpu_max_freq_mhz"] = freq_val // 1_000_000
        elif freq_val > 1000:
            info["gpu_max_freq_mhz"] = freq_val // 1000
        else:
            info["gpu_max_freq_mhz"] = freq_val

    # ── RAM ──
    meminfo = _adb_shell("cat /proc/meminfo")
    for line in meminfo.splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2:
                info["ram_total_mb"] = int(parts[1]) // 1024
        elif line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                info["ram_available_mb"] = int(parts[1]) // 1024

    # ── Storage ──
    df_out = _adb_shell("df /data | tail -1")
    if df_out:
        parts = df_out.split()
        if len(parts) >= 4:
            try:
                info["storage_total_mb"] = int(parts[1]) // 1024 if parts[1].isdigit() else None
                info["storage_free_mb"] = int(parts[3]) // 1024 if parts[3].isdigit() else None
            except (ValueError, IndexError):
                pass

    # ── NNAPI / NPU ──
    info["nnapi_available"] = bool(_adb_shell(
        "ls /vendor/lib64/libneuralnetworks.so 2>/dev/null || "
        "ls /system/lib64/libneuralnetworks.so 2>/dev/null"
    ))
    # MediaTek APU
    info["has_mediatek_apu"] = bool(_adb_shell("ls /dev/apusys* 2>/dev/null"))
    # Qualcomm HTA/DSP
    info["has_qualcomm_dsp"] = bool(_adb_shell("ls /dev/adsprpc-smd* 2>/dev/null"))
    # Samsung NPU (Exynos)
    info["has_samsung_npu"] = bool(_adb_shell("ls /dev/vertex* 2>/dev/null"))

    # ── Vulkan support ──
    vulkan = _adb_shell("cmd gpu vkjson 2>/dev/null | head -5")
    info["vulkan_available"] = "apiVersion" in vulkan if vulkan else False

    # ── Thermal zones (list names for vendor-agnostic mapping) ──
    tz_types = _adb_shell("cat /sys/class/thermal/thermal_zone*/type 2>/dev/null")
    info["thermal_zones"] = [z.strip() for z in tz_types.splitlines() if z.strip()]

    # ── Battery ──
    batt = _adb_shell("dumpsys battery")
    for line in batt.splitlines():
        line = line.strip()
        if line.startswith("level:"):
            try:
                info["battery_level_pct"] = int(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("temperature:"):
            try:
                info["battery_temp_c"] = round(int(line.split(":")[1].strip()) / 10.0, 1)
            except (ValueError, IndexError):
                pass

    return info


def get_recommended_settings(info: dict) -> dict:
    """Return recommended benchmark settings based on device capabilities."""
    rec = {}

    ram = info.get("ram_total_mb", 4096)
    cores = info.get("cpu_cores", 4)

    # Thread count: use big cores only (typically half), max 4 for stability
    rec["threads"] = min(cores // 2, 4) if cores >= 4 else max(1, cores - 1)

    # Max model size based on available RAM (leave ~2GB for system)
    available = info.get("ram_available_mb", ram // 2)
    rec["max_model_size_mb"] = max(256, available - 512)

    # Wait time: slower SoCs need more time
    max_freq = max(info.get("cpu_max_freq_mhz", [2000]) or [2000])
    if max_freq < 1500:
        rec["wait_seconds"] = 90
    elif max_freq < 2000:
        rec["wait_seconds"] = 60
    else:
        rec["wait_seconds"] = 45

    # Available delegates
    delegates = ["cpu"]
    if info.get("nnapi_available"):
        delegates.append("nnapi")
    # GPU delegate generally available if there's a GPU renderer
    if info.get("gpu_renderer"):
        delegates.append("gpu")
    rec["available_delegates"] = delegates

    # GGUF model recommendations
    if ram >= 8192:
        rec["max_gguf_params"] = "3.8B (Q4_K_M)"
    elif ram >= 6144:
        rec["max_gguf_params"] = "2B (Q4_K_M)"
    elif ram >= 4096:
        rec["max_gguf_params"] = "1.5B (Q4_K_M)"
    else:
        rec["max_gguf_params"] = "0.5B (Q4_K_M)"

    return rec


def print_device_info(info: dict, recommendations: dict | None = None):
    """Pretty-print device info."""
    w = 70
    print("=" * w)
    print("  DEVICE INFORMATION")
    print("=" * w)
    print(f"  Device:       {info.get('brand', '?')} {info.get('model', '?')}")
    print(f"  Android:      {info.get('android_version', '?')} (SDK {info.get('sdk_version', '?')})")
    print(f"  SoC:          {info.get('soc', '?')} {info.get('soc_model', '')}")
    print(f"  ABI:          {info.get('abi', '?')}")
    print(f"  CPU Cores:    {info.get('cpu_cores', '?')}")
    if info.get("cpu_clusters"):
        for cl in info["cpu_clusters"]:
            print(f"    Cluster:    {len(cl['cores'])}x cores @ {cl['max_mhz']} MHz")
    if info.get("gpu_renderer"):
        print(f"  GPU:          {info['gpu_renderer']}")
    if info.get("gpu_max_freq_mhz"):
        print(f"  GPU Max Freq: {info['gpu_max_freq_mhz']} MHz")
    print(f"  RAM:          {info.get('ram_total_mb', '?')} MB total, {info.get('ram_available_mb', '?')} MB available")
    if info.get("storage_free_mb"):
        print(f"  Storage:      {info['storage_free_mb']} MB free")
    print(f"  NNAPI:        {'Yes' if info.get('nnapi_available') else 'No'}")
    if info.get("has_mediatek_apu"):
        print(f"  MediaTek APU: Yes")
    if info.get("has_qualcomm_dsp"):
        print(f"  Qualcomm DSP: Yes")
    if info.get("has_samsung_npu"):
        print(f"  Samsung NPU:  Yes")
    print(f"  Vulkan:       {'Yes' if info.get('vulkan_available') else 'No'}")
    if info.get("battery_level_pct") is not None:
        print(f"  Battery:      {info['battery_level_pct']}% ({info.get('battery_temp_c', '?')}°C)")
    print("=" * w)

    if recommendations:
        print(f"\n  RECOMMENDED SETTINGS")
        print(f"  " + "-" * 40)
        print(f"  Threads:          {recommendations['threads']}")
        print(f"  Wait time:        {recommendations['wait_seconds']}s")
        print(f"  Max model size:   {recommendations['max_model_size_mb']} MB")
        print(f"  Max GGUF:         {recommendations['max_gguf_params']}")
        print(f"  Delegates:        {', '.join(recommendations['available_delegates'])}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Detect Android device capabilities")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--save", type=Path, default=None, help="Save to file")
    args = parser.parse_args()

    info = get_device_info()
    rec = get_recommended_settings(info)

    if args.json:
        combined = {"device": info, "recommendations": rec}
        output = json.dumps(combined, indent=2)
        print(output)
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            with open(args.save, "w") as f:
                f.write(output)
            print(f"Saved to: {args.save}", file=sys.stderr)
    else:
        print_device_info(info, rec)
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            with open(args.save, "w") as f:
                json.dump({"device": info, "recommendations": rec}, f, indent=2)
            print(f"Saved to: {args.save}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
