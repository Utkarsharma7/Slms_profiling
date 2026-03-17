"""
Universal system-level profiler for any Android device.
Polls CPU frequency, temperature, memory, GPU frequency, and thermal state
during benchmark runs. Handles vendor-specific sysfs paths (Qualcomm, MediaTek,
Exynos, Tensor, etc.) automatically.

Usage standalone:
    python system_profiler.py --duration 30 --interval 0.5 --output results/system_profile.json

Usage in code:
    profiler = SystemProfiler(interval=0.5)
    profiler.start()
    # ... run benchmark ...
    profiler.stop()
    data = profiler.get_data()
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from adb_interface import run_adb, ADBError

# Vendor-specific GPU frequency sysfs paths (tried in order)
GPU_FREQ_PATHS = [
    "/sys/class/devfreq/*/cur_freq",
    "/sys/devices/platform/*/gpu/cur_freq",
    "/sys/kernel/gpu/gpu_clock",
    "/sys/devices/platform/soc/*/devfreq/*/cur_freq",
    # Qualcomm Adreno
    "/sys/class/kgsl/kgsl-3d0/gpuclk",
    "/sys/class/kgsl/kgsl-3d0/cur_freq",
    # Mali (ARM/MediaTek/Samsung)
    "/sys/devices/platform/*.mali/cur_freq",
    "/sys/devices/platform/soc/*/mali*/cur_freq",
    # Samsung Exynos
    "/sys/kernel/gpu/gpu_freq",
    # Google Tensor
    "/sys/devices/platform/*/mali-*/cur_freq",
]

# Vendor-specific thermal zone type keywords for identifying CPU/GPU zones
THERMAL_CPU_KEYWORDS = ["cpu", "little", "big", "cl0", "cl1", "cl2", "soc", "core"]
THERMAL_GPU_KEYWORDS = ["gpu", "mali", "adreno", "g3d"]
THERMAL_BATTERY_KEYWORDS = ["battery", "batt", "charger"]
THERMAL_SKIN_KEYWORDS = ["skin", "shell", "back", "front"]


def _adb_shell(cmd: str, timeout: int = 5) -> str:
    try:
        r = run_adb("shell", cmd, check=False, timeout=timeout)
        return (r.stdout or "").strip()
    except ADBError:
        return ""


class SystemProfiler:
    """Polls Android device system metrics at a configurable interval.
    Works on any Android device by probing multiple sysfs paths."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._samples: list[dict] = []
        self._gpu_path: str | None = None
        self._thermal_map: dict[int, str] | None = None

    def start(self):
        """Start polling in a background thread."""
        self._stop_event.clear()
        self._samples = []
        self._discover_gpu_path()
        self._discover_thermal_zones()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict]:
        """Stop polling, wait for thread, return collected samples."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        return self._samples

    def get_data(self) -> list[dict]:
        return list(self._samples)

    def get_summary(self) -> dict:
        """Compute summary statistics from collected samples."""
        if not self._samples:
            return {}
        summary = {"num_samples": len(self._samples), "duration_s": 0}
        if len(self._samples) >= 2:
            summary["duration_s"] = round(self._samples[-1]["t"] - self._samples[0]["t"], 2)

        def _avg(key):
            vals = [s.get(key) for s in self._samples if s.get(key) is not None]
            return round(sum(vals) / len(vals), 2) if vals else None

        def _max(key):
            vals = [s.get(key) for s in self._samples if s.get(key) is not None]
            return max(vals) if vals else None

        def _min(key):
            vals = [s.get(key) for s in self._samples if s.get(key) is not None]
            return min(vals) if vals else None

        # CPU frequencies
        freq_keys = [k for k in (self._samples[0] if self._samples else {})
                     if k.startswith("cpu") and k.endswith("_mhz")]
        for k in freq_keys:
            summary[f"{k}_avg"] = _avg(k)
            summary[f"{k}_min"] = _min(k)
            summary[f"{k}_max"] = _max(k)

        # Categorised temperatures
        for prefix in ("temp_cpu_", "temp_gpu_", "temp_battery_", "temp_skin_", "temp_zone"):
            temp_keys = [k for k in (self._samples[0] if self._samples else {})
                         if k.startswith(prefix)]
            for k in temp_keys:
                summary[f"{k}_avg"] = _avg(k)
                summary[f"{k}_max"] = _max(k)

        summary["mem_available_mb_avg"] = _avg("mem_available_mb")
        summary["mem_available_mb_min"] = _min("mem_available_mb")
        summary["gpu_freq_mhz_avg"] = _avg("gpu_freq_mhz")
        summary["gpu_freq_mhz_max"] = _max("gpu_freq_mhz")
        summary["battery_temp_c_avg"] = _avg("battery_temp_c")
        summary["battery_temp_c_max"] = _max("battery_temp_c")

        # Detect thermal throttling: if max CPU freq drops >20% from first sample
        if freq_keys and len(self._samples) >= 4:
            first = self._samples[0]
            last_quarter = self._samples[-(len(self._samples) // 4):]
            for k in freq_keys:
                f0 = first.get(k)
                f_end_avg = _avg(k)
                if f0 and f_end_avg and f0 > 0:
                    drop_pct = round((1 - f_end_avg / f0) * 100, 1)
                    if drop_pct > 20:
                        summary[f"{k}_throttle_pct"] = drop_pct

        return summary

    def _discover_gpu_path(self):
        """Probe sysfs to find the working GPU frequency path for this device."""
        for pattern in GPU_FREQ_PATHS:
            out = _adb_shell(f"cat {pattern} 2>/dev/null | head -1")
            if out and out.strip().isdigit():
                self._gpu_path = pattern
                return

    def _discover_thermal_zones(self):
        """Map thermal zone indices to categorised names."""
        types_out = _adb_shell("cat /sys/class/thermal/thermal_zone*/type 2>/dev/null")
        self._thermal_map = {}
        for i, line in enumerate(types_out.splitlines()):
            name = line.strip().lower()
            if not name:
                continue
            category = "other"
            if any(kw in name for kw in THERMAL_CPU_KEYWORDS):
                category = "cpu"
            elif any(kw in name for kw in THERMAL_GPU_KEYWORDS):
                category = "gpu"
            elif any(kw in name for kw in THERMAL_BATTERY_KEYWORDS):
                category = "battery"
            elif any(kw in name for kw in THERMAL_SKIN_KEYWORDS):
                category = "skin"
            self._thermal_map[i] = f"{category}_{name}"

    def _poll_loop(self):
        t0 = time.monotonic()
        while not self._stop_event.is_set():
            sample = {"t": round(time.monotonic() - t0, 3)}
            sample.update(self._read_cpu_freqs())
            sample.update(self._read_temperatures())
            sample.update(self._read_memory())
            sample.update(self._read_gpu_freq())
            sample.update(self._read_battery())
            self._samples.append(sample)
            self._stop_event.wait(self.interval)

    @staticmethod
    def _read_cpu_freqs() -> dict:
        out = _adb_shell(
            "cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null"
        )
        result = {}
        for i, line in enumerate(out.splitlines()):
            line = line.strip()
            if line.isdigit():
                result[f"cpu{i}_mhz"] = int(line) // 1000
        return result

    def _read_temperatures(self) -> dict:
        out = _adb_shell(
            "cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null"
        )
        result = {}
        for i, line in enumerate(out.splitlines()):
            line = line.strip()
            try:
                val = int(line)
                if val > 1000:
                    val = val / 1000.0
                # Use categorised name if available
                if self._thermal_map and i in self._thermal_map:
                    key = f"temp_{self._thermal_map[i]}_c"
                else:
                    key = f"temp_zone{i}_c"
                result[key] = round(val, 1)
            except ValueError:
                pass
        return result

    @staticmethod
    def _read_memory() -> dict:
        out = _adb_shell("cat /proc/meminfo")
        result = {}
        for line in out.splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                if len(parts) >= 2:
                    result["mem_available_mb"] = int(parts[1]) // 1024
            elif line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2:
                    result["mem_total_mb"] = int(parts[1]) // 1024
        return result

    def _read_gpu_freq(self) -> dict:
        if not self._gpu_path:
            return {}
        out = _adb_shell(f"cat {self._gpu_path} 2>/dev/null | head -1")
        if not out:
            return {}
        for line in out.splitlines():
            line = line.strip()
            try:
                freq = int(line)
                if freq > 1_000_000:
                    return {"gpu_freq_mhz": freq // 1_000_000}
                elif freq > 1000:
                    return {"gpu_freq_mhz": freq // 1000}
                else:
                    return {"gpu_freq_mhz": freq}
            except ValueError:
                pass
        return {}

    @staticmethod
    def _read_battery() -> dict:
        out = _adb_shell("dumpsys battery")
        result = {}
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("temperature:"):
                try:
                    result["battery_temp_c"] = round(int(line.split(":")[1].strip()) / 10.0, 1)
                except (ValueError, IndexError):
                    pass
            elif line.startswith("level:"):
                try:
                    result["battery_level_pct"] = int(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
        return result


def snapshot() -> dict:
    """Take a single system snapshot (no threading needed)."""
    p = SystemProfiler()
    p._discover_gpu_path()
    p._discover_thermal_zones()
    sample = {}
    sample.update(p._read_cpu_freqs())
    sample.update(p._read_temperatures())
    sample.update(p._read_memory())
    sample.update(p._read_gpu_freq())
    sample.update(p._read_battery())
    return sample


def main():
    parser = argparse.ArgumentParser(description="Profile Android device system metrics over time")
    parser.add_argument("--duration", type=float, default=30, help="How long to profile (seconds)")
    parser.add_argument("--interval", type=float, default=0.5, help="Polling interval (seconds)")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--snapshot", action="store_true", help="Take one snapshot and exit")
    args = parser.parse_args()

    if args.snapshot:
        data = snapshot()
        print(json.dumps(data, indent=2))
        return 0

    profiler = SystemProfiler(interval=args.interval)
    print(f"Profiling for {args.duration}s (interval={args.interval}s) ...")
    profiler.start()
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\nStopped early.")
    profiler.stop()

    data = profiler.get_data()
    summary = profiler.get_summary()
    print(f"\nCollected {len(data)} samples over {summary.get('duration_s', '?')}s")
    print("\nSummary:")
    for k, v in summary.items():
        if v is not None:
            print(f"  {k}: {v}")

    # Flag thermal throttling
    throttle_keys = [k for k in summary if k.endswith("_throttle_pct")]
    if throttle_keys:
        print("\n  THERMAL THROTTLING DETECTED:")
        for k in throttle_keys:
            core = k.replace("_throttle_pct", "")
            print(f"    {core}: freq dropped {summary[k]}% during run")

    if args.output:
        out = args.output
    else:
        results_dir = SCRIPT_DIR / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        out = results_dir / "system_profile.json"
    with open(out, "w") as f:
        json.dump({"summary": summary, "samples": data}, f, indent=2)
    print(f"\nSaved to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
