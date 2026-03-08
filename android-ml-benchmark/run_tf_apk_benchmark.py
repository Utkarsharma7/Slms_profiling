"""
Run the TensorFlow Lite benchmark APK on your phone and show results on your PC.
Uses logcat to capture the benchmark output and prints a summary + saves to results/.

How inference is measured (no real images): The benchmark app does not use real images.
It feeds the model dummy input (correct shape, e.g. [1,224,224,3] for MobileNet) and
measures how long one forward pass takes. So you get timing and memory, not classification
results. Any .tflite model can be benchmarked this way.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

# Add project root so we can use adb_interface
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from adb_interface import run_adb, get_device, get_connected_devices, NoDeviceError, ADBError


TF_PACKAGE = "org.tensorflow.lite.benchmark"
TF_ACTIVITY = "org.tensorflow.lite.benchmark/.BenchmarkModelActivity"
DEVICE_TMP = "/data/local/tmp"
RESULTS_DIR = SCRIPT_DIR / "results"


def run_adb_safe(*args, timeout=120):
    """Run ADB without raising; return (stdout, stderr, returncode)."""
    try:
        r = run_adb(*args, timeout=timeout, check=False)
        return (r.stdout or "", r.stderr or "", r.returncode)
    except ADBError:
        return ("", "", 1)


def clear_logcat():
    run_adb("logcat", "-c", check=False)


def get_logcat(tag="tflite"):
    out, _, _ = run_adb_safe("logcat", "-d", "-s", f"{tag}:I", timeout=15)
    return out or ""


def push_model(local_path: Path, device_name: str = "model.tflite") -> str:
    """Push model to device; returns device path."""
    device_path = f"{DEVICE_TMP}/{device_name}"
    run_adb("push", str(local_path), device_path, timeout=60)
    return device_path


def launch_benchmark(device_model_path: str, num_threads: int = 4):
    """Start the TF benchmark app with the given model path."""
    args_str = f"--graph={device_model_path} --num_threads={num_threads}"
    run_adb(
        "shell", "am", "start", "-n", TF_ACTIVITY,
        "--es", "args", args_str,
        timeout=15,
    )


def parse_logcat_output(log: str) -> dict:
    """Extract benchmark metrics from tflite logcat lines."""
    result = {
        "init_ms": None,
        "first_inference_us": None,
        "warmup_avg_us": None,
        "inference_avg_us": None,
        "inference_avg_ms": None,
        "memory_init_mb": None,
        "memory_overall_mb": None,
        "raw_inference_line": None,
        "raw_memory_line": None,
    }
    # Inference timings in us: Init: 44497, First inference: 39744, Warmup (avg): 24724.1, Inference (avg): 23866.5
    m = re.search(
        r"Inference timings in us:\s*Init:\s*([\d.]+),\s*First inference:\s*([\d.]+),\s*Warmup \(avg\):\s*([\d.]+),\s*Inference \(avg\):\s*([\d.]+)",
        log,
    )
    if m:
        result["init_ms"] = round(float(m.group(1)) / 1000, 2)
        result["first_inference_us"] = float(m.group(2))
        result["warmup_avg_us"] = float(m.group(3))
        result["inference_avg_us"] = float(m.group(4))
        result["inference_avg_ms"] = round(float(m.group(4)) / 1000, 3)
        result["raw_inference_line"] = m.group(0)
    # Memory footprint delta from the start of the tool (MB): init=4.85156 overall=5.10938
    m2 = re.search(
        r"Memory footprint delta.*?init=([\d.]+)\s+overall=([\d.]+)",
        log,
        re.DOTALL,
    )
    if m2:
        result["memory_init_mb"] = round(float(m2.group(1)), 2)
        result["memory_overall_mb"] = round(float(m2.group(2)), 2)
        result["raw_memory_line"] = m2.group(0)
    return result


def print_results(model_name: str, parsed: dict):
    """Print a simple table to the console."""
    print("\n" + "=" * 50)
    print(f"  Benchmark: {model_name}")
    print("=" * 50)
    if parsed.get("inference_avg_ms") is not None:
        print(f"  Init (ms):           {parsed['init_ms']}")
        print(f"  First inference (µs): {parsed['first_inference_us']}")
        print(f"  Warmup avg (µs):      {parsed['warmup_avg_us']}")
        print(f"  Inference avg (µs):   {parsed['inference_avg_us']}")
        print(f"  Inference avg (ms):   {parsed['inference_avg_ms']}")
    if parsed.get("memory_overall_mb") is not None:
        print(f"  Memory init (MB):     {parsed['memory_init_mb']}")
        print(f"  Memory overall (MB):  {parsed['memory_overall_mb']}")
    if parsed.get("inference_avg_ms") is None and parsed.get("raw_inference_line") is None:
        print("  (No inference timings found in logcat. Wait longer or check app.)")
    print("=" * 50 + "\n")


def run_one_benchmark(model_path: Path, args) -> dict:
    """Run benchmark for one model; return parsed result dict."""
    clear_logcat()
    print(f"  Pushing: {model_path.name}")
    device_path = push_model(model_path, "model.tflite")
    launch_benchmark(device_path, num_threads=args.threads)
    time.sleep(args.wait)
    log = get_logcat()
    parsed = parse_logcat_output(log)
    parsed["model"] = model_path.name
    parsed["device_model_path"] = device_path
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Run TF benchmark APK and show results on PC")
    parser.add_argument("model", type=Path, nargs="?", default=None,
                        help="Path to one .tflite model. If omitted and --all is not set, uses first .tflite in models/")
    parser.add_argument("--all", action="store_true", help="Run benchmark for every .tflite in models/ and save combined results")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads (default: 4)")
    parser.add_argument("--wait", type=float, default=45, help="Seconds to wait before reading logcat (default: 45)")
    parser.add_argument("--output", "-o", type=Path, help="Save JSON here (default: results/tf_apk_benchmark_results.json)")
    args = parser.parse_args()

    if not get_connected_devices():
        print("Error: No Android device connected. Connect via USB and enable USB debugging.", file=sys.stderr)
        sys.exit(1)

    models_dir = (SCRIPT_DIR / "models").resolve()
    if args.all:
        model_paths = sorted(models_dir.glob("*.tflite")) if models_dir.is_dir() else []
        if not model_paths:
            print(f"Error: No .tflite files in {models_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Benchmarking {len(model_paths)} model(s) in models/")
        all_results = []
        for i, p in enumerate(model_paths):
            print(f"[{i+1}/{len(model_paths)}] {p.name}")
            parsed = run_one_benchmark(p, args)
            all_results.append(parsed)
            print_results(p.name, parsed)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = args.output or RESULTS_DIR / "tf_apk_benchmark_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"All results saved to: {out_path}")
        return 0

    if args.model is not None:
        model_path = args.model.resolve()
    else:
        if not models_dir.is_dir():
            print("Error: No models/ folder or no model path given.", file=sys.stderr)
            sys.exit(1)
        first = next((p for p in sorted(models_dir.glob("*.tflite"))), None)
        if not first:
            print("Error: No .tflite files in models/. Put at least one or pass model path.", file=sys.stderr)
            sys.exit(1)
        model_path = first

    if not model_path.is_file():
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    print("Clearing logcat...")
    print(f"Pushing model to device: {model_path.name}")
    print("Starting benchmark on phone...")
    parsed = run_one_benchmark(model_path, args)
    print_results(model_path.name, parsed)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output or RESULTS_DIR / "tf_apk_benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)
    print(f"Results saved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
