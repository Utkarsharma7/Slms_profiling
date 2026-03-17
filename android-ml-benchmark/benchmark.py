"""
Universal Android ML Benchmark.
Drop .tflite, .onnx, .pt, .ptl, .gguf models into models/ and run:
    python benchmark.py --all
Automatically picks the right backend per model format.
Supports delegates (cpu/gpu/nnapi) for TFLite and ONNX, per-operator profiling,
system profiling, and device auto-detection.
Works on any Android device connected via USB.
"""
import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from adb_interface import get_connected_devices
from backends import get_backend_for_model, BACKENDS
from backends.tflite_apk import TFLiteAPKBackend, DELEGATES as TFLITE_DELEGATES
from backends.onnx_binary import ONNXRuntimeBackend, ONNX_EXECUTION_PROVIDERS
from backends.pytorch_binary import PyTorchMobileBackend
from backends.llamacpp_binary import LlamaCppBackend
from system_profiler import SystemProfiler
from device_info import get_device_info, get_recommended_settings, print_device_info

MODELS_DIR = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results"
SUPPORTED_EXTENSIONS = tuple(BACKENDS.keys())


def scan_models(models_dir: Path) -> list[Path]:
    if not models_dir.is_dir():
        return []
    return sorted(
        p for p in models_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def print_result(r: dict):
    w = 60
    tag = f"[{r.get('backend', '?')}]"
    print("\n" + "=" * w)
    print(f"  {r.get('model', '?')}  {tag}")
    print("=" * w)
    if r.get("error"):
        print(f"  ERROR: {r['error']}")
    if r.get("inference_avg_ms") is not None:
        print(f"  Inference avg (ms):       {r['inference_avg_ms']}")
    if r.get("inference_avg_us") is not None:
        print(f"  Inference avg (us):       {r['inference_avg_us']}")
    if r.get("init_ms") is not None:
        print(f"  Init (ms):                {r['init_ms']}")
    if r.get("first_inference_us") is not None:
        print(f"  First inference (us):     {r['first_inference_us']}")
    if r.get("warmup_avg_us") is not None:
        print(f"  Warmup avg (us):          {r['warmup_avg_us']}")
    for pct in ("50", "90", "95", "99"):
        key = f"p{pct}_ms"
        if r.get(key) is not None:
            print(f"  P{pct} latency (ms):        {r[key]}")
    if r.get("memory_overall_mb") is not None:
        print(f"  Memory overall (MB):      {r['memory_overall_mb']}")
    if r.get("memory_init_mb") is not None:
        print(f"  Memory init (MB):         {r['memory_init_mb']}")
    if r.get("prompt_tokens_per_sec") is not None:
        print(f"  Prompt (tokens/s):        {r['prompt_tokens_per_sec']}")
    if r.get("generation_tokens_per_sec") is not None:
        print(f"  Generation (tokens/s):    {r['generation_tokens_per_sec']}")
    if r.get("model_size_gib") is not None:
        print(f"  Model size (GiB):         {r['model_size_gib']}")
    if r.get("throughput_per_sec") is not None:
        print(f"  Throughput (inf/s):       {r['throughput_per_sec']}")
    if r.get("top_ops"):
        print("  Top operators:")
        for op in r["top_ops"][:5]:
            print(f"    {op['op']:<30} {op['time']:>8} {op['pct']:>6.1f}%")
    if r.get("system_profile"):
        sp = r["system_profile"]
        print("  System profile:")
        if sp.get("battery_temp_c_max"):
            print(f"    Battery temp (max):     {sp['battery_temp_c_max']}C")
        for k, v in sp.items():
            if k.startswith("cpu") and k.endswith("_mhz_avg") and v is not None:
                print(f"    {k}:  {v}")
            if k.endswith("_throttle_pct"):
                print(f"    THROTTLING: {k.replace('_throttle_pct','')} dropped {v}%")
    print("=" * w)


def print_summary_table(results: list[dict]):
    print("\n" + "=" * 95)
    print("  SUMMARY")
    print("=" * 95)
    header = f"  {'Model':<30} {'Backend':<20} {'Delegate':<8} {'Avg (ms)':>10} {'Mem (MB)':>10} {'tok/s':>8}"
    print(header)
    print("  " + "-" * 90)
    for r in results:
        model = (r.get("model") or "?")[:29]
        backend = (r.get("backend") or "?")[:19]
        delegate = (r.get("delegate") or "")[:7]
        avg = r.get("inference_avg_ms")
        avg_str = f"{avg:.3f}" if avg is not None else ("ERR" if r.get("error") else "-")
        mem = r.get("memory_overall_mb")
        mem_str = f"{mem:.2f}" if mem is not None else "-"
        tps = r.get("generation_tokens_per_sec") or r.get("prompt_tokens_per_sec")
        tps_str = f"{tps:.1f}" if tps is not None else "-"
        print(f"  {model:<30} {backend:<20} {delegate:<8} {avg_str:>10} {mem_str:>10} {tps_str:>8}")
    print("=" * 95 + "\n")


def check_backends():
    """Print which backends are available."""
    tfl = TFLiteAPKBackend()
    onnx = ONNXRuntimeBackend()
    pt = PyTorchMobileBackend()
    lc = LlamaCppBackend()
    print("Backend availability:")
    print(f"  TFLite APK (.tflite):          {'YES' if tfl.is_available() else 'NO  (adb install -r -d benchmark.apk)'}")
    print(f"  llama.cpp (.gguf):             {'YES' if lc.is_available() else 'NO  (put llama-bench in binaries/)'}")
    print(f"  ONNX Runtime (.onnx):          {'YES' if onnx.is_available() else 'NO  (put onnxruntime_perf_test in binaries/)'}")
    print(f"  PyTorch Mobile (.pt, .ptl):    {'YES' if pt.is_available() else 'NO  (put speed_benchmark_torch in binaries/)'}")
    print()


def make_backend(backend_cls, args, delegate="cpu"):
    """Instantiate the right backend with the right constructor args."""
    if backend_cls == TFLiteAPKBackend:
        return backend_cls(
            num_threads=args.threads,
            wait_seconds=args.wait,
            delegate=delegate,
            op_profiling=not args.no_op_profiling,
        )
    elif backend_cls == ONNXRuntimeBackend:
        return backend_cls(
            num_threads=args.threads,
            wait_seconds=args.wait,
            execution_provider=delegate,
            op_profiling=not args.no_op_profiling,
        )
    elif backend_cls == PyTorchMobileBackend:
        return backend_cls(
            num_threads=args.threads,
            wait_seconds=args.wait,
            input_dims=args.input_dims,
            input_type=args.input_type,
        )
    else:
        return backend_cls(
            num_threads=args.threads,
            wait_seconds=args.wait,
        )


def get_delegates_for_backend(backend_cls, delegate_arg: str) -> list[str]:
    """Return list of delegates to run for a given backend class."""
    if backend_cls == TFLiteAPKBackend:
        if delegate_arg == "all":
            return list(TFLITE_DELEGATES)
        return [delegate_arg]
    elif backend_cls == ONNXRuntimeBackend:
        if delegate_arg == "all":
            return list(ONNX_EXECUTION_PROVIDERS)
        return [delegate_arg]
    return ["cpu"]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ML models (.tflite, .onnx, .pt, .ptl, .gguf) on any Android device over USB.",
    )
    parser.add_argument("model", type=Path, nargs="?", default=None,
                        help="Path to one model. Omit to use --all or first model in models/")
    parser.add_argument("--all", action="store_true",
                        help="Benchmark every supported model in models/")
    parser.add_argument("--threads", type=int, default=None,
                        help="CPU threads (default: auto-detected from device)")
    parser.add_argument("--wait", type=float, default=None,
                        help="Seconds to wait per model (default: auto-detected from device)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output JSON path (default: results/benchmark_results.json)")
    parser.add_argument("--check", action="store_true",
                        help="Just check which backends are available and exit")
    # Delegate control (works for TFLite and ONNX)
    parser.add_argument("--delegate", type=str, default="cpu",
                        choices=["cpu", "gpu", "nnapi", "xnnpack", "all"],
                        help="Delegate/EP: cpu, gpu, nnapi, xnnpack, or all (runs all per-backend)")
    parser.add_argument("--no-op-profiling", action="store_true",
                        help="Disable per-operator profiling for TFLite/ONNX")
    # PyTorch-specific
    parser.add_argument("--input-dims", type=str, default="1,3,224,224",
                        help="Input dimensions for PyTorch models (default: 1,3,224,224)")
    parser.add_argument("--input-type", type=str, default="float",
                        help="Input type for PyTorch models (default: float)")
    # System profiling
    parser.add_argument("--profile", action="store_true",
                        help="Collect system metrics (CPU freq, temp, memory) during each benchmark")
    parser.add_argument("--profile-interval", type=float, default=0.5,
                        help="System profiling poll interval in seconds (default: 0.5)")
    # Device info
    parser.add_argument("--device-info", action="store_true",
                        help="Print device info and recommended settings, then exit")
    parser.add_argument("--no-device-info", action="store_true",
                        help="Skip device info banner at startup")
    args = parser.parse_args()

    if not args.check and not args.device_info and not get_connected_devices():
        print("Error: No Android device connected.", file=sys.stderr)
        sys.exit(1)

    if args.check:
        check_backends()
        return 0

    # Detect device and auto-tune settings
    dev_info = None
    rec = None
    if get_connected_devices():
        try:
            dev_info = get_device_info()
            rec = get_recommended_settings(dev_info)
        except Exception:
            pass

    if args.device_info:
        if dev_info:
            print_device_info(dev_info, rec)
        else:
            print("Could not detect device info.", file=sys.stderr)
        return 0

    # Apply auto-detected defaults if user didn't override
    if args.threads is None:
        args.threads = rec["threads"] if rec else 4
    if args.wait is None:
        args.wait = rec["wait_seconds"] if rec else 45

    if not args.no_device_info and dev_info:
        print_device_info(dev_info, rec)

    if args.all:
        model_paths = scan_models(MODELS_DIR)
        if not model_paths:
            print(f"Error: No supported models in {MODELS_DIR}", file=sys.stderr)
            sys.exit(1)
    elif args.model:
        model_paths = [args.model.resolve()]
    else:
        model_paths = scan_models(MODELS_DIR)
        if model_paths:
            model_paths = [model_paths[0]]
        else:
            print("Error: No model specified and models/ is empty.", file=sys.stderr)
            sys.exit(1)

    check_backends()

    all_results = []
    step = 0
    total = len(model_paths)
    for i, mp in enumerate(model_paths):
        if not mp.is_file():
            print(f"[{i+1}/{total}] SKIP: {mp} (not found)")
            continue

        try:
            backend_cls = get_backend_for_model(mp)
        except ValueError as e:
            print(f"[{i+1}/{total}] SKIP: {mp.name} ({e})")
            continue

        delegates_to_run = get_delegates_for_backend(backend_cls, args.delegate)

        for delegate in delegates_to_run:
            step += 1
            backend = make_backend(backend_cls, args, delegate=delegate)
            has_delegate = backend_cls in (TFLiteAPKBackend, ONNXRuntimeBackend)
            label = f"{mp.name}" + (f" ({delegate.upper()})" if has_delegate else "")
            print(f"\n[{step}] {label}  ->  {backend.name()}")

            if not backend.is_available():
                result = backend._empty_result(mp)
                result["error"] = f"Backend '{backend.name()}' not available. See --check."
                all_results.append(result)
                print_result(result)
                continue

            profiler = None
            if args.profile:
                profiler = SystemProfiler(interval=args.profile_interval)
                profiler.start()

            result = backend.run(mp)

            if profiler:
                profiler.stop()
                result["system_profile"] = profiler.get_summary()

            # Attach device info to each result for report.py
            if dev_info:
                result["device"] = f"{dev_info.get('brand','')} {dev_info.get('model','')}"
                result["soc"] = dev_info.get("soc", "")

            all_results.append(result)
            print_result(result)

    print_summary_table(all_results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output or RESULTS_DIR / "benchmark_results.json"

    # Include device info at top level of the output
    output_data = {
        "device_info": dev_info,
        "recommendations": rec,
        "results": all_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"Results saved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
