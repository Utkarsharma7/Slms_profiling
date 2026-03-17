"""
Unified report generator for benchmark results.
Reads JSON result files from results/ and produces:
  - Device info header
  - A formatted comparison table (terminal)
  - Per-delegate comparison for TFLite/ONNX models
  - Bottleneck highlights
  - CSV export for spreadsheets

Usage:
    python report.py                         # reads results/benchmark_results.json
    python report.py results/run1.json results/run2.json
    python report.py --csv results/report.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"


def load_results(paths: list[Path]) -> tuple[list[dict], dict | None]:
    """Load results from JSON files. Returns (results_list, device_info_or_None)."""
    all_results = []
    device_info = None
    for p in paths:
        if not p.exists():
            print(f"Warning: {p} not found, skipping", file=sys.stderr)
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        # New format: {"device_info": {...}, "results": [...]}
        if isinstance(data, dict) and "results" in data:
            all_results.extend(data["results"])
            if data.get("device_info") and not device_info:
                device_info = data["device_info"]
        # Old format: bare list
        elif isinstance(data, list):
            all_results.extend(data)
        elif isinstance(data, dict):
            all_results.append(data)
    return all_results, device_info


def print_device_header(info: dict):
    """Print device info at top of report."""
    if not info:
        return
    w = 120
    print(f"\n{'=' * w}")
    print("  DEVICE")
    print(f"{'=' * w}")
    parts = []
    if info.get("brand"):
        parts.append(f"{info['brand']} {info.get('model', '')}")
    if info.get("soc"):
        parts.append(f"SoC: {info['soc']} {info.get('soc_model', '')}")
    if info.get("android_version"):
        parts.append(f"Android {info['android_version']}")
    if info.get("ram_total_mb"):
        parts.append(f"RAM: {info['ram_total_mb']} MB")
    if info.get("cpu_cores"):
        parts.append(f"CPU: {info['cpu_cores']} cores")
    if info.get("gpu_renderer"):
        parts.append(f"GPU: {info['gpu_renderer']}")
    print(f"  {' | '.join(parts)}")
    print(f"{'=' * w}")


def print_table(results: list[dict]):
    """Print a rich comparison table to the terminal."""
    if not results:
        print("No results to display.")
        return

    sep = "=" * 120
    print(f"\n{sep}")
    print("  BENCHMARK COMPARISON REPORT")
    print(sep)

    header = (
        f"  {'Model':<30} {'Format':<7} {'Backend':<22} {'Dlg':<8} "
        f"{'Avg(ms)':>10} {'Mem(MB)':>8} {'pp tok/s':>9} {'tg tok/s':>9} {'Init(ms)':>9}"
    )
    print(header)
    print("  " + "-" * 115)

    for r in results:
        model = (r.get("model") or "?")[:29]
        fmt = (r.get("format") or "?")[:6]
        backend = (r.get("backend") or "?")[:21]
        dlg = (r.get("delegate") or "")[:7]
        avg = r.get("inference_avg_ms")
        avg_s = f"{avg:.3f}" if avg is not None else ("ERR" if r.get("error") else "-")
        mem = r.get("memory_overall_mb")
        mem_s = f"{mem:.1f}" if mem is not None else "-"
        pp = r.get("prompt_tokens_per_sec")
        pp_s = f"{pp:.1f}" if pp is not None else "-"
        tg = r.get("generation_tokens_per_sec")
        tg_s = f"{tg:.1f}" if tg is not None else "-"
        init = r.get("init_ms")
        init_s = f"{init:.1f}" if init is not None else "-"
        print(f"  {model:<30} {fmt:<7} {backend:<22} {dlg:<8} {avg_s:>10} {mem_s:>8} {pp_s:>9} {tg_s:>9} {init_s:>9}")

    print(sep)


def print_delegate_comparison(results: list[dict]):
    """For models benchmarked with multiple delegates, show which is fastest."""
    from collections import defaultdict
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        if r.get("delegate") and r.get("inference_avg_ms") is not None:
            by_model[r["model"]].append(r)

    comparisons = {m: runs for m, runs in by_model.items() if len(runs) > 1}
    if not comparisons:
        return

    print("\n" + "=" * 80)
    print("  DELEGATE / EXECUTION PROVIDER COMPARISON")
    print("=" * 80)
    for model, runs in sorted(comparisons.items()):
        sorted_runs = sorted(runs, key=lambda x: x["inference_avg_ms"])
        fastest = sorted_runs[0]
        slowest = sorted_runs[-1]
        speedup = round(slowest["inference_avg_ms"] / fastest["inference_avg_ms"], 1) if fastest["inference_avg_ms"] > 0 else 0
        print(f"\n  {model}  (best: {fastest['delegate'].upper()}, {speedup}x faster than worst):")
        for run in sorted_runs:
            marker = " << FASTEST" if run is fastest else ""
            mem = run.get('memory_overall_mb')
            mem_str = f", mem: {mem} MB" if mem else ""
            print(f"    {run['delegate'].upper():>8}: {run['inference_avg_ms']:.3f} ms{mem_str}{marker}")
    print("=" * 80)


def print_bottleneck_analysis(results: list[dict], device_info: dict | None = None):
    """Highlight potential bottlenecks based on metrics and device capabilities."""
    issues = []
    ram = (device_info or {}).get("ram_total_mb", 8192)

    for r in results:
        model = r.get("model", "?")
        delegate = r.get("delegate", "cpu")
        if r.get("error"):
            issues.append(f"  FAIL  {model} [{delegate}]: {r['error'][:70]}")
            continue
        avg = r.get("inference_avg_ms")
        if avg is not None and avg > 1000:
            issues.append(f"  SLOW  {model} [{delegate}]: Inference > 1s ({avg:.0f}ms) - too slow for real-time")
        if avg is not None and avg > 100 and delegate in ("gpu", "nnapi"):
            issues.append(f"  WARN  {model} [{delegate}]: Still > 100ms with HW accel - limited delegate support?")
        mem = r.get("memory_overall_mb")
        if mem is not None and mem > ram * 0.6:
            issues.append(f"  MEM   {model}: Uses {mem:.0f}MB (>{60}% of device RAM)")
        tg = r.get("generation_tokens_per_sec")
        if tg is not None and tg < 5:
            issues.append(f"  SLOW  {model}: Generation < 5 tok/s ({tg:.1f}) - poor interactive experience")

        # Thermal throttling from system profile
        sp = r.get("system_profile", {})
        for k, v in sp.items():
            if k.endswith("_throttle_pct") and v > 15:
                core = k.replace("_throttle_pct", "")
                issues.append(f"  THERM {model}: {core} freq dropped {v}% (thermal throttling)")

    if issues:
        print("\n" + "=" * 90)
        print("  BOTTLENECK ALERTS")
        print("=" * 90)
        for issue in issues:
            print(issue)
        print("=" * 90)
    else:
        print("\n  No bottleneck alerts.")


def export_csv(results: list[dict], path: Path):
    """Export results to CSV."""
    if not results:
        print("No results to export.")
        return
    # Collect all keys, excluding nested dicts
    skip_keys = {"op_profile", "top_ops", "raw_output", "system_profile"}
    all_keys = []
    for r in results:
        for k in r:
            if k not in all_keys and k not in skip_keys:
                all_keys.append(k)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            clean = {k: v for k, v in r.items() if k not in skip_keys}
            writer.writerow(clean)
    print(f"CSV exported to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark comparison reports")
    parser.add_argument("files", nargs="*", type=Path,
                        help="JSON result files (default: results/benchmark_results.json)")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Export CSV to this path")
    parser.add_argument("--no-delegate-compare", action="store_true",
                        help="Skip delegate comparison section")
    parser.add_argument("--no-bottleneck", action="store_true",
                        help="Skip bottleneck analysis")
    args = parser.parse_args()

    if args.files:
        paths = args.files
    else:
        default = RESULTS_DIR / "benchmark_results.json"
        if default.exists():
            paths = [default]
        else:
            all_json = sorted(RESULTS_DIR.glob("*.json")) if RESULTS_DIR.is_dir() else []
            all_json = [j for j in all_json if "system_profile" not in j.name]
            if all_json:
                paths = all_json
                print(f"Loading {len(paths)} result files from {RESULTS_DIR}")
            else:
                print("No result files found. Run benchmark.py first.", file=sys.stderr)
                return 1

    results, device_info = load_results(paths)
    if not results:
        print("No benchmark results loaded.", file=sys.stderr)
        return 1

    print(f"Loaded {len(results)} benchmark result(s)")
    print_device_header(device_info)
    print_table(results)

    if not args.no_delegate_compare:
        print_delegate_comparison(results)
    if not args.no_bottleneck:
        print_bottleneck_analysis(results, device_info)

    csv_path = args.csv or RESULTS_DIR / "benchmark_report.csv"
    export_csv(results, csv_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
