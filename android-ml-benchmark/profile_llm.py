"""
LLM phase + operator profiler orchestrator.

Drops any number of GGUF variants (e.g. Llama-3.2-1B-Instruct-*.gguf) into
models/, then:

    python profile_llm.py --model-glob "Llama-3.2-1B-Instruct-*.gguf"

runs llama-phase-profiler on each variant `--repeats` times, aggregates
median / p5 / p95 of every metric across repeats, and writes:

    results/phase_profile/<quant>.json           # per-variant aggregate + raw runs
    results/phase_profile/all_variants.json      # combined cross-variant file

Pair with `phase_report.py` for a readable comparison.
"""
import argparse
import json
import statistics
import sys
import time
from fnmatch import fnmatch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from adb_interface import get_connected_devices
from backends.llamacpp_phase_profiler import (
    LlamaCppPhaseProfilerBackend,
    infer_quant_from_filename,
)
from system_profiler import SystemProfiler
from device_info import get_device_info, get_recommended_settings, print_device_info

MODELS_DIR = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results" / "phase_profile"
DEFAULT_PROMPT = SCRIPT_DIR / "prompts" / "fixed_128.txt"


# ----------------------------------------------------------------------
# Aggregation helpers
# ----------------------------------------------------------------------

def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    vs = sorted(values)
    if len(vs) == 1:
        return vs[0]
    k = (len(vs) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(vs) - 1)
    frac = k - lo
    return vs[lo] + (vs[hi] - vs[lo]) * frac


def _stats(values: list[float]) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"median": None, "p5": None, "p95": None, "min": None, "max": None, "n": 0}
    return {
        "median": statistics.median(values),
        "p5": _percentile(values, 5),
        "p95": _percentile(values, 95),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def _collect_leaf_paths(obj, prefix=""):
    """
    Walk a run JSON and yield (path, value) for every numeric leaf. Used to
    aggregate across repeats without hardcoding every key.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _collect_leaf_paths(v, prefix + ("." if prefix else "") + str(k))
    elif isinstance(obj, list):
        # We don't aggregate element-by-element for lists (their length can
        # vary e.g. per_step_ms); the orchestrator keeps these per-run.
        return
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix, float(obj)


def _set_by_path(d: dict, path: str, value):
    parts = path.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def aggregate_runs(runs: list[dict]) -> dict:
    """
    Build an aggregate dict with the same shape as a single run, where each
    numeric leaf is replaced by {median, p5, p95, min, max, n}. Per-step
    arrays (variable length) are kept as a list of lists.
    """
    if not runs:
        return {}
    # Collect values grouped by key-path.
    bucket: dict[str, list[float]] = {}
    for r in runs:
        for path, val in _collect_leaf_paths(r):
            bucket.setdefault(path, []).append(val)

    agg: dict = {}
    for path, values in bucket.items():
        _set_by_path(agg, path, _stats(values))

    # Preserve non-numeric meta and full per-step traces (useful for plots).
    meta0 = runs[0].get("meta", {}) or {}
    agg_meta = agg.get("meta", {}) or {}
    for k, v in meta0.items():
        if not isinstance(v, (int, float)):
            agg_meta[k] = v
    agg["meta"] = agg_meta

    agg["per_step_ms_runs"] = [
        (r.get("decode") or {}).get("per_step_ms") or [] for r in runs
    ]
    return agg


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------

def find_variants(models_dir: Path, pattern: str) -> list[Path]:
    if not models_dir.is_dir():
        return []
    return sorted(
        p for p in models_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".gguf" and fnmatch(p.name, pattern)
    )


def _ram_ok(dev_info: dict, model_path: Path, limit_pct: float = 70.0) -> tuple[bool, str]:
    try:
        size_mb = model_path.stat().st_size / (1024 * 1024)
    except OSError:
        size_mb = 0
    avail_mb = 0
    ram = dev_info.get("ram") or {}
    if isinstance(ram, dict):
        avail_mb = ram.get("available_mb") or ram.get("total_mb") or 0
    elif isinstance(ram, (int, float)):
        avail_mb = ram
    if not avail_mb:
        return True, ""
    cap = avail_mb * (limit_pct / 100.0)
    if size_mb > cap:
        return False, (
            f"model size {size_mb:.0f} MB exceeds {limit_pct:.0f}% of available RAM "
            f"{avail_mb:.0f} MB (cap {cap:.0f} MB)"
        )
    return True, ""


def _thermally_throttled(summary: dict) -> bool:
    """Heuristic: any CPU cluster dropped >20% during the run."""
    if not summary:
        return False
    for k, v in summary.items():
        if k.endswith("_throttle_pct") and isinstance(v, (int, float)) and v >= 20.0:
            return True
    return False


def run_variant(
    model_path: Path,
    *,
    threads: int,
    n_gen: int,
    n_ctx: int,
    seed: int,
    prompt_file: Path,
    repeats: int,
    cooldown: float,
    capture_system: bool,
    profile_interval: float,
    dump_raw_ops: bool,
    wait_seconds: float,
) -> dict:
    runs: list[dict] = []
    backend = LlamaCppPhaseProfilerBackend(
        num_threads=threads,
        wait_seconds=wait_seconds,
        prompt_file=prompt_file,
        n_gen=n_gen,
        n_ctx=n_ctx,
        seed=seed,
        dump_raw_ops=dump_raw_ops,
    )

    for i in range(repeats):
        print(f"    repeat {i+1}/{repeats}  -> running profiler...", flush=True)
        sys_prof = None
        sys_summary = None
        if capture_system:
            sys_prof = SystemProfiler(interval=profile_interval)
            sys_prof.start()
        try:
            result = backend.run(model_path)
        finally:
            if sys_prof is not None:
                sys_prof.stop()
                try:
                    sys_summary = sys_prof.get_summary()
                except Exception:
                    sys_summary = None

        result["repeat_idx"] = i
        if sys_summary is not None:
            result["system_profile"] = sys_summary
            result["thermally_throttled"] = _thermally_throttled(sys_summary)
        runs.append(result)

        dec = (result.get("decode") or {})
        ph = (result.get("phases_ms") or {})
        tps = dec.get("tokens_per_sec")
        print(
            "      "
            + f"prefill={ph.get('prefill')} ms, "
            + f"decode={ph.get('decode')} ms, "
            + f"gen tok/s={tps}"
        )

        if i < repeats - 1 and cooldown > 0:
            print(f"      cooldown {cooldown:.0f}s ...", flush=True)
            time.sleep(cooldown)

    agg = aggregate_runs(runs)
    return {"runs": runs, "aggregate": agg}


def main():
    parser = argparse.ArgumentParser(description="Per-phase + per-op LLM profiler orchestrator.")
    parser.add_argument("--model-glob", default="Llama-3.2-1B-Instruct-*.gguf",
                        help="Filename glob under models/ (default: Llama-3.2-1B-Instruct-*.gguf)")
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT,
                        help="Path to fixed prompt file (default: prompts/fixed_128.txt)")
    parser.add_argument("--n-gen", type=int, default=64,
                        help="Tokens to decode per run (default: 64)")
    parser.add_argument("--n-ctx", type=int, default=2048,
                        help="Context size (default: 2048)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed recorded in meta (default: 42)")
    parser.add_argument("--threads", type=int, default=None,
                        help="CPU threads (default: device recommendation)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Runs per variant (default: 3)")
    parser.add_argument("--cooldown", type=float, default=30.0,
                        help="Seconds to sleep between runs (default: 30)")
    parser.add_argument("--wait", type=float, default=None,
                        help="Base wait_seconds for backend (default: device recommendation)")
    parser.add_argument("--no-system-profile", action="store_true",
                        help="Disable parallel CPU-freq/temp sampling")
    parser.add_argument("--profile-interval", type=float, default=0.5,
                        help="System-profiler poll interval (default: 0.5s)")
    parser.add_argument("--dump-raw-ops", action="store_true",
                        help="Also dump every op event per run (large)")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR,
                        help="Where to write JSON outputs (default: results/phase_profile)")
    parser.add_argument("--ram-limit-pct", type=float, default=70.0,
                        help="Skip variants whose file size exceeds this %% of available RAM (default: 70)")
    parser.add_argument("--skip-ram-check", action="store_true",
                        help="Run every variant regardless of RAM headroom")
    args = parser.parse_args()

    if not get_connected_devices():
        print("Error: no Android device connected.", file=sys.stderr)
        return 1

    dev_info = None
    rec = None
    try:
        dev_info = get_device_info()
        rec = get_recommended_settings(dev_info)
    except Exception as e:
        print(f"Warning: could not read device info ({e}).", file=sys.stderr)

    if dev_info:
        print_device_info(dev_info, rec)

    if args.threads is None:
        args.threads = rec["threads"] if rec else 4
    if args.wait is None:
        args.wait = rec["wait_seconds"] if rec else 45

    variants = find_variants(MODELS_DIR, args.model_glob)
    if not variants:
        print(
            f"Error: no GGUFs matching '{args.model_glob}' in {MODELS_DIR}",
            file=sys.stderr,
        )
        return 1

    print(f"\nFound {len(variants)} variant(s) matching '{args.model_glob}':")
    for v in variants:
        print(f"  {v.name}  ({v.stat().st_size / (1024*1024):.0f} MB)")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_variants = {
        "meta": {
            "prompt_file": str(args.prompt_file),
            "n_gen": args.n_gen,
            "n_ctx": args.n_ctx,
            "seed": args.seed,
            "threads": args.threads,
            "repeats": args.repeats,
            "cooldown_s": args.cooldown,
            "device": {
                "soc": (dev_info or {}).get("soc"),
                "brand": (dev_info or {}).get("brand"),
                "model": (dev_info or {}).get("model"),
            } if dev_info else None,
        },
        "variants": {},
    }

    for idx, model_path in enumerate(variants, start=1):
        quant = infer_quant_from_filename(model_path.name) or model_path.stem
        print(f"\n[{idx}/{len(variants)}] {model_path.name}  (quant={quant})")

        if dev_info and not args.skip_ram_check:
            ok, why = _ram_ok(dev_info, model_path, args.ram_limit_pct)
            if not ok:
                print(f"    SKIP: {why}")
                all_variants["variants"][quant] = {
                    "model_file": model_path.name,
                    "skipped_reason": why,
                }
                _save(args.output_dir, quant, all_variants["variants"][quant])
                continue

        variant_block = run_variant(
            model_path,
            threads=args.threads,
            n_gen=args.n_gen,
            n_ctx=args.n_ctx,
            seed=args.seed,
            prompt_file=args.prompt_file,
            repeats=args.repeats,
            cooldown=args.cooldown,
            capture_system=not args.no_system_profile,
            profile_interval=args.profile_interval,
            dump_raw_ops=args.dump_raw_ops,
            wait_seconds=args.wait,
        )
        variant_block["model_file"] = model_path.name
        variant_block["quant"] = quant
        all_variants["variants"][quant] = variant_block
        _save(args.output_dir, quant, variant_block)

        # Persist the combined file incrementally so a crash mid-run still
        # leaves valid data on disk.
        with open(args.output_dir / "all_variants.json", "w", encoding="utf-8") as f:
            json.dump(all_variants, f, indent=2, default=str)

    print(f"\nDone. Outputs in: {args.output_dir}")
    return 0


def _save(out_dir: Path, quant: str, blob: dict):
    out_path = out_dir / f"{quant}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2, default=str)
    print(f"    wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
