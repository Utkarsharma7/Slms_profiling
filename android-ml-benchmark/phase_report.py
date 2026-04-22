"""
Turn the output of profile_llm.py into human-readable tables.

Reads results/phase_profile/all_variants.json (or an explicit path), then
emits:
  - A phase-breakdown table (ms per phase per variant, median across repeats).
  - A top-N operator breakdown for prefill and decode per variant.
  - A per-sublayer breakdown (q_proj / o_proj / ffn_down / ...) per variant.
  - A quantization sensitivity summary vs the F16 baseline (if present).

Output goes to stdout as markdown, and optionally to CSVs.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

DEFAULT_INPUT = Path(__file__).resolve().parent / "results" / "phase_profile" / "all_variants.json"

# Canonical phase ordering in the output tables.
PHASE_ORDER = [
    "model_load",
    "context_init",
    "tokenize",
    "prefill",
    "decode",
    "sample",
    "detokenize",
    "teardown",
    "total",
]


def _median(agg_leaf):
    if isinstance(agg_leaf, dict):
        return agg_leaf.get("median")
    if isinstance(agg_leaf, (int, float)):
        return agg_leaf
    return None


def _fmt(val, decimals=2):
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def _variants_iter(doc):
    variants = doc.get("variants") or {}
    # Keep a natural ordering: F16 > Q8_0 > Q6_K > Q5_K_M > Q4_K_M > Q3_K_L > Q2_K
    order = ["F16", "F32", "BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_L", "Q3_K_M", "Q2_K"]
    known = [q for q in order if q in variants]
    extra = [q for q in variants.keys() if q not in order]
    for q in known + extra:
        yield q, variants[q]


def _agg_for(block):
    # Either {"aggregate": {...}, "runs": [...]} or a legacy/skipped block.
    return block.get("aggregate") or {}


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------

def print_phase_table(doc, out_fp=sys.stdout):
    cols = list(_variants_iter(doc))
    headers = ["Phase (ms, median)"] + [q for q, _ in cols]
    print("### Phase breakdown (median ms across repeats)\n", file=out_fp)
    print("| " + " | ".join(headers) + " |", file=out_fp)
    print("|" + "|".join(["---"] * len(headers)) + "|", file=out_fp)
    for phase in PHASE_ORDER:
        row = [phase]
        for _, block in cols:
            if block.get("skipped_reason"):
                row.append("SKIP")
                continue
            ph = (_agg_for(block).get("phases_ms") or {}).get(phase)
            row.append(_fmt(_median(ph)))
        print("| " + " | ".join(row) + " |", file=out_fp)
    print("", file=out_fp)


def print_rates_and_memory(doc, out_fp=sys.stdout):
    cols = list(_variants_iter(doc))
    print("### Throughput and memory (median)\n", file=out_fp)
    headers = [
        "Metric",
        *[q for q, _ in cols],
    ]
    print("| " + " | ".join(headers) + " |", file=out_fp)
    print("|" + "|".join(["---"] * len(headers)) + "|", file=out_fp)

    def row(label, getter):
        cells = [label]
        for _, block in cols:
            if block.get("skipped_reason"):
                cells.append("SKIP")
                continue
            cells.append(_fmt(getter(_agg_for(block))))
        print("| " + " | ".join(cells) + " |", file=out_fp)

    row("prefill tok/s",  lambda a: _median((a.get("prefill")  or {}).get("tokens_per_sec")))
    row("decode tok/s",   lambda a: _median((a.get("decode")   or {}).get("tokens_per_sec")))
    row("first_token_ms", lambda a: _median((a.get("decode")   or {}).get("first_token_ms")))
    row("rss_after_load MB",    lambda a: _median((a.get("memory_mb") or {}).get("rss_after_load")))
    row("rss_peak MB",          lambda a: _median((a.get("memory_mb") or {}).get("rss_peak")))
    print("", file=out_fp)


def _top_n_map(map_obj: dict, n: int):
    if not map_obj:
        return []
    items = []
    for k, v in map_obj.items():
        m = _median(v)
        if m is None:
            continue
        items.append((k, m))
    items.sort(key=lambda kv: kv[1], reverse=True)
    return items[:n]


def print_per_op_table(doc, phase: str, top_n: int, out_fp=sys.stdout):
    cols = list(_variants_iter(doc))
    print(f"### Top {top_n} ops in `{phase}` (ms, median)\n", file=out_fp)
    headers = ["op"] + [q for q, _ in cols]
    print("| " + " | ".join(headers) + " |", file=out_fp)
    print("|" + "|".join(["---"] * len(headers)) + "|", file=out_fp)

    # Union of "top ops" across all variants so the row set is consistent.
    all_ops = set()
    per_variant_map = {}
    for q, block in cols:
        if block.get("skipped_reason"):
            per_variant_map[q] = {}
            continue
        m = ((_agg_for(block).get(phase) or {}).get("per_op_ms")) or {}
        per_variant_map[q] = m
        for k, _ in _top_n_map(m, top_n):
            all_ops.add(k)

    # Sort ops by max-across-variants median time, desc.
    def key_fn(op):
        return max(
            (_median(per_variant_map[q].get(op)) or 0.0) for q, _ in cols
        )
    for op in sorted(all_ops, key=key_fn, reverse=True):
        row = [op]
        for q, _ in cols:
            val = _median(per_variant_map[q].get(op))
            row.append(_fmt(val))
        print("| " + " | ".join(row) + " |", file=out_fp)
    print("", file=out_fp)


def print_per_sublayer_table(doc, phase: str, out_fp=sys.stdout):
    cols = list(_variants_iter(doc))
    print(f"### Per-sublayer breakdown in `{phase}` (ms, median)\n", file=out_fp)
    headers = ["sublayer"] + [q for q, _ in cols]
    print("| " + " | ".join(headers) + " |", file=out_fp)
    print("|" + "|".join(["---"] * len(headers)) + "|", file=out_fp)

    canon = ["q_proj", "k_proj", "v_proj", "qkv_proj", "o_proj",
             "ffn_gate", "ffn_up", "ffn_down", "rms_norm", "rope"]
    per_variant = {}
    for q, block in cols:
        if block.get("skipped_reason"):
            per_variant[q] = {}
            continue
        per_variant[q] = ((_agg_for(block).get(phase) or {}).get("per_sublayer_ms")) or {}

    present = set()
    for m in per_variant.values():
        present.update(m.keys())
    rows = [x for x in canon if x in present] + sorted(present - set(canon))
    for sub in rows:
        row = [sub]
        for q, _ in cols:
            row.append(_fmt(_median(per_variant[q].get(sub))))
        print("| " + " | ".join(row) + " |", file=out_fp)
    print("", file=out_fp)


def print_sensitivity(doc, out_fp=sys.stdout):
    """How much faster/smaller/lighter are the quants vs the F16 baseline?"""
    cols = dict(_variants_iter(doc))
    if "F16" not in cols:
        return
    base = _agg_for(cols["F16"])
    base_dec = (base.get("decode") or {})
    base_pf = (base.get("prefill") or {})
    base_mem = (base.get("memory_mb") or {})

    def ratio(a, b):
        if a is None or b in (None, 0):
            return None
        return a / b

    print("### Quantization sensitivity vs F16 baseline\n", file=out_fp)
    print("| Variant | decode tok/s (x F16) | prefill tok/s (x F16) | rss_peak (x F16) | total ms (x F16) |",
          file=out_fp)
    print("|---|---|---|---|---|", file=out_fp)
    base_tot = _median((base.get("phases_ms") or {}).get("total"))
    base_peak = _median(base_mem.get("rss_peak"))
    base_dec_tps = _median(base_dec.get("tokens_per_sec"))
    base_pf_tps = _median(base_pf.get("tokens_per_sec"))

    for q, block in cols.items():
        if q == "F16":
            continue
        if block.get("skipped_reason"):
            print(f"| {q} | SKIP | SKIP | SKIP | SKIP |", file=out_fp)
            continue
        a = _agg_for(block)
        dec_tps = _median((a.get("decode") or {}).get("tokens_per_sec"))
        pf_tps  = _median((a.get("prefill") or {}).get("tokens_per_sec"))
        tot     = _median((a.get("phases_ms") or {}).get("total"))
        peak    = _median((a.get("memory_mb") or {}).get("rss_peak"))
        print(
            f"| {q} | "
            f"{_fmt(ratio(dec_tps, base_dec_tps), 3)} | "
            f"{_fmt(ratio(pf_tps, base_pf_tps), 3)} | "
            f"{_fmt(ratio(peak, base_peak), 3)} | "
            f"{_fmt(ratio(tot, base_tot), 3)} |",
            file=out_fp,
        )
    print("", file=out_fp)


# ----------------------------------------------------------------------
# CSV exports
# ----------------------------------------------------------------------

def write_csv(doc, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = list(_variants_iter(doc))
    variant_names = [q for q, _ in cols]

    # Phases CSV
    with open(out_dir / "phases.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["phase"] + variant_names)
        for phase in PHASE_ORDER:
            row = [phase]
            for _, block in cols:
                a = _agg_for(block)
                ph = (a.get("phases_ms") or {}).get(phase)
                row.append(_median(ph) if not block.get("skipped_reason") else "")
            w.writerow(row)

    # Per-op CSVs (prefill + decode).
    for phase in ("prefill", "decode"):
        with open(out_dir / f"per_op_{phase}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            all_ops = set()
            per_variant = {}
            for q, block in cols:
                m = ((_agg_for(block).get(phase) or {}).get("per_op_ms")) or {}
                per_variant[q] = m
                all_ops.update(m.keys())
            w.writerow(["op"] + variant_names)
            for op in sorted(all_ops):
                row = [op]
                for q in variant_names:
                    row.append(_median(per_variant[q].get(op)))
                w.writerow(row)

    # Per-sublayer CSVs.
    for phase in ("prefill", "decode"):
        with open(out_dir / f"per_sublayer_{phase}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            all_keys = set()
            per_variant = {}
            for q, block in cols:
                m = ((_agg_for(block).get(phase) or {}).get("per_sublayer_ms")) or {}
                per_variant[q] = m
                all_keys.update(m.keys())
            w.writerow(["sublayer"] + variant_names)
            for k in sorted(all_keys):
                row = [k]
                for q in variant_names:
                    row.append(_median(per_variant[q].get(k)))
                w.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Cross-variant phase-profile report.")
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT,
                        help=f"Path to all_variants.json (default: {DEFAULT_INPUT})")
    parser.add_argument("--top-n", type=int, default=10,
                        help="How many top ops per phase (default: 10)")
    parser.add_argument("--csv-dir", type=Path, default=None,
                        help="Also write CSVs to this directory")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write markdown to this file instead of stdout")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Error: {args.input} not found. Run profile_llm.py first.", file=sys.stderr)
        return 1

    with open(args.input, "r", encoding="utf-8") as f:
        doc = json.load(f)

    out_fp = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        meta = doc.get("meta") or {}
        print(f"# LLM phase profile report\n", file=out_fp)
        if meta:
            dev = meta.get("device") or {}
            print(
                f"- Prompt: `{meta.get('prompt_file')}`  "
                f"n_gen={meta.get('n_gen')}  n_ctx={meta.get('n_ctx')}  "
                f"threads={meta.get('threads')}  repeats={meta.get('repeats')}",
                file=out_fp,
            )
            if dev:
                print(
                    f"- Device: {dev.get('brand')} {dev.get('model')} "
                    f"(SoC {dev.get('soc')})\n",
                    file=out_fp,
                )

        print_phase_table(doc, out_fp)
        print_rates_and_memory(doc, out_fp)
        print_per_op_table(doc, "prefill", args.top_n, out_fp)
        print_per_op_table(doc, "decode", args.top_n, out_fp)
        print_per_sublayer_table(doc, "prefill", out_fp)
        print_per_sublayer_table(doc, "decode", out_fp)
        print_sensitivity(doc, out_fp)

        if args.csv_dir:
            write_csv(doc, args.csv_dir)
            print(f"\nCSVs written to {args.csv_dir}", file=out_fp)
    finally:
        if args.output:
            out_fp.close()
            print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
