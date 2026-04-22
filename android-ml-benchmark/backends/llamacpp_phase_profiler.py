"""
Per-phase LLM profiler backend.

Thin wrapper around the custom C++ binary `llama-phase-profiler` (built from
../profiler/). Pushes the binary, prompt, and GGUF model to the device,
runs the profiler, parses the structured JSON it writes to stdout, and
returns it as a result dict.

Output schema is documented in profiler/README.md and the top-level plan.
"""
import json
import re
from pathlib import Path
from typing import Optional

from backends.base import BackendBase, DEVICE_TMP

SCRIPT_DIR = Path(__file__).resolve().parent.parent
BINARY_NAME = "llama-phase-profiler"
DEVICE_BINARY = f"{DEVICE_TMP}/{BINARY_NAME}"
DEVICE_PROMPT = f"{DEVICE_TMP}/phase_profiler_prompt.txt"


# Best-effort quant label from a GGUF filename, e.g.
#   "Llama-3.2-1B-Instruct-Q4_K_M.gguf" -> "Q4_K_M"
#   "Llama-3.2-1B-Instruct-f16.gguf"    -> "F16"
_QUANT_RE = re.compile(r"(?i)(f16|f32|bf16|q\d_[a-z0-9_]+|q\d)(?=\.gguf$|$)")


def infer_quant_from_filename(name: str) -> Optional[str]:
    m = _QUANT_RE.search(name)
    if not m:
        return None
    q = m.group(1).upper()
    # normalize a few common aliases
    if q == "F16":
        return "F16"
    return q


class LlamaCppPhaseProfilerBackend(BackendBase):
    """Phase + op-level profiler for GGUF models."""

    def __init__(
        self,
        num_threads: int = 4,
        wait_seconds: float = 45,
        prompt_file: Optional[Path] = None,
        n_gen: int = 64,
        n_ctx: int = 2048,
        seed: int = 42,
        dump_raw_ops: bool = False,
    ):
        super().__init__(num_threads=num_threads, wait_seconds=wait_seconds)
        self.prompt_file = Path(prompt_file) if prompt_file else None
        self.n_gen = int(n_gen)
        self.n_ctx = int(n_ctx)
        self.seed = int(seed)
        self.dump_raw_ops = bool(dump_raw_ops)

    def name(self) -> str:
        return "llama.cpp phase profiler"

    def _local_binary(self) -> Path:
        return SCRIPT_DIR / "binaries" / BINARY_NAME

    def is_available(self) -> bool:
        if not self._local_binary().is_file():
            return False
        if self.prompt_file and not self.prompt_file.is_file():
            return False
        return True

    def run(self, model_path: Path) -> dict:
        result = self._empty_result(model_path)
        result["format"] = ".gguf"
        result["quant"] = infer_quant_from_filename(model_path.name)
        result["n_gen_requested"] = self.n_gen
        result["n_ctx"] = self.n_ctx

        if not self._local_binary().is_file():
            result["error"] = (
                f"{BINARY_NAME} not found at {self._local_binary()}. "
                "Build it via profiler/build_android.sh (see profiler/README.md)."
            )
            return result

        if self.prompt_file is None or not self.prompt_file.is_file():
            result["error"] = (
                "No prompt file provided or file missing. Pass prompt_file=... "
                "(e.g. android-ml-benchmark/prompts/fixed_128.txt)."
            )
            return result

        device_model = f"{DEVICE_TMP}/{model_path.name}"

        # Fresh slate so stale files don't cause confusion.
        self.rm(DEVICE_BINARY)
        self.rm(DEVICE_PROMPT)
        self.rm(device_model)

        self.push(self._local_binary(), DEVICE_BINARY)
        self.shell_cmd(f"chmod 755 {DEVICE_BINARY}", timeout=10)

        self.push(self.prompt_file, DEVICE_PROMPT)
        self.push(model_path, device_model)

        cmd_parts = [
            DEVICE_BINARY,
            f"-m {device_model}",
            f"--prompt-file {DEVICE_PROMPT}",
            f"--n-gen {self.n_gen}",
            f"-t {self.num_threads}",
            f"--seed {self.seed}",
            f"--n-ctx {self.n_ctx}",
        ]
        if self.dump_raw_ops:
            cmd_parts.append("--dump-raw-ops")
        cmd = " ".join(cmd_parts)

        # Scale adb-shell timeout by model size; big F16 models can take a while.
        try:
            size_mb = model_path.stat().st_size / (1024 * 1024)
        except OSError:
            size_mb = 0
        extra = int(min(900, size_mb / 2))
        timeout_s = int(self.wait_seconds + 240 + extra)

        output = self.shell_cmd(cmd, timeout=timeout_s)

        # Clean up device side before we parse (keeps /data/local/tmp tidy).
        self.rm(device_model)
        self.rm(DEVICE_PROMPT)
        self.rm(DEVICE_BINARY)

        parsed = _extract_json(output)
        if parsed is None:
            result["error"] = "Could not parse JSON output from llama-phase-profiler"
            result["raw_output"] = output[:2000]
            return result

        result.update(parsed)
        # Mirror a few fields onto the existing BackendBase metric keys so the
        # standard summary table still shows something useful.
        phases = parsed.get("phases_ms") or {}
        decode = parsed.get("decode") or {}
        if decode.get("tokens_per_sec") is not None:
            result["generation_tokens_per_sec"] = round(float(decode["tokens_per_sec"]), 3)
        prefill = parsed.get("prefill") or {}
        if prefill.get("tokens_per_sec") is not None:
            result["prompt_tokens_per_sec"] = round(float(prefill["tokens_per_sec"]), 3)
        if phases.get("model_load") is not None:
            result["init_ms"] = round(float(phases["model_load"]), 3)
        if decode.get("first_token_ms") is not None:
            result["first_inference_us"] = round(float(decode["first_token_ms"]) * 1000.0, 1)
        mem = parsed.get("memory_mb") or {}
        if mem.get("rss_peak") is not None:
            result["memory_overall_mb"] = round(float(mem["rss_peak"]), 2)
        if mem.get("rss_after_load") is not None:
            result["memory_init_mb"] = round(float(mem["rss_after_load"]), 2)
        return result


def _extract_json(output: str) -> Optional[dict]:
    """
    The profiler prints one JSON line to stdout on success. adb shell may
    interleave logs / CRLF, so we scan for the first '{' and the last '}'
    and try to parse that substring.
    """
    if not output:
        return None
    start = output.find("{")
    end = output.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    blob = output[start : end + 1]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return None
