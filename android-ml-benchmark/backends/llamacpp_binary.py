"""
llama.cpp benchmark via llama-bench or llama-cli binary on Android.
Model format: .gguf

Binary: llama-bench (arm64, built from llama.cpp source or via Termux)
Place it in: android-ml-benchmark/binaries/llama-bench

Usage on device:
  llama-bench -m model.gguf -t 4 -r 5
Output contains lines like:
  | model | size | params | backend | threads | test | t/s |
  | llama 1B | 0.6 GiB | 1.10 B | CPU | 4 | pp512 | 45.67 |
  | llama 1B | 0.6 GiB | 1.10 B | CPU | 4 | tg128 | 12.34 |
"""
import re
from pathlib import Path

from backends.base import BackendBase, DEVICE_TMP

SCRIPT_DIR = Path(__file__).resolve().parent.parent
BINARY_NAME = "llama-bench"
DEVICE_BINARY = f"{DEVICE_TMP}/{BINARY_NAME}"


class LlamaCppBackend(BackendBase):

    def name(self) -> str:
        return "llama.cpp"

    def _local_binary(self) -> Path:
        return SCRIPT_DIR / "binaries" / BINARY_NAME

    def is_available(self) -> bool:
        return self._local_binary().is_file()

    def run(self, model_path: Path) -> dict:
        result = self._empty_result(model_path)
        result["format"] = ".gguf"

        if not self.is_available():
            result["error"] = (
                f"llama.cpp binary not found at {self._local_binary()}. "
                "Build llama-bench for Android arm64 (via NDK or Termux) and put it in binaries/."
            )
            return result

        device_model = f"{DEVICE_TMP}/{model_path.name}"
        self.push(self._local_binary(), DEVICE_BINARY)
        self.shell_cmd(f"chmod 755 {DEVICE_BINARY}", timeout=10)
        self.push(model_path, device_model)

        cmd = (
            f"{DEVICE_BINARY} "
            f"-m {device_model} "
            f"-t {self.num_threads} "
            f"-r 3"
        )
        output = self.shell_cmd(cmd, timeout=int(self.wait_seconds + 120))
        result.update(self._parse(output))
        if not result.get("prompt_tokens_per_sec") and not result.get("generation_tokens_per_sec") and not result.get("error"):
            result["error"] = "Could not parse llama-bench output"
            result["raw_output"] = output[:800]
        return result

    def _empty_result(self, model_path: Path) -> dict:
        base = super()._empty_result(model_path)
        base.update({
            "prompt_tokens_per_sec": None,
            "generation_tokens_per_sec": None,
            "model_size_gib": None,
            "params_b": None,
        })
        return base

    @staticmethod
    def _parse(output: str) -> dict:
        parsed = {}

        # llama-bench outputs markdown-style table rows:
        #   | model | size | params | backend | threads | test | t/s |
        #   | llama 1B Q4_K_M | 0.62 GiB | 1.10 B | CPU | 4 | pp512 | 45.67 |
        #   | llama 1B Q4_K_M | 0.62 GiB | 1.10 B | CPU | 4 | tg128 | 12.34 |
        pp_speeds = []
        tg_speeds = []
        for line in output.splitlines():
            # Skip header/separator lines
            if "|" not in line or "model" in line.lower() and "test" in line.lower():
                continue
            cols = [c.strip() for c in line.split("|") if c.strip()]
            if len(cols) < 6:
                continue
            try:
                test_name = cols[-2].strip()
                speed = float(cols[-1].strip())
                if test_name.startswith("pp"):
                    pp_speeds.append(speed)
                elif test_name.startswith("tg"):
                    tg_speeds.append(speed)
                # Extract model size and params from first row
                if not parsed.get("model_size_gib"):
                    size_m = re.search(r"([\d.]+)\s*GiB", cols[1] if len(cols) > 1 else "")
                    if size_m:
                        parsed["model_size_gib"] = float(size_m.group(1))
                    param_m = re.search(r"([\d.]+)\s*B", cols[2] if len(cols) > 2 else "")
                    if param_m:
                        parsed["params_b"] = float(param_m.group(1))
            except (ValueError, IndexError):
                continue

        if pp_speeds:
            parsed["prompt_tokens_per_sec"] = round(sum(pp_speeds) / len(pp_speeds), 2)
        if tg_speeds:
            parsed["generation_tokens_per_sec"] = round(sum(tg_speeds) / len(tg_speeds), 2)
            parsed["inference_avg_ms"] = round(1000.0 / (sum(tg_speeds) / len(tg_speeds)), 3)

        # Fallback: parse "llama_print_timings" style output from llama-cli
        if not pp_speeds and not tg_speeds:
            m_eval = re.search(r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", output)
            if m_eval:
                total_ms = float(m_eval.group(1))
                tokens = int(m_eval.group(2))
                if tokens > 0:
                    ms_per_tok = total_ms / tokens
                    parsed["generation_tokens_per_sec"] = round(1000.0 / ms_per_tok, 2)
                    parsed["inference_avg_ms"] = round(ms_per_tok, 3)
            m_prompt = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", output)
            if m_prompt:
                total_ms = float(m_prompt.group(1))
                tokens = int(m_prompt.group(2))
                if tokens > 0:
                    parsed["prompt_tokens_per_sec"] = round(1000.0 * tokens / total_ms, 2)

        # Memory from "llama_print_timings" or system
        m_mem = re.search(r"mem(?:ory)?\s*(?:per token|required)[:\s]+([\d.]+)\s*(MiB|MB|GiB|GB)", output, re.IGNORECASE)
        if m_mem:
            val = float(m_mem.group(1))
            unit = m_mem.group(2).upper()
            if "G" in unit:
                val *= 1024
            parsed["memory_overall_mb"] = round(val, 2)

        return parsed
