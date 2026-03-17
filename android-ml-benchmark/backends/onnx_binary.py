"""
ONNX Runtime benchmark via onnxruntime_perf_test binary on Android.
Model format: .onnx

Supports execution providers (delegates):
  - cpu: Default CPU execution provider
  - nnapi: Android NNAPI (routes to DSP/NPU/GPU depending on SoC)
  - xnnpack: XNNPACK for optimised CPU inference

Binary: onnxruntime_perf_test (arm64, built from ONNX Runtime source)
Place it in: android-ml-benchmark/binaries/onnxruntime_perf_test

Usage on device:
  onnxruntime_perf_test -m model.onnx -r 50 -e cpu
  onnxruntime_perf_test -m model.onnx -r 50 -e nnapi
  onnxruntime_perf_test -m model.onnx -r 50 -e cpu -o 1  (op profiling)
Output contains lines like:
  "Average time: 12.345 ms"
  "Total time: 617.25 ms"
  "Total iterations: 50"
"""
import re
from pathlib import Path

from backends.base import BackendBase, DEVICE_TMP

SCRIPT_DIR = Path(__file__).resolve().parent.parent
BINARY_NAME = "onnxruntime_perf_test"
DEVICE_BINARY = f"{DEVICE_TMP}/{BINARY_NAME}"

ONNX_EXECUTION_PROVIDERS = ("cpu", "nnapi", "xnnpack")


class ONNXRuntimeBackend(BackendBase):

    def __init__(self, num_threads: int = 4, wait_seconds: float = 45,
                 execution_provider: str = "cpu", op_profiling: bool = True):
        super().__init__(num_threads, wait_seconds)
        self.execution_provider = execution_provider.lower()
        self.op_profiling = op_profiling

    def name(self) -> str:
        return f"ONNX Runtime ({self.execution_provider.upper()})"

    def _local_binary(self) -> Path:
        return SCRIPT_DIR / "binaries" / BINARY_NAME

    def is_available(self) -> bool:
        return self._local_binary().is_file()

    def run(self, model_path: Path) -> dict:
        result = self._empty_result(model_path)
        result["delegate"] = self.execution_provider

        if not self.is_available():
            result["error"] = (
                f"ONNX binary not found at {self._local_binary()}. "
                "Build onnxruntime_perf_test for Android arm64 and put it in binaries/."
            )
            return result

        device_model = f"{DEVICE_TMP}/{model_path.name}"
        self.push(self._local_binary(), DEVICE_BINARY)
        self.shell_cmd(f"chmod 755 {DEVICE_BINARY}", timeout=10)
        self.push(model_path, device_model)

        ep_flag = self._ep_flag()
        cmd = (
            f"{DEVICE_BINARY} "
            f"-m {device_model} "
            f"-r 50 "
            f"-e {ep_flag} "
            f"-x {self.num_threads}"
        )
        if self.op_profiling:
            cmd += " -o 1"

        output = self.shell_cmd(cmd, timeout=int(self.wait_seconds + 60))
        result.update(self._parse(output))

        if self.op_profiling:
            ops = self._parse_op_profile(output)
            if ops:
                result["op_profile"] = ops
                result["top_ops"] = ops[:5]

        if not result.get("inference_avg_ms") and not result.get("error"):
            result["error"] = "Could not parse ONNX benchmark output"
            result["raw_output"] = output[:800]
        return result

    def _ep_flag(self) -> str:
        """Map our execution_provider name to onnxruntime_perf_test -e flag."""
        mapping = {
            "cpu": "cpu",
            "nnapi": "nnapi",
            "xnnpack": "xnnpack",
        }
        return mapping.get(self.execution_provider, "cpu")

    @staticmethod
    def _parse(output: str) -> dict:
        parsed = {}

        # "Average inference time cost: 12.345 ms" (common ORT format)
        m = re.search(
            r"(?:Average\s+(?:inference\s+)?time\s*(?:cost)?|Avg\s+latency|average)[:\s]+([\d.]+)\s*ms",
            output, re.IGNORECASE,
        )
        if m:
            parsed["inference_avg_ms"] = round(float(m.group(1)), 3)
            parsed["inference_avg_us"] = round(float(m.group(1)) * 1000, 1)

        # Percentile latencies: "P50 latency (ms): 12.34"
        for pct in ("50", "90", "95", "99"):
            pm = re.search(rf"P{pct}\s*(?:latency)?\s*\(?ms\)?[:\s]+([\d.]+)", output, re.IGNORECASE)
            if pm:
                parsed[f"p{pct}_ms"] = round(float(pm.group(1)), 3)

        # "Total time: 617.25 ms"
        m2 = re.search(r"Total time[:\s]+([\d.]+)\s*ms", output, re.IGNORECASE)

        # "Total iterations: 50" or "Total runs: 50"
        m3 = re.search(r"(?:Total iterations|Total runs|Runs)[:\s]+(\d+)", output, re.IGNORECASE)

        if not parsed.get("inference_avg_ms") and m2 and m3:
            total = float(m2.group(1))
            iters = int(m3.group(1))
            if iters > 0:
                parsed["inference_avg_ms"] = round(total / iters, 3)
                parsed["inference_avg_us"] = round(parsed["inference_avg_ms"] * 1000, 1)

        # Init time: "Session creation time cost: 123.45 ms" or "Init time: 123 ms"
        m_init = re.search(
            r"(?:Session creation time cost|Init time|Initialization)[:\s]+([\d.]+)\s*ms",
            output, re.IGNORECASE,
        )
        if m_init:
            parsed["init_ms"] = round(float(m_init.group(1)), 2)

        # First run: "First inference time cost: 45.67 ms"
        m_first = re.search(
            r"(?:First (?:inference|run) (?:time )?cost|1st run)[:\s]+([\d.]+)\s*(us|ms)",
            output, re.IGNORECASE,
        )
        if m_first:
            val = float(m_first.group(1))
            if m_first.group(2).lower() == "ms":
                val *= 1000
            parsed["first_inference_us"] = round(val, 1)

        # Warmup: "Warmup time cost: 234.56 ms"
        m_warm = re.search(
            r"Warmup (?:time )?(?:cost)?[:\s]+([\d.]+)\s*ms",
            output, re.IGNORECASE,
        )
        if m_warm:
            parsed["warmup_avg_us"] = round(float(m_warm.group(1)) * 1000, 1)

        # Memory: "Peak working set size: 123456789 bytes" or "peak memory: 123 MB"
        m4 = re.search(
            r"(?:peak\s*(?:working set|memory)\s*(?:size)?|total\s*memory)[:\s]+([\d.]+)\s*(bytes|KB|MB|GB)?",
            output, re.IGNORECASE,
        )
        if m4:
            val = float(m4.group(1))
            unit = (m4.group(2) or "bytes").upper()
            if unit.startswith("B"):
                val /= (1024 * 1024)
            elif unit.startswith("K"):
                val /= 1024
            elif unit.startswith("G"):
                val *= 1024
            parsed["memory_overall_mb"] = round(val, 2)

        return parsed

    @staticmethod
    def _parse_op_profile(output: str) -> list[dict]:
        """
        Parse per-operator profiling from onnxruntime_perf_test output.
        Typical format:
          kernel_time(ms)  kernel_name
          0.123           Conv
          0.456           MatMul
        Or:
          Node    | Time (ms) | Percentage
          Conv_0  | 1.234     | 25.00%
        """
        ops = []

        # Try table format with pipes
        for line in output.splitlines():
            m = re.match(
                r"\s*(\w[\w/_.]*)\s*\|\s*([\d.]+)\s*(?:ms)?\s*\|\s*([\d.]+)\s*%?",
                line,
            )
            if m:
                name = m.group(1).strip()
                if name.lower() in ("node", "operator", "name"):
                    continue
                ops.append({
                    "op": name,
                    "time": float(m.group(2)),
                    "pct": float(m.group(3)),
                })

        # Try whitespace-delimited format
        if not ops:
            for line in output.splitlines():
                m = re.match(
                    r"\s*([\d.]+)\s+(\w[\w/_.]*)",
                    line,
                )
                if m:
                    try:
                        t = float(m.group(1))
                        ops.append({"op": m.group(2), "time": t, "pct": 0.0})
                    except ValueError:
                        pass

        # Compute percentages if not present
        if ops and all(op["pct"] == 0.0 for op in ops):
            total = sum(op["time"] for op in ops)
            if total > 0:
                for op in ops:
                    op["pct"] = round(op["time"] / total * 100, 2)

        ops.sort(key=lambda x: x["time"], reverse=True)
        return ops
