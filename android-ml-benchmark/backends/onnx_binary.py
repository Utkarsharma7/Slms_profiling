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
SHARED_LIB = "libonnxruntime.so"
DEVICE_SHARED_LIB = f"{DEVICE_TMP}/{SHARED_LIB}"

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

    def _local_shared_lib(self) -> Path:
        return SCRIPT_DIR / "binaries" / SHARED_LIB

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
        # Fresh run: remove any leftovers
        self.rm(DEVICE_BINARY)
        self.rm(DEVICE_SHARED_LIB)
        self.rm(device_model)

        self.push(self._local_binary(), DEVICE_BINARY)
        self.shell_cmd(f"chmod 755 {DEVICE_BINARY}", timeout=10)

        # onnxruntime_perf_test is dynamically linked to libonnxruntime.so
        if self._local_shared_lib().is_file():
            self.push(self._local_shared_lib(), DEVICE_SHARED_LIB)
        else:
            result["error"] = (
                f"Missing {SHARED_LIB} next to onnxruntime_perf_test in binaries/. "
                f"Expected at {self._local_shared_lib()}."
            )
            return result

        self.push(model_path, device_model)

        ep_flag = self._ep_flag()
        # NOTE: onnxruntime_perf_test expects model path as a positional argument.
        # -m is "test mode" (duration/times), not "model".
        cmd = (
            f"LD_LIBRARY_PATH={DEVICE_TMP} {DEVICE_BINARY} "
            f"-e {ep_flag} "
            f"-x {self.num_threads} "
            f"-m times "
            f"-r 50 "
            f"-S 1 "
            f"-I "
            f"{device_model}"
        )

        output = self.shell_cmd(cmd, timeout=int(self.wait_seconds + 120))
        result.update(self._parse(output))

        # extra perf_test metrics
        m_ips = re.search(r"Number of inferences per second:\s*([\d.]+)", output, re.IGNORECASE)
        if m_ips:
            result["throughput_per_sec"] = round(float(m_ips.group(1)), 3)
        m_cpu = re.search(r"Avg CPU usage:\s*([\d.]+)\s*%", output, re.IGNORECASE)
        if m_cpu:
            result["cpu_usage_avg_pct"] = round(float(m_cpu.group(1)), 1)
        m_peak = re.search(r"Peak working set size:\s*(\d+)\s*bytes", output, re.IGNORECASE)
        if m_peak:
            result["peak_working_set_bytes"] = int(m_peak.group(1))

        # per-op profiling isn't stable/standardized for perf_test output; keep it off by default

        if not result.get("inference_avg_ms") and not result.get("error"):
            result["error"] = "Could not parse ONNX benchmark output"
            result["raw_output"] = output[:2000]

        # Clean up device (always try, even if parse failed)
        self.rm(device_model)
        self.rm(DEVICE_BINARY)
        self.rm(DEVICE_SHARED_LIB)
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

        # "Average inference time cost total: 27.05 ms" (common perf_test format)
        m = re.search(
            r"(?:Average\s+(?:inference\s+)?time\s*(?:cost)?(?:\s*total)?|"
            r"Average\s+inference\s+time\s+cost\s+total|"
            r"Avg\s+latency|average)[:\s]+([\d.]+)\s*ms",
            output, re.IGNORECASE,
        )
        if m:
            parsed["inference_avg_ms"] = round(float(m.group(1)), 3)
            parsed["inference_avg_us"] = round(float(m.group(1)) * 1000, 1)

        # Percentile latencies can be in seconds: "P50 Latency: 0.027 s"
        for pct in ("50", "90", "95", "99"):
            pm_ms = re.search(rf"P{pct}\s*(?:Latency|latency)?[:\s]+([\d.]+)\s*ms", output, re.IGNORECASE)
            pm_s = re.search(rf"P{pct}\s*(?:Latency|latency)?[:\s]+([\d.]+)\s*s", output, re.IGNORECASE)
            if pm_ms:
                parsed[f"p{pct}_ms"] = round(float(pm_ms.group(1)), 3)
            elif pm_s:
                parsed[f"p{pct}_ms"] = round(float(pm_s.group(1)) * 1000.0, 3)

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

        # Init time: "Session creation time cost: 0.034 s" or "... 123.45 ms"
        m_init = re.search(
            r"(?:Session creation time cost|Init time|Initialization)[:\s]+([\d.]+)\s*(ms|s)",
            output, re.IGNORECASE,
        )
        if m_init:
            val = float(m_init.group(1))
            unit = m_init.group(2).lower()
            if unit == "s":
                val *= 1000.0
            parsed["init_ms"] = round(val, 2)

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

        # Memory: "Peak working set size: 66596864 bytes"
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
