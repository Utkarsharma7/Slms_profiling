"""
TFLite benchmark via the official TensorFlow Lite benchmark APK.
Model format: .tflite
Supports delegates: cpu, gpu, nnapi
Supports per-operator profiling via --enable_op_profiling
"""
import re
import time
from pathlib import Path

from backends.base import BackendBase, DEVICE_TMP


TF_ACTIVITY = "org.tensorflow.lite.benchmark/.BenchmarkModelActivity"

DELEGATES = ("cpu", "gpu", "nnapi")


class TFLiteAPKBackend(BackendBase):

    def __init__(self, num_threads: int = 4, wait_seconds: float = 45,
                 delegate: str = "cpu", op_profiling: bool = True):
        super().__init__(num_threads, wait_seconds)
        self.delegate = delegate.lower()
        self.op_profiling = op_profiling

    def name(self) -> str:
        return f"TFLite APK ({self.delegate.upper()})"

    def is_available(self) -> bool:
        out = self.shell_cmd("pm list packages org.tensorflow.lite.benchmark", timeout=10)
        return "org.tensorflow.lite.benchmark" in out

    def run(self, model_path: Path) -> dict:
        result = self._empty_result(model_path)
        result["delegate"] = self.delegate
        device_model = f"{DEVICE_TMP}/model.tflite"

        self.clear_logcat()
        self.push(model_path, device_model)

        args_str = f"--graph={device_model} --num_threads={self.num_threads}"
        if self.delegate == "gpu":
            args_str += " --use_gpu=true"
        elif self.delegate == "nnapi":
            args_str += " --use_nnapi=true"
        if self.op_profiling:
            args_str += " --enable_op_profiling=true"

        self.shell_cmd(
            f"am start -n {TF_ACTIVITY} --es args \"{args_str}\"",
            timeout=15,
        )
        time.sleep(self.wait_seconds)

        log = self.get_logcat("tflite")
        result.update(self._parse(log))
        return result

    @staticmethod
    def _parse(log: str) -> dict:
        parsed = {}
        m = re.search(
            r"Inference timings in us:\s*Init:\s*([\d.]+),\s*First inference:\s*([\d.]+),"
            r"\s*Warmup \(avg\):\s*([\d.]+),\s*Inference \(avg\):\s*([\d.]+)",
            log,
        )
        if m:
            parsed["init_ms"] = round(float(m.group(1)) / 1000, 2)
            parsed["first_inference_us"] = float(m.group(2))
            parsed["warmup_avg_us"] = float(m.group(3))
            parsed["inference_avg_us"] = float(m.group(4))
            parsed["inference_avg_ms"] = round(float(m.group(4)) / 1000, 3)
        m2 = re.search(
            r"Memory footprint delta.*?init=([\d.]+)\s+overall=([\d.]+)",
            log, re.DOTALL,
        )
        if m2:
            parsed["memory_init_mb"] = round(float(m2.group(1)), 2)
            parsed["memory_overall_mb"] = round(float(m2.group(2)), 2)

        # Per-operator profiling: extract operator timings
        op_profile = _parse_op_profiling(log)
        if op_profile:
            parsed["op_profile"] = op_profile
            parsed["top_ops"] = op_profile[:5]

        if not m:
            parsed["error"] = "No inference timings in logcat (model may be too large or unsupported)"
        return parsed


def _parse_op_profiling(log: str) -> list[dict]:
    """
    Parse per-operator profiling from logcat.
    Lines look like:
      [node type]         [start]  [first] [avg ms]    [%]   [cdf%]  [mem KB] [times called] [Name]
      CONV_2D                0.006  0.123     0.100   25.00%  25.00%     0.000          1      [input]/Conv2D
    Or simpler lines with just operator, time, percentage.
    """
    ops = []
    # Pattern: operator name followed by timing data
    # The TFLite benchmark outputs a table; try to capture rows
    for line in log.splitlines():
        # Match lines that look like profiling output
        m = re.match(
            r"\s*(\w[\w/]*)\s+"           # operator type
            r"(?:[\d.]+\s+)*?"            # optional start/first columns
            r"([\d.]+)\s+"                # avg time (ms or us)
            r"([\d.]+)%",                 # percentage
            line,
        )
        if m:
            ops.append({
                "op": m.group(1),
                "time": float(m.group(2)),
                "pct": float(m.group(3)),
            })
    # Also try a simpler pattern for some TFLite versions
    if not ops:
        for line in log.splitlines():
            m = re.match(
                r".*?Number of nodes.*?(\d+)",
                line,
            )
    return ops
