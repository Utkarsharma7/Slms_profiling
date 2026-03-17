"""
PyTorch Mobile benchmark via speed_benchmark_torch binary on Android.
Model format: .pt or .ptl (TorchScript or PyTorch Lite Interpreter)

Binary: speed_benchmark_torch (arm64, built from PyTorch source)
Place it in: android-ml-benchmark/binaries/speed_benchmark_torch

Usage on device:
  speed_benchmark_torch --model=model.ptl --input_dims="1,3,224,224" --input_type=float --warmup=10 --iter=50
Output contains lines like:
  "Main run finished. Microseconds per iter: 12345.6. Iters per second: 81.0"
"""
import re
from pathlib import Path

from backends.base import BackendBase, DEVICE_TMP

SCRIPT_DIR = Path(__file__).resolve().parent.parent
BINARY_NAME = "speed_benchmark_torch"
DEVICE_BINARY = f"{DEVICE_TMP}/{BINARY_NAME}"


class PyTorchMobileBackend(BackendBase):

    def __init__(self, num_threads: int = 4, wait_seconds: float = 45,
                 input_dims: str = "1,3,224,224", input_type: str = "float"):
        super().__init__(num_threads, wait_seconds)
        self.input_dims = input_dims
        self.input_type = input_type

    def name(self) -> str:
        return "PyTorch Mobile"

    def _local_binary(self) -> Path:
        return SCRIPT_DIR / "binaries" / BINARY_NAME

    def is_available(self) -> bool:
        return self._local_binary().is_file()

    def run(self, model_path: Path) -> dict:
        result = self._empty_result(model_path)

        if not self.is_available():
            result["error"] = (
                f"PyTorch binary not found at {self._local_binary()}. "
                "Build speed_benchmark_torch for Android arm64 and put it in binaries/."
            )
            return result

        device_model = f"{DEVICE_TMP}/{model_path.name}"
        self.push(self._local_binary(), DEVICE_BINARY)
        self.shell_cmd(f"chmod 755 {DEVICE_BINARY}", timeout=10)
        self.push(model_path, device_model)

        cmd = (
            f"{DEVICE_BINARY} "
            f"--model={device_model} "
            f"--input_dims=\"{self.input_dims}\" "
            f"--input_type={self.input_type} "
            f"--warmup=10 --iter=50"
        )
        output = self.shell_cmd(cmd, timeout=int(self.wait_seconds + 60))
        result.update(self._parse(output))
        if not result.get("inference_avg_ms") and not result.get("error"):
            result["error"] = "Could not parse PyTorch benchmark output"
            result["raw_output"] = output[:500]
        return result

    @staticmethod
    def _parse(output: str) -> dict:
        parsed = {}
        # "Main run finished. Microseconds per iter: 12345.6. Iters per second: 81.0"
        m = re.search(
            r"Microseconds per iter[:\s]+([\d.]+)",
            output, re.IGNORECASE,
        )
        if m:
            us = float(m.group(1))
            parsed["inference_avg_us"] = us
            parsed["inference_avg_ms"] = round(us / 1000, 3)

        m2 = re.search(r"Iters per second[:\s]+([\d.]+)", output, re.IGNORECASE)
        if m2:
            parsed["throughput_per_sec"] = float(m2.group(1))

        # Fallback: "latency: X us" or "avg: X ms"
        if not parsed.get("inference_avg_us"):
            m3 = re.search(r"(?:latency|avg)[:\s]+([\d.]+)\s*(us|ms)", output, re.IGNORECASE)
            if m3:
                val = float(m3.group(1))
                if m3.group(2).lower() == "ms":
                    parsed["inference_avg_ms"] = val
                    parsed["inference_avg_us"] = val * 1000
                else:
                    parsed["inference_avg_us"] = val
                    parsed["inference_avg_ms"] = round(val / 1000, 3)

        return parsed
