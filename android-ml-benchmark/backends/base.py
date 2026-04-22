"""
Base class for all benchmark backends.
Each backend knows how to: push model, run benchmark, parse output, return metrics.
"""
from abc import ABC, abstractmethod
from pathlib import Path

from adb_interface import run_adb, ADBError

DEVICE_TMP = "/data/local/tmp"


class BackendBase(ABC):
    """All backends return a dict with at least these keys (None if unavailable)."""

    METRIC_KEYS = [
        "model",
        "format",
        "backend",
        "delegate",
        "init_ms",
        "first_inference_us",
        "warmup_avg_us",
        "inference_avg_us",
        "inference_avg_ms",
        "memory_init_mb",
        "memory_overall_mb",
        "error",
    ]

    def __init__(self, num_threads: int = 4, wait_seconds: float = 45):
        self.num_threads = num_threads
        self.wait_seconds = wait_seconds

    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name, e.g. 'TFLite APK'."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the backend's binary/APK is installed and ready."""
        ...

    @abstractmethod
    def run(self, model_path: Path) -> dict:
        """
        Run benchmark for one model. Returns dict with at least METRIC_KEYS.
        """
        ...

    def _empty_result(self, model_path: Path) -> dict:
        return {k: None for k in self.METRIC_KEYS} | {
            "model": model_path.name,
            "format": model_path.suffix.lower(),
            "backend": self.name(),
        }

    @staticmethod
    def clear_logcat():
        try:
            run_adb("logcat", "-c", check=False)
        except ADBError:
            pass

    @staticmethod
    def get_logcat(tag: str = "llama") -> str:
        try:
            r = run_adb("logcat", "-d", "-s", f"{tag}:I", check=False, timeout=15)
            return (r.stdout or "") + (r.stderr or "")
        except ADBError:
            return ""

    @staticmethod
    def push(local_path: Path, device_path: str):
        # Scale adb push timeout for large model files (GGUF can be hundreds of MB).
        size_mb = local_path.stat().st_size / (1024 * 1024)
        timeout = max(120, int(size_mb / 2) + 60)  # ~2 MB/s + 60s buffer
        run_adb("push", str(local_path), device_path, timeout=timeout)

    @staticmethod
    def shell_cmd(cmd: str, timeout: int = 180) -> str:
        try:
            r = run_adb("shell", cmd, check=False, timeout=timeout)
            return (r.stdout or "") + (r.stderr or "")
        except ADBError as e:
            return str(e)

    @staticmethod
    def rm(device_path: str):
        """Best-effort remove a file from the device."""
        try:
            run_adb("shell", f"rm -f {device_path}", check=False, timeout=10)
        except ADBError:
            pass
