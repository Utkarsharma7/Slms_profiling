"""
ADB interface for communicating with an Android device over USB.
Handles device detection, pushing files, running shell commands, and pulling output.
"""
import subprocess
import sys
from pathlib import Path
from typing import Optional


class ADBError(Exception):
    """Raised when an ADB command fails."""
    pass


class NoDeviceError(ADBError):
    """Raised when no Android device is connected or authorized."""
    pass


def run_adb(
    *args: str,
    check: bool = True,
    capture_output: bool = True,
    text: bool = True,
    timeout: Optional[int] = 60,
) -> subprocess.CompletedProcess:
    """
    Run an ADB command. All args are passed after 'adb'.
    Example: run_adb('devices') -> adb devices
    """
    cmd = ["adb"] + list(args)
    try:
        return subprocess.run(
            cmd,
            check=check,
            capture_output=capture_output,
            text=text,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as e:
        raise ADBError(f"ADB failed: {' '.join(cmd)}\n{e.stderr or e.stdout or ''}")
    except FileNotFoundError:
        raise ADBError(
            "ADB not found. Install Android Platform Tools and add to PATH:\n"
            "  https://developer.android.com/studio/releases/platform-tools"
        )
    except subprocess.TimeoutExpired:
        raise ADBError(f"ADB command timed out: {' '.join(cmd)}")


def get_connected_devices() -> list[str]:
    """
    Return list of connected device serials in 'device' state.
    Empty list if none connected or USB debugging not authorized.
    """
    result = run_adb("devices", check=True)
    lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    devices = []
    for line in lines[1:]:  # skip header "List of devices attached"
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            devices.append(parts[0])
    return devices


def get_device(serial: Optional[str] = None) -> Optional[str]:
    """
    Return the device serial to use. If serial is given and in the list, use it.
    Otherwise return the first connected device, or None.
    """
    devices = get_connected_devices()
    if not devices:
        return None
    if serial:
        return serial if serial in devices else devices[0]
    return devices[0]


def push_file(local_path: Path, device_path: str, serial: Optional[str] = None) -> None:
    """Push a file to the device. device_path is the full path on the device (e.g. /data/local/tmp/model.gguf)."""
    device = get_device(serial)
    if not device:
        raise NoDeviceError("No Android device connected. Connect via USB and enable USB debugging.")
    args = ["-s", device, "push", str(local_path), device_path]
    run_adb(*args, timeout=120)


def shell(
    command: str,
    serial: Optional[str] = None,
    timeout: Optional[int] = 120,
) -> str:
    """
    Run a command on the device via adb shell. Returns combined stdout + stderr.
    """
    device = get_device(serial)
    if not device:
        raise NoDeviceError("No Android device connected. Connect via USB and enable USB debugging.")
    result = run_adb("-s", device, "shell", command, timeout=timeout)
    return (result.stdout or "") + (result.stderr or "")


def file_exists_on_device(device_path: str, serial: Optional[str] = None) -> bool:
    """Return True if the path exists on the device (file or directory)."""
    out = shell(f"test -e '{device_path}' && echo 1 || echo 0", serial=serial, timeout=10)
    return "1" in out


def remove_on_device(device_path: str, serial: Optional[str] = None) -> None:
    """Remove a file or directory on the device."""
    shell(f"rm -rf '{device_path}'", serial=serial, timeout=30)
