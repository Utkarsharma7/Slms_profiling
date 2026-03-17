from backends.base import BackendBase
from backends.tflite_apk import TFLiteAPKBackend
from backends.onnx_binary import ONNXRuntimeBackend
from backends.pytorch_binary import PyTorchMobileBackend
from backends.llamacpp_binary import LlamaCppBackend

BACKENDS = {
    ".tflite": TFLiteAPKBackend,
    ".onnx": ONNXRuntimeBackend,
    ".pt": PyTorchMobileBackend,
    ".ptl": PyTorchMobileBackend,
    ".gguf": LlamaCppBackend,
}


def get_backend_for_model(model_path):
    """Return the right backend class based on file extension."""
    ext = model_path.suffix.lower()
    cls = BACKENDS.get(ext)
    if cls is None:
        raise ValueError(
            f"Unsupported model format: {ext}\n"
            f"Supported: {', '.join(BACKENDS.keys())}"
        )
    return cls
