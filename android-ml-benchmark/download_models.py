"""
Download TFLite, GGUF, and ONNX models for benchmarking.
Run: python download_models.py --all
Or:  python download_models.py --tflite
Or:  python download_models.py --gguf
Or:  python download_models.py --onnx
"""
import argparse
import sys
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"

# ──────────────────────────────────────────────────────────────────
# TFLite models (small, ready to benchmark with APK)
# ──────────────────────────────────────────────────────────────────
TFLITE_MODELS = {
    "mobilenet_v2_1.0_224.tflite": {
        "url": "https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tflite",
        "category": "image_classification",
        "size_mb": 14,
    },
    "efficientnet_lite0_int8.tflite": {
        "url": "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/image_classification/android/lite-model_efficientnet_lite0_int8_2.tflite",
        "category": "image_classification",
        "size_mb": 5,
    },
    "efficientnet_lite2_int8.tflite": {
        "url": "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/image_classification/android/lite-model_efficientnet_lite2_int8_1.tflite",
        "category": "image_classification",
        "size_mb": 8,
    },
    "ssd_mobilenet_v1.tflite": {
        "url": "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/android/lite-model_ssd_mobilenet_v1_1_metadata_2.tflite",
        "category": "object_detection",
        "size_mb": 4,
    },
    "efficientdet_lite0.tflite": {
        "url": "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/android/lite-model_efficientdet_lite0_detection_metadata_1.tflite",
        "category": "object_detection",
        "size_mb": 5,
    },
    "deeplabv3_257_mv_gpu.tflite": {
        "url": "https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite",
        "category": "segmentation",
        "size_mb": 2,
    },
    "movenet_lightning_f16.tflite": {
        "url": "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite",
        "category": "pose_estimation",
        "size_mb": 5,
    },
    "mobilebert_float.tflite": {
        "url": "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/nl_classifier/android/mobilebert_with_metadata.tflite",
        "category": "nlp",
        "size_mb": 25,
    },
}

# ──────────────────────────────────────────────────────────────────
# GGUF models (LLMs for llama.cpp)
# ──────────────────────────────────────────────────────────────────
GGUF_MODELS = {
    "tinyllama-1.1b-chat-q4_k_m.gguf": {
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "params": "1.1B",
        "size_mb": 669,
    },
    "qwen2.5-0.5b-instruct-q4_k_m.gguf": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "params": "0.5B",
        "size_mb": 386,
    },
    "qwen2.5-1.5b-instruct-q4_k_m.gguf": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "params": "1.5B",
        "size_mb": 986,
    },
    "smollm2-360m-instruct-q4_k_m.gguf": {
        "url": "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf",
        "params": "360M",
        "size_mb": 230,
    },
    "llama-3.2-1b-instruct-q4_k_m.gguf": {
        "url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "params": "1B",
        "size_mb": 776,
    },
    "gemma-2-2b-it-q4_k_m.gguf": {
        "url": "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
        "params": "2B",
        "size_mb": 1630,
    },
    "phi-3.5-mini-instruct-q4_k_m.gguf": {
        "url": "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "params": "3.8B",
        "size_mb": 2180,
    },
}

# ──────────────────────────────────────────────────────────────────
# ONNX models (vision, NLP, general-purpose)
# From ONNX Model Zoo & HuggingFace
# ──────────────────────────────────────────────────────────────────
ONNX_MODELS = {
    # Image classification
    "mobilenetv2-12.onnx": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
        "category": "image_classification",
        "size_mb": 14,
    },
    "squeezenet1.1-7.onnx": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx",
        "category": "image_classification",
        "size_mb": 5,
    },
    "shufflenet-v2-12.onnx": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/shufflenet/model/shufflenet-v2-12.onnx",
        "category": "image_classification",
        "size_mb": 9,
    },
    "efficientnet-lite4-11.onnx": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx",
        "category": "image_classification",
        "size_mb": 50,
    },
    # Object detection
    "ssd-mobilenetv1-12.onnx": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_12.onnx",
        "category": "object_detection",
        "size_mb": 28,
    },
    # Super resolution
    "super-resolution-10.onnx": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx",
        "category": "super_resolution",
        "size_mb": 1,
    },
    # NLP
    "bert-base-uncased-3.onnx": {
        "url": "https://github.com/onnx/models/raw/main/validated/text/machine_comprehension/bert-squad/model/bertsquad-12.onnx",
        "category": "nlp_qa",
        "size_mb": 430,
    },
    # Emotion / face
    "emotion-ferplus-8.onnx": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
        "category": "emotion_detection",
        "size_mb": 34,
    },
}


def download_file(url: str, dest: Path, label: str = ""):
    """Download a file with progress."""
    if dest.exists():
        print(f"  SKIP (exists): {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"  Downloading: {label or dest.name}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        mb = downloaded / (1024 * 1024)
                        print(f"\r    {mb:.1f} MB / {total/(1024*1024):.1f} MB ({pct:.0f}%)", end="", flush=True)
                    else:
                        mb = downloaded / (1024 * 1024)
                        print(f"\r    {mb:.1f} MB", end="", flush=True)
        tmp.rename(dest)
        print(f"\r    Done: {dest.name} ({downloaded/(1024*1024):.1f} MB)          ")
    except Exception as e:
        print(f"\r    FAILED: {e}")
        if tmp.exists():
            tmp.unlink()


def main():
    all_catalogs = {
        "tflite": ("TFLite", TFLITE_MODELS),
        "gguf": ("GGUF (llama.cpp)", GGUF_MODELS),
        "onnx": ("ONNX", ONNX_MODELS),
    }

    parser = argparse.ArgumentParser(description="Download models for benchmarking")
    parser.add_argument("--all", action="store_true", help="Download all models (TFLite + GGUF + ONNX)")
    parser.add_argument("--tflite", action="store_true", help="Download TFLite models only")
    parser.add_argument("--gguf", action="store_true", help="Download GGUF models only")
    parser.add_argument("--onnx", action="store_true", help="Download ONNX models only")
    parser.add_argument("--list", action="store_true", help="List available models and exit")
    parser.add_argument("--name", type=str, help="Download one specific model by name")
    args = parser.parse_args()

    if args.list:
        print("TFLite models:")
        for name, info in TFLITE_MODELS.items():
            print(f"  {name:<45} {info['category']:<25} ~{info['size_mb']} MB")
        print(f"\nGGUF models (for llama.cpp):")
        for name, info in GGUF_MODELS.items():
            print(f"  {name:<45} {info['params']:<8} ~{info['size_mb']} MB")
        print(f"\nONNX models:")
        for name, info in ONNX_MODELS.items():
            print(f"  {name:<45} {info['category']:<25} ~{info['size_mb']} MB")
        total = sum(v["size_mb"] for v in TFLITE_MODELS.values())
        total += sum(v["size_mb"] for v in GGUF_MODELS.values())
        total += sum(v["size_mb"] for v in ONNX_MODELS.values())
        print(f"\nTotal: ~{total} MB ({total/1024:.1f} GB) if downloading everything")
        return 0

    if args.name:
        merged = {**TFLITE_MODELS, **GGUF_MODELS, **ONNX_MODELS}
        if args.name in merged:
            info = merged[args.name]
            download_file(info["url"], MODELS_DIR / args.name, args.name)
        else:
            print(f"Model '{args.name}' not found. Use --list to see available models.")
            return 1
        return 0

    if not (args.all or args.tflite or args.gguf or args.onnx):
        parser.print_help()
        print("\nUse --list to see available models, --all to download everything.")
        return 0

    if args.all or args.tflite:
        print(f"Downloading {len(TFLITE_MODELS)} TFLite models to {MODELS_DIR}")
        for name, info in TFLITE_MODELS.items():
            download_file(info["url"], MODELS_DIR / name, f"{name} (~{info['size_mb']} MB)")

    if args.all or args.gguf:
        print(f"\nDownloading {len(GGUF_MODELS)} GGUF models to {MODELS_DIR}")
        for name, info in GGUF_MODELS.items():
            download_file(info["url"], MODELS_DIR / name, f"{name} ({info['params']}, ~{info['size_mb']} MB)")

    if args.all or args.onnx:
        print(f"\nDownloading {len(ONNX_MODELS)} ONNX models to {MODELS_DIR}")
        for name, info in ONNX_MODELS.items():
            download_file(info["url"], MODELS_DIR / name, f"{name} (~{info['size_mb']} MB)")

    print("\nDone. Models are in:", MODELS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
