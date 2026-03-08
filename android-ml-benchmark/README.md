# Android ML Benchmark

Run ML models (.tflite) on your Android phone over USB and see **inference latency** and **memory** on your PC. Uses the TensorFlow Lite benchmark APK; no IDE or app build required.

**How it works:** The benchmark does **not** use real images (or any real input). It feeds each model **dummy input** with the correct shape and measures how long one forward pass takes. So you get timing and memory for any .tflite (image classifier, object detector, etc.) without providing data.

---

## What you need

- **PC:** Python 3.8+, ADB (Android Platform Tools)
- **Phone:** Android 7+, USB cable, USB debugging on
- **Files:** TensorFlow benchmark APK (e.g. `android_aarch64_benchmark_model.apk`) + a `.tflite` model

---

## Setup (once)

**1. Install ADB**  
Windows: `winget install Google.PlatformTools`  
Or download [Platform Tools](https://developer.android.com/studio/releases/platform-tools) and add to PATH.

**2. USB debugging on phone**  
Settings → About phone → tap Build number 7 times → Developer options → USB debugging on. Connect via USB and allow debugging.

**3. Install the benchmark APK (once per phone)**  
```bash
adb install -r -d "path\to\android_aarch64_benchmark_model.apk"
```

**4. Put your .tflite model** in **`android-ml-benchmark/models/`**.

---

## Run benchmark and see results on PC

```bash
cd android-ml-benchmark
python run_tf_apk_benchmark.py
```

This pushes the model to the phone, runs the benchmark, reads the log, and prints a summary in the terminal and saves to **results/tf_apk_benchmark_results.json**.

**One specific model:**
```bash
python run_tf_apk_benchmark.py "models\your_model.tflite"
```

**All .tflite models in `models/`** (run each and save combined results):
```bash
python run_tf_apk_benchmark.py --all
```

**Other options:** `--wait 10`, `-o my_results.json`, `--help`

---

## Project layout

```
android-ml-benchmark/
├── adb_interface.py           # ADB helpers (used by the script)
├── run_tf_apk_benchmark.py     # Run benchmark, show results on PC
├── requirements.txt           # Optional (no required deps)
├── models/                    # Put .tflite files here
├── results/                   # tf_apk_benchmark_results.json
└── README.md
```

---

## If something goes wrong

- **No device** → `adb devices`. Check cable, USB debugging, and “Allow” on the phone.
- **ADB not found** → Install Platform Tools and add to PATH.
