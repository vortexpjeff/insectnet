"""InsectNet Capture Sidecar — real-time insect audio classification on BirdNET-Pi.

Watches BirdNET-Pi's StreamData/ for new WAVs, runs them through BirdNET TFLite
inference, classifies via a trained sklearn head, and keeps tagged detections.

Mirrors BirdNET-Pi's own philosophy: detections leave artifacts, non-detections
leave zero traces. Runs alongside BirdNET-Pi without touching its files.

Usage (on BirdNET-Pi):
    python3 -m insectnet.capture --threshold 0.3

Usage (pull from Athena):
    python3 -m insectnet.capture --pull
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from . import birdnet

# ─────────────────────────────────────────────
# Defaults (customize for your BirdNET-Pi)
# ─────────────────────────────────────────────
DEFAULT_STREAMDATA = "/home/birdnetpi/BirdSongs/StreamData"
DEFAULT_BIRDNET_MODEL = "/home/birdnetpi/BirdNET-Pi/model/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite"
DEFAULT_PI_HOST = "birdnetpi@192.168.1.223"
DEFAULT_CAPTURE_DIR = "~/insectnet_capture"
DEFAULT_THRESHOLD = 0.3
DEFAULT_NOISE_FLOOR = 0.002
PI_SERVICES = ["birdnet_recording", "birdnet_analysis", "birdnet_log", "birdnet_stats"]

# ─────────────────────────────────────────────
# Global state (not pretty, but needed for signal handlers)
# ─────────────────────────────────────────────
interpreter = None
classifier_data: dict | None = None
capture_dir: str = ""
log_path: str = ""
temp_dir: str = ""
show_scores = False
threshold = DEFAULT_THRESHOLD
noise_floor = DEFAULT_NOISE_FLOOR
running = False
stats = {"processed": 0, "kept": 0, "discarded": 0, "errors": 0}

# Paths set from config
STREAMDATA_DIR: str = DEFAULT_STREAMDATA
BIRDNET_MODEL_PATH: str = DEFAULT_BIRDNET_MODEL
PI_HOST: str = DEFAULT_PI_HOST


# ─────────────────────────────────────────────
# Classifier helpers
# ─────────────────────────────────────────────


def load_classifier(joblib_path: str) -> dict:
    """Load a trained InsectNet sklearn classifier.

    Expected dict keys: scaler, classifier, classes.
    """
    import joblib

    m = joblib.load(joblib_path)
    return {
        "scaler": m["scaler"],
        "classifier": m["classifier"],
        "classes": m["classes"],
    }


def classify_logits(logits: np.ndarray, clf_data: dict) -> dict[str, float]:
    """Run the sklearn head on BirdNET logits.

    Returns dict of {class_name: confidence}.
    """
    X = clf_data["scaler"].transform(logits.reshape(1, -1))
    probs = clf_data["classifier"].predict_proba(X)[0]
    return {cls: float(probs[i]) for i, cls in enumerate(clf_data["classes"])}


# ─────────────────────────────────────────────
# WAV processing
# ─────────────────────────────────────────────


def process_wav(wav_path: str) -> bool:
    """Process one WAV: copy → RMS gate → TFLite → classify → keep/discard.

    Returns True if kept (detection or uncertain), False if discarded.
    """
    global stats
    basename = os.path.basename(wav_path)

    # Copy to temp (avoids race with BirdNET deletion)
    temp_path = os.path.join(temp_dir, basename)
    try:
        shutil.copy2(wav_path, temp_path)
    except (OSError, IOError):
        stats["errors"] += 1
        return False

    stats["processed"] += 1

    try:
        audio = birdnet.load_audio(temp_path)
        rms_val = birdnet.rms(audio)

        if rms_val < noise_floor:
            os.remove(temp_path)
            stats["discarded"] += 1
            return False

        logits = birdnet.extract_logits(audio, interpreter)
        scores = classify_logits(logits, classifier_data)
        top_class = max(scores, key=lambda k: scores[k])
        top_conf = scores[top_class]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        if show_scores:
            parts = sorted(scores.items(), key=lambda x: -x[1])
            score_str = " | ".join(f"{k}: {v*100:.1f}%" for k, v in parts)
            print(f"  [{timestamp}] {basename} | RMS={rms_val:.4f} | {score_str}")

        if top_class == "background":
            os.remove(temp_path)
            stats["discarded"] += 1
            return False

        # Non-background: keep or mark uncertain
        if top_conf >= threshold:
            class_dir = os.path.join(capture_dir, top_class)
        else:
            class_dir = os.path.join(capture_dir, "uncertain")

        os.makedirs(class_dir, exist_ok=True)
        dest_path = os.path.join(
            class_dir,
            f"{timestamp}_{top_class}_{top_conf:.3f}.wav",
        )
        shutil.move(temp_path, dest_path)

        log_entry = {
            "timestamp": timestamp,
            "source_file": basename,
            "top_class": top_class,
            "confidence": round(top_conf, 4),
            "rms": round(rms_val, 6),
            "saved_as": os.path.basename(dest_path),
            "all_scores": {k: round(v, 4) for k, v in scores.items()},
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        stats["kept"] += 1
        if show_scores:
            print(f"  ✓ KEPT as {top_class} ({top_conf*100:.1f}%)")
        return True

    except Exception as e:
        stats["errors"] += 1
        print(f"  ✗ Error processing {basename}: {e}", file=sys.stderr)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False


# ─────────────────────────────────────────────
# Inotify watcher
# ─────────────────────────────────────────────


def watch_streamdata() -> None:
    """Watch StreamData/ for new WAVs via inotifywait."""
    global running
    print(f"Watching {STREAMDATA_DIR} for WAVs...")
    sys.stdout.flush()

    proc = subprocess.Popen(
        ["inotifywait", "-m", "-e", "close_write", "--format", "%f", STREAMDATA_DIR],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if not running:
                break
            filename = line.strip()
            if not filename.endswith(".wav"):
                continue
            wav_path = os.path.join(STREAMDATA_DIR, filename)
            time.sleep(0.3)
            process_wav(wav_path)
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


# ─────────────────────────────────────────────
# Health checks
# ─────────────────────────────────────────────


def check_pi_services() -> bool:
    """Verify BirdNET-Pi's core services are running."""
    all_ok = True
    for s in PI_SERVICES:
        try:
            result = subprocess.run(
                ["systemctl", "is-active", s],
                capture_output=True, text=True, timeout=5,
            )
            status = result.stdout.strip()
            if status != "active":
                print(f"  ⚠ {s}: {status}")
                all_ok = False
            else:
                print(f"  ✓ {s}: active")
        except Exception as e:
            print(f"  ⚠ {s}: check failed ({e})")
            all_ok = False
    return all_ok


def check_pi_resources() -> None:
    """Check disk and inotify processes on the Pi."""
    try:
        result = subprocess.run(
            ["df", "-h", "/"],
            capture_output=True, text=True, timeout=5,
        )
        disk_line = result.stdout.strip().split("\n")[-1]
        parts = disk_line.split()
        if len(parts) >= 5:
            print(f"  Disk: {parts[-2]} free ({parts[-3]} used)")

        result = subprocess.run(
            ["pgrep", "-a", "inotifywait"],
            capture_output=True, text=True, timeout=5,
        )
        inotify_lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        for line in inotify_lines:
            print(f"  Inotify: {line}")
        if not inotify_lines:
            print("  Inotify: (none)")
    except Exception as e:
        print(f"  Resource check failed: {e}")


# ─────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────


def cleanup_temp() -> None:
    """Remove leftover temp files."""
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
        os.makedirs(temp_dir, exist_ok=True)
        print("  Temp files cleaned")


def print_summary() -> None:
    """Print session summary with per-class breakdown."""
    print(f"\n{'='*50}")
    print(f"  Session Summary")
    print(f"{'='*50}")
    print(f"  WAVs processed:  {stats['processed']}")
    print(f"  Detections kept: {stats['kept']}")
    print(f"  Discarded:       {stats['discarded']}")
    print(f"  Errors:          {stats['errors']}")

    if stats["kept"] > 0 and capture_dir and os.path.exists(capture_dir):
        class_counts: dict[str, int] = {}
        for root, dirs, files in os.walk(capture_dir):
            cls_name = os.path.basename(root)
            wavs = [f for f in files if f.endswith(".wav")]
            if wavs:
                class_counts[cls_name] = len(wavs)
        if class_counts:
            print(f"\n  By class:")
            for cls_name, count in sorted(class_counts.items()):
                print(f"    {cls_name}: {count}")
        print(f"\n  Captures: {capture_dir}")
        print(f"  Log file: {log_path}")
    print(f"{'='*50}\n")


def signal_handler(sig, frame) -> None:
    """Handle SIGINT/SIGTERM/SIGHUP gracefully."""
    global running
    if not running:
        return
    print("\n\nShutting down...")
    cleanup_temp()
    print_summary()
    print("Verifying BirdNET-Pi services...")
    all_ok = check_pi_services()
    if all_ok:
        print("\nBirdNET-Pi is healthy. Sidecar removed cleanly.")
    else:
        print("\n⚠ Some BirdNET-Pi services may need attention.")
    sys.exit(0)


# ─────────────────────────────────────────────
# Pull mode (Athena)
# ─────────────────────────────────────────────


def pull_captures() -> str:
    """Pull captures from BirdNET-Pi to the current directory (run on Athena).

    Uses SSH_ASKPASS for password auth.
    """
    askpass = "/tmp/bn_askpass.sh"
    if not os.path.exists(askpass):
        with open(askpass, "w") as f:
            f.write("#!/bin/sh\necho 'birdnetpi'\n")
        os.chmod(askpass, 0o755)
        print(f"  Created {askpass}")

    env = os.environ.copy()
    env["DISPLAY"] = ":0"
    env["SSH_ASKPASS"] = askpass
    env["SSH_ASKPASS_REQUIRE"] = "force"

    session_dir = f"insectnet_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(session_dir, exist_ok=True)

    print(f"Pulling captures to {session_dir}/")

    rsync_cmd = [
        "rsync", "-av",
        "-e", "ssh -o StrictHostKeyChecking=accept-new",
        f"{PI_HOST}:~/insectnet_capture/captures/",
        f"{session_dir}/captures/",
    ]
    result = subprocess.run(rsync_cmd, env=env, capture_output=True, text=True, timeout=120)
    if result.stdout:
        for line in result.stdout.split("\n"):
            if line.strip() and not line.startswith("sending") and not line.startswith("sent"):
                print(f"  {line}")
    if result.returncode != 0 and result.stderr:
        print(f"  rsync stderr: {result.stderr.strip()}")

    scp_cmd = [
        "scp", "-o", "StrictHostKeyChecking=accept-new",
        f"{PI_HOST}:~/insectnet_capture/detections.jsonl",
        f"{session_dir}/detections.jsonl",
    ]
    result = subprocess.run(scp_cmd, env=env, capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        print(f"  detections.jsonl ✓")
    else:
        print(f"  No detections log yet (first session?)")

    ccount = 0
    captures_dir = os.path.join(session_dir, "captures")
    if os.path.exists(captures_dir):
        for root, dirs, files in os.walk(captures_dir):
            cls_name = os.path.basename(root)
            wavs = [f for f in files if f.endswith(".wav")]
            if wavs:
                ccount += len(wavs)
                print(f"    {cls_name}: {len(wavs)} WAVs")

    print(f"\nDone. {ccount} WAVs pulled to {session_dir}/")
    return session_dir


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="InsectNet Capture Sidecar — watch StreamData/, classify, keep detections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--streamdata", default=DEFAULT_STREAMDATA,
                        help=f"BirdNET StreamData dir (default: {DEFAULT_STREAMDATA})")
    parser.add_argument("--birdnet-model", default=DEFAULT_BIRDNET_MODEL,
                        help=f"BirdNET TFLite model path (default: {DEFAULT_BIRDNET_MODEL})")
    parser.add_argument("--pi-host", default=DEFAULT_PI_HOST,
                        help=f"BirdNET-Pi SSH host (default: {DEFAULT_PI_HOST})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Min confidence for non-background (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--noise-floor", type=float, default=DEFAULT_NOISE_FLOOR,
                        help=f"RMS floor to skip inference (default: {DEFAULT_NOISE_FLOOR})")
    parser.add_argument("--capture-dir", default=None,
                        help="Output directory (default: ~/insectnet_capture)")
    parser.add_argument("--show", action="store_true",
                        help="Print per-class scores for every WAV")
    parser.add_argument("--pull", action="store_true",
                        help="Pull captures from Pi to current dir (run on workstation)")
    return parser


def main() -> None:
    global interpreter, classifier_data, capture_dir, log_path, temp_dir
    global show_scores, threshold, noise_floor, running
    global STREAMDATA_DIR, BIRDNET_MODEL_PATH, PI_HOST

    parser = build_parser()
    args = parser.parse_args()

    # Apply overrides
    STREAMDATA_DIR = args.streamdata
    BIRDNET_MODEL_PATH = args.birdnet_model
    PI_HOST = args.pi_host

    # ── Pull mode (workstation) ──
    if args.pull:
        pull_captures()
        return

    # ── Capture mode (Pi) ──
    threshold = args.threshold
    noise_floor = args.noise_floor
    show_scores = args.show

    base_dir = args.capture_dir or os.path.expanduser(DEFAULT_CAPTURE_DIR)
    capture_dir = os.path.join(base_dir, "captures")
    log_path = os.path.join(base_dir, "detections.jsonl")
    temp_dir = os.path.join(base_dir, "tmp")

    os.makedirs(capture_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # ── Pre-flight ──
    print(f"{'='*50}")
    print(f"  InsectNet Capture Sidecar")
    print(f"{'='*50}")
    print(f"\nEnvironment:")
    print(f"  StreamData:  {STREAMDATA_DIR}")
    print(f"  Captures:    {capture_dir}")
    print(f"  Log:         {log_path}")
    print(f"  Threshold:   {threshold}")
    print(f"  Noise floor: {noise_floor}")
    print(f"  Show scores: {show_scores}")

    print(f"\nBirdNET-Pi services:")
    services_ok = check_pi_services()
    if not services_ok:
        print("\n⚠ BirdNET-Pi services issue! Sidecar can still run but check the Pi.")
    check_pi_resources()

    # ── Load TFLite ──
    print(f"\nLoading BirdNET TFLite model...")
    if not os.path.exists(BIRDNET_MODEL_PATH):
        print(f"ERROR: Model not found at {BIRDNET_MODEL_PATH}")
        sys.exit(1)
    interpreter = birdnet.load_tflite(BIRDNET_MODEL_PATH)
    mb = os.path.getsize(BIRDNET_MODEL_PATH) // 1024
    print(f"  ✓ Model loaded ({mb} KB)")

    # ── Load classifier ──
    clf_path = os.path.join(base_dir, "classifier.joblib")
    if not os.path.exists(clf_path):
        print(f"ERROR: Classifier not found at {clf_path}")
        print(f"  Deploy your model: scp models/6class.joblib {PI_HOST}:{clf_path}")
        sys.exit(1)
    classifier_data = load_classifier(clf_path)
    classes_str = ", ".join(classifier_data["classes"])
    print(f"  ✓ Classifier loaded ({len(classifier_data['classes'])} classes: {classes_str})")

    # ── Signal handlers ──
    running = True
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGHUP, signal_handler)

    # ── Start ──
    print(f"\nReady. Watching for WAVs... (Ctrl+C to stop)")
    print(f"{'='*50}\n")
    sys.stdout.flush()
    watch_streamdata()


if __name__ == "__main__":
    main()
