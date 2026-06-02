"""BirdNET TFLite inference wrapper.

Loads a BirdNET TFLite model and runs 3-second audio windows through it,
returning the 6,522-dim logit vector for downstream classification.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

SAMPLE_RATE = 48000
WINDOW_SIZE = 144000  # 3 seconds at 48 kHz


def load_tflite(model_path: str | Path, num_threads: int = 2):
    """Load the BirdNET TFLite model. Returns interpreter."""
    import tflite_runtime.interpreter as tfl

    interp = tfl.Interpreter(
        model_path=str(model_path),
        num_threads=num_threads,
    )
    interp.allocate_tensors()
    return interp


def extract_logits(audio: np.ndarray, interpreter) -> np.ndarray:
    """Run audio through BirdNET and return the 6,522-dim logit vector.

    Args:
        audio: 1D float32 array at 48 kHz.
        interpreter: Loaded TFLite interpreter from load_tflite().

    Returns:
        6,522-dim float64 logit vector.
    """
    target = WINDOW_SIZE
    if len(audio) > target:
        start = len(audio) // 2 - target // 2
        window = audio[start : start + target]
    else:
        window = np.pad(audio, (0, target - len(audio)))

    inp_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]
    interpreter.set_tensor(inp_idx, np.array([window.astype(np.float32)]))
    interpreter.invoke()
    return interpreter.get_tensor(out_idx)[0].astype(np.float64)


def load_audio(path: str | Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a WAV file as mono float32 at target sample rate."""
    import librosa

    audio, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return audio


def rms(audio: np.ndarray) -> float:
    """Root-mean-square amplitude of an audio signal."""
    return float(math.sqrt(np.mean(audio.astype(np.float64) ** 2)))
