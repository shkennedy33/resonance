"""Core DSP primitives: timelines, fades, mixing, normalization, file I/O.

Conventions
-----------
* A signal is a 1-D float64 numpy array, nominal range [-1, 1].
* Stereo is represented two ways:
    - as a (L, R) tuple of 1-D arrays  (what generators return / consume)
    - as an (N, 2) array               (what we hand to soundfile)
  Use `stereo(L, R)` to go tuple -> (N, 2).
* Sample rate defaults to 48 kHz (PipeWire native on this box).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import soundfile as sf

SR = 48_000


def seconds(dur: float, sr: int = SR) -> int:
    """Number of samples in `dur` seconds."""
    return int(round(dur * sr))


def timeline(dur: float, sr: int = SR) -> np.ndarray:
    """Time axis in seconds, shape (N,)."""
    return np.arange(seconds(dur, sr), dtype=np.float64) / sr


def fade(x: np.ndarray, ms: float = 40.0, sr: int = SR) -> np.ndarray:
    """Equal-power (cosine) fade in and out on a 1-D signal, in place-safe copy."""
    x = np.asarray(x, dtype=np.float64).copy()
    n = int(round(ms / 1000.0 * sr))
    n = min(n, x.shape[0] // 2)
    if n <= 0:
        return x
    ramp = np.sin(np.linspace(0.0, np.pi / 2, n)) ** 2  # smooth 0->1
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x


def stereo(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Stack two 1-D channels into an (N, 2) array (pads the shorter with zeros)."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    n = max(left.shape[0], right.shape[0])
    out = np.zeros((n, 2), dtype=np.float64)
    out[: left.shape[0], 0] = left
    out[: right.shape[0], 1] = right
    return out


def mix(*layers, gains=None):
    """Sum any number of (L, R) tuples into one (L, R) tuple.

    layers : each a (L, R) tuple of 1-D arrays (lengths may differ; padded).
    gains  : optional per-layer linear gains (default 1.0 each).
    """
    if gains is None:
        gains = [1.0] * len(layers)
    n = max(max(len(L), len(R)) for (L, R) in layers)
    accL = np.zeros(n, dtype=np.float64)
    accR = np.zeros(n, dtype=np.float64)
    for (L, R), g in zip(layers, gains):
        accL[: len(L)] += g * np.asarray(L, dtype=np.float64)
        accR[: len(R)] += g * np.asarray(R, dtype=np.float64)
    return accL, accR


def normalize(L: np.ndarray, R: np.ndarray, peak: float = 0.89):
    """Scale a stereo pair so the largest absolute sample hits `peak`."""
    L = np.asarray(L, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    m = max(np.max(np.abs(L)) if L.size else 0.0,
            np.max(np.abs(R)) if R.size else 0.0)
    if m <= 0:
        return L, R
    g = peak / m
    return L * g, R * g


def write(path: str | Path, L: np.ndarray, R: np.ndarray,
          sr: int = SR, subtype: str = "PCM_24") -> Path:
    """Write a stereo (L, R) pair to disk. Format inferred from extension.

    subtype PCM_24 = 24-bit WAV (good default). Use 'FLOAT' for 32-bit float,
    or write a .flac path for lossless-compressed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = stereo(L, R)
    if path.suffix.lower() == ".flac":
        subtype = "PCM_24"
    sf.write(str(path), data, sr, subtype=subtype)
    return path
