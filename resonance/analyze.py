"""Verification & visualization.

The whole point of building SAM ourselves is that we can *look inside it*.
`analyze_sam` produces a 4-panel figure that proves a rendered SAM signal is
doing exactly what patent US20130010967A1 describes:

  1. Waveform zoom (L vs R)  — see the two channels drift in/out of phase.
  2. Interaural phase difference over time — should be a clean sinusoid at f_m
     with peak amplitude 2*arc_deg. THIS is the spatial swing, measured.
  3. Magnitude spectrum of one channel — phase modulation produces Bessel
     sidebands at f_s +/- n*f_m. Their presence proves it's genuine PM.
  4. Mono-sum envelope — (L+R)/2 = 2A*sin(carrier)*cos(swing), i.e. an
     amplitude 'tremolo' at 2*f_m. This is the effect the patent names.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from .core import SR


def interaural_phase(L: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Instantaneous interaural phase difference (radians) via analytic signal.
    Carrier term cancels in the subtraction, leaving the modulation."""
    phL = np.unwrap(np.angle(hilbert(L)))
    phR = np.unwrap(np.angle(hilbert(R)))
    return phL - phR


def analyze_sam(L: np.ndarray, R: np.ndarray, *,
                f_carrier: float, f_mod: float, arc_deg: float,
                sr: int = SR, title: str = "SAM analysis",
                path: str | Path = "out/sam_analysis.png") -> Path:
    t = np.arange(len(L)) / sr

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # --- 1. waveform zoom: a few carrier cycles near a moment of max swing ---
    # center the zoom where sin(2*pi*f_mod*t)=1  => t = 1/(4 f_mod)
    tc = 1.0 / (4 * f_mod)
    i0 = int(tc * sr)
    span = int(sr / f_carrier * 6)  # ~6 carrier cycles
    sl = slice(max(0, i0 - span // 2), i0 + span // 2)
    ax[0, 0].plot(t[sl] * 1000, L[sl], label="L", lw=1.4)
    ax[0, 0].plot(t[sl] * 1000, R[sl], label="R", lw=1.4, alpha=0.8)
    ax[0, 0].set_title("1. Waveform (zoom near peak swing) — channels phase-shifted")
    ax[0, 0].set_xlabel("ms"); ax[0, 0].set_ylabel("amp"); ax[0, 0].legend(loc="upper right")

    # --- 2. interaural phase difference over time (zoom to ~5 cycles of f_m) ---
    ipd_deg = np.rad2deg(interaural_phase(L, R))
    predicted = 2 * arc_deg
    edge = int(0.2 * sr)
    measured = (np.max(ipd_deg[edge:-edge]) - np.min(ipd_deg[edge:-edge])) / 2
    z0 = edge
    z1 = z0 + int(5 * sr / f_mod)  # 5 cycles of the spatial swing
    ax[0, 1].plot(t[z0:z1], ipd_deg[z0:z1], color="#c0392b", lw=1.6)
    ax[0, 1].axhline(+predicted, ls="--", c="gray", lw=1)
    ax[0, 1].axhline(-predicted, ls="--", c="gray", lw=1)
    ax[0, 1].set_title(f"2. Interaural phase diff — sound swings side-to-side at "
                       f"f_m={f_mod} Hz\npredicted ±{predicted:.0f}°, measured ±{measured:.0f}° "
                       f"(5 cycles shown)")
    ax[0, 1].set_xlabel("s"); ax[0, 1].set_ylabel("IPD (deg)")

    # --- 3. spectrum with Bessel sidebands ---
    win = np.hanning(len(L))
    spec = np.abs(np.fft.rfft(L * win))
    freqs = np.fft.rfftfreq(len(L), 1 / sr)
    spec_db = 20 * np.log10(spec / (np.max(spec) + 1e-12) + 1e-12)
    lo, hi = max(0, f_carrier - 6 * f_mod - 20), f_carrier + 6 * f_mod + 20
    m = (freqs >= lo) & (freqs <= hi)
    ax[1, 0].plot(freqs[m], spec_db[m], color="#2c3e50", lw=1.0)
    for n in range(-5, 6):
        fx = f_carrier + n * f_mod
        if lo <= fx <= hi:
            ax[1, 0].axvline(fx, ls=":", c="#27ae60", lw=0.8, alpha=0.7)
    ax[1, 0].set_ylim(-80, 3)
    ax[1, 0].set_title(f"3. Spectrum of L — carrier {f_carrier} Hz + PM sidebands "
                       f"every {f_mod} Hz (green)")
    ax[1, 0].set_xlabel("Hz"); ax[1, 0].set_ylabel("dB")

    # --- 4. mono-sum tremolo envelope (zoom to ~5 cycles of f_m) ---
    mono = 0.5 * (L + R)
    env = np.abs(hilbert(mono))
    ax[1, 1].plot(t[z0:z1], mono[z0:z1], color="#bdc3c7", lw=0.6, label="(L+R)/2")
    ax[1, 1].plot(t[z0:z1], env[z0:z1], color="#8e44ad", lw=2.0, label="envelope")
    ax[1, 1].set_title("4. Mono-sum 'tremolo' — amplitude pulses at 2·f_m "
                       "(the effect the patent names)")
    ax[1, 1].set_xlabel("s"); ax[1, 1].set_ylabel("amp"); ax[1, 1].legend(loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=110)
    plt.close(fig)
    return path


def spectrum(x: np.ndarray, *, sr: int = SR, fmax: float = 1000.0,
             title: str = "spectrum", path: str | Path = "out/spectrum.png") -> Path:
    """Quick single-channel magnitude spectrum, for sanity checks."""
    win = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    spec_db = 20 * np.log10(spec / (np.max(spec) + 1e-12) + 1e-12)
    m = freqs <= fmax
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs[m], spec_db[m], lw=1.0)
    ax.set_title(title); ax.set_xlabel("Hz"); ax.set_ylabel("dB"); ax.set_ylim(-90, 3)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=110)
    plt.close(fig)
    return path
