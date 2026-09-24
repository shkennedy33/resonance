"""Signal generators. Every generator returns a (L, R) tuple of 1-D float64 arrays.

The star of the show is `sam()`, a faithful implementation of the Spatial Angle
Modulation equations from Monroe Institute patent US20130010967A1:

    S_L(t) = A * sin[ 2*pi*f_s*t  +  phi_p*sin(2*pi*f_m*t)  +  phi_L ]
    S_R(t) = A * sin[ 2*pi*f_s*t  -  phi_p*sin(2*pi*f_m*t)  +  phi_R ]

where
    f_s   = carrier ("source") frequency, Hz          (patent: 300 or 440)
    f_m   = spatial-oscillation / entrainment freq, Hz (e.g. 40 for gamma)
    phi_p = peak phase deviation, radians              (= arc size)
    phi_L, phi_R = absolute channel phase offsets      (= arc center / aim)
"""
from __future__ import annotations

import numpy as np

from .core import SR, timeline, fade


# --------------------------------------------------------------------------- #
#  Plain tone (building block)                                                 #
# --------------------------------------------------------------------------- #
def tone(freq: float, dur: float, *, amp: float = 0.5, phase: float = 0.0,
         sr: int = SR, fade_ms: float = 40.0):
    """A mono sine, returned as identical (L, R) so it drops into the mixer."""
    t = timeline(dur, sr)
    x = amp * np.sin(2 * np.pi * freq * t + phase)
    x = fade(x, fade_ms, sr)
    return x, x.copy()


# --------------------------------------------------------------------------- #
#  SAM — Spatial Angle Modulation                                             #
# --------------------------------------------------------------------------- #
def sam(f_carrier: float, f_mod: float, dur: float, *,
        arc_deg: float = 75.0,
        mode: str = "phase",
        level_depth: float = 0.35,
        bias_deg: float = 0.0,
        amp: float = 0.5,
        sr: int = SR,
        fade_ms: float = 60.0):
    """Spatial Angle Modulation tone.

    Parameters
    ----------
    f_carrier : carrier frequency f_s in Hz. Keep < ~1000 Hz so interaural
                *phase* is the dominant localization cue (below ~1.5 kHz the
                brain uses phase/time delay; above it, level). 300 Hz is a good
                default per the patent.
    f_mod     : f_m, the spatial-oscillation rate = your entrainment target
                (e.g. 10 alpha, 6 theta, 40 gamma). No 30 Hz ceiling here.
    dur       : seconds.
    arc_deg   : peak phase deviation phi_p in *degrees* (converted to radians).
                Sets how wide the sound swings. The interaural phase difference
                swings +/- 2*phi_p. Keep phi_p <= 90 deg so 2*phi_p <= 180 deg
                and the motion stays unambiguous (no phase wrap-around).
    mode      : "phase"    -> patent-faithful, phase-only (constant level).
                "natural"  -> also pans loudness with position (adds the ILD
                              cue a real moving source has). More convincing
                              motion; `level_depth` sets how much.
                "circular" -> level cue is 90 deg out of phase with the phase
                              cue, so the source traces an ellipse/orbit around
                              the head instead of a straight left-right line.
    level_depth : 0..1, strength of the loudness pan for natural/circular.
    bias_deg  : shifts the arc center left(-)/right(+), i.e. phi_L - phi_R.
                Per the patent this biases which hemisphere is driven harder
                (contralateral pathway). 0 = centered/bilateral.
    amp       : per-channel amplitude (headroom for layering; 0.5 = -6 dBFS).

    Returns
    -------
    (L, R) tuple of 1-D float64 arrays.
    """
    t = timeline(dur, sr)
    phi_p = np.deg2rad(arc_deg)
    bias = np.deg2rad(bias_deg)
    phi_L = +bias / 2.0
    phi_R = -bias / 2.0

    swing = phi_p * np.sin(2 * np.pi * f_mod * t)          # the moving term
    carrier = 2 * np.pi * f_carrier * t

    L = np.sin(carrier + swing + phi_L)
    R = np.sin(carrier - swing + phi_R)

    if mode == "phase":
        gL = gR = 1.0
    elif mode == "natural":
        # loudness tracks position: source-left => left ear louder (in phase)
        pan = level_depth * np.sin(2 * np.pi * f_mod * t)
        gL = 1.0 + pan
        gR = 1.0 - pan
    elif mode == "circular":
        # loudness 90 deg out of phase with the phase swing => elliptical orbit
        pan = level_depth * np.cos(2 * np.pi * f_mod * t)
        gL = 1.0 + pan
        gR = 1.0 - pan
    else:
        raise ValueError(f"unknown SAM mode: {mode!r}")

    L = amp * gL * L
    R = amp * gR * R
    return fade(L, fade_ms, sr), fade(R, fade_ms, sr)


# --------------------------------------------------------------------------- #
#  Binaural beats                                                             #
# --------------------------------------------------------------------------- #
def binaural(f_carrier: float, f_beat: float, dur: float, *,
             amp: float = 0.5, sr: int = SR, fade_ms: float = 60.0):
    """Classic binaural beat: carrier in the left ear, carrier+beat in the right.

    The perceived beat is created in the brainstem (superior olive) and gets
    weak/absent above ~30 Hz — that's the fundamental limit SAM sidesteps.
    """
    t = timeline(dur, sr)
    L = amp * np.sin(2 * np.pi * f_carrier * t)
    R = amp * np.sin(2 * np.pi * (f_carrier + f_beat) * t)
    return fade(L, fade_ms, sr), fade(R, fade_ms, sr)


# --------------------------------------------------------------------------- #
#  Monaural beats                                                             #
# --------------------------------------------------------------------------- #
def monaural(f_carrier: float, f_beat: float, dur: float, *,
             amp: float = 0.5, sr: int = SR, fade_ms: float = 60.0):
    """Monaural beat: the two tones are physically summed (real amplitude
    beating in the air/signal), identical in both ears. Works on speakers."""
    t = timeline(dur, sr)
    x = np.sin(2 * np.pi * f_carrier * t) + np.sin(2 * np.pi * (f_carrier + f_beat) * t)
    x = amp * 0.5 * x
    x = fade(x, fade_ms, sr)
    return x, x.copy()


# --------------------------------------------------------------------------- #
#  Isochronic tones                                                          #
# --------------------------------------------------------------------------- #
def isochronic(f_carrier: float, f_pulse: float, dur: float, *,
               duty: float = 0.5, edge_ms: float = 5.0,
               amp: float = 0.5, sr: int = SR, fade_ms: float = 60.0):
    """Isochronic tone: a carrier switched on/off at `f_pulse`, identical in both
    ears. Strong entrainment, and works without headphones — but hard gating
    sounds harsh, so we use raised-cosine edges (`edge_ms`) to soften clicks.

    duty    : fraction of each cycle the tone is ON (0..1).
    edge_ms : rise/fall time of each pulse edge in ms.
    """
    t = timeline(dur, sr)
    carrier = np.sin(2 * np.pi * f_carrier * t)

    period = 1.0 / f_pulse
    phase = (t % period) / period          # 0..1 within each pulse cycle
    edge = max(edge_ms / 1000.0 / period, 1e-4)   # edge as fraction of cycle

    gate = np.zeros_like(t)
    on = phase < duty
    gate[on] = 1.0
    # raised-cosine rising edge
    rising = phase < edge
    gate[rising] = 0.5 * (1 - np.cos(np.pi * phase[rising] / edge))
    # raised-cosine falling edge (end of the ON region)
    falling = (phase >= duty - edge) & (phase < duty)
    gate[falling] = 0.5 * (1 + np.cos(np.pi * (phase[falling] - (duty - edge)) / edge))

    x = amp * carrier * gate
    x = fade(x, fade_ms, sr)
    return x, x.copy()


# --------------------------------------------------------------------------- #
#  Noise beds                                                                 #
# --------------------------------------------------------------------------- #
def _shaped_noise(dur: float, exponent: float, amp: float, sr: int, seed: int):
    """White noise shaped to 1/f**exponent in the frequency domain.
    exponent=1 -> pink, exponent=2 -> brown/red. Independent L/R for width."""
    n = int(round(dur * sr))
    rng = np.random.default_rng(seed)

    def one():
        white = rng.standard_normal(n)
        spec = np.fft.rfft(white)
        f = np.fft.rfftfreq(n, 1 / sr)
        scale = np.ones_like(f)
        scale[1:] = 1.0 / (f[1:] ** (exponent / 2.0))
        shaped = np.fft.irfft(spec * scale, n=n)
        shaped /= np.max(np.abs(shaped)) + 1e-12
        return amp * shaped

    return one(), one()


def pink_noise(dur: float, *, amp: float = 0.35, sr: int = SR, seed: int = 0,
               fade_ms: float = 200.0):
    """Pink (1/f) noise bed — natural, soothing masker."""
    L, R = _shaped_noise(dur, 1.0, amp, sr, seed)
    return fade(L, fade_ms, sr), fade(R, fade_ms, sr)


def brown_noise(dur: float, *, amp: float = 0.4, sr: int = SR, seed: int = 0,
                fade_ms: float = 200.0):
    """Brown/red (1/f^2) noise bed — deeper, rumblier, surf-like."""
    L, R = _shaped_noise(dur, 2.0, amp, sr, seed)
    return fade(L, fade_ms, sr), fade(R, fade_ms, sr)
