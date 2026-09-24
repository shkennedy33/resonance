"""Spherical-head binaural spatialization.

Instead of faking spatial position with a raw phase offset (the first SAM
engine's trick, which wraps/ambiguates past ~90 deg), this places a real moving
point source using the two physical localization cues:

  ITD  interaural TIME difference  — true sub-millisecond delay between ears,
       realized with fractional delay lines. Time-VARYING delay at f_mod IS what
       produces the phase-modulation sidebands, so this is a more honest SAM and
       it has no phase-wrap ceiling — the source can circle fully around you.

  ILD  interaural LEVEL difference — head-shadow, which grows with frequency.
       At SAM's low carriers (300 Hz) it's physically small; `shadow` lets you
       exaggerate it for a stronger sense of motion.

Model: Woodworth spherical-head ITD, head radius 8.75 cm, c = 343 m/s.

Two entry points:
  render_path(...)   offline -> (L, R) arrays
  Spatializer        stateful, block-by-block, for the real-time engine
"""
from __future__ import annotations

import numpy as np

from .core import SR, fade
from .paths import get_path, TWO_PI

C = 343.0          # speed of sound, m/s
HEAD_R = 0.0875    # effective head radius, m
_D0 = 0.0016       # base delay (s) so per-ear delay stays positive (> max ITD/2)


def _lateral(az, el):
    """Collapse (az, el) to the effective left-right angle that sets ITD/ILD."""
    return np.arcsin(np.clip(np.sin(az) * np.cos(el), -1.0, 1.0))


def woodworth_itd(az, el=0.0):
    """Interaural time difference (seconds). + => source to the RIGHT (right ear
    leads / shorter delay). Woodworth ray model: (r/c)(theta + sin theta)."""
    lat = _lateral(az, el)
    return (HEAD_R / C) * (lat + np.sin(lat))


def ild_db(az, el, f_carrier, shadow=1.0):
    """Interaural level difference (dB). + => right ear louder. Head-shadow scales
    with frequency (tiny at low carriers), then multiplied by user `shadow`."""
    lat = _lateral(az, el)
    f_factor = np.clip(f_carrier / 1500.0, 0.0, 1.0)   # ~0 low freq -> 1 by 1.5 kHz
    return shadow * 12.0 * f_factor * np.sin(lat)


def _frac_read(ext, pos):
    """Linear-interpolated read of buffer `ext` at fractional sample positions."""
    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0
    i0 = np.clip(i0, 0, len(ext) - 2)
    return ext[i0] * (1.0 - frac) + ext[i0 + 1] * frac


def _spatialize(mono, az, el, f_carrier, sr, shadow, history=None, itd_gain=1.0):
    """Core: turn a mono source + (az, el) trajectories into (L, R) using ITD+ILD.
    If `history` (last samples of prior mono) is given, delay reads see across the
    block boundary (for real-time continuity). `itd_gain` scales the time delay:
    1.0 = physically honest, >1 = hyper-real (bigger motion than a real head).
    Returns (L, R, new_history)."""
    n = len(mono)
    itd = woodworth_itd(az, el) * itd_gain
    ild = ild_db(az, el, f_carrier, shadow)

    maxd = int(np.ceil((_D0 + 0.0025) * sr)) + 4
    if history is None:
        history = np.zeros(maxd, dtype=np.float64)
    ext = np.concatenate([history[-maxd:], mono])
    base = maxd + np.arange(n)

    dL = np.clip((_D0 + itd / 2.0) * sr, 1.0, maxd - 2)  # right source => left later
    dR = np.clip((_D0 - itd / 2.0) * sr, 1.0, maxd - 2)
    L = _frac_read(ext, base - dL)
    R = _frac_read(ext, base - dR)

    gL = 10.0 ** ((-ild / 2.0) / 20.0)
    gR = 10.0 ** ((+ild / 2.0) / 20.0)
    L *= gL
    R *= gR
    return L, R, ext[-maxd:].copy()


def render_path(path="pendulum", f_carrier=300.0, f_mod=40.0, dur=10.0, *,
                extent=None, orient=0.0, shadow=1.0, distance=0.6, itd_gain=1.0,
                amp=0.6, sr=SR, fade_ms=60.0, path_kw=None):
    """Offline render of a SAM tone moving along a named spatial path.

    path     : name in resonance.paths.PATHS
    orient   : constant azimuth offset (rad) = aim the whole figure left/right
    shadow   : ILD exaggeration (0 = pure ITD/phase, 1 = physical, >1 = stronger)
    itd_gain : time-delay scale (1 = physical, >1 = hyper-real motion)
    distance : source distance (m); scales loudness ~1/distance
    extent   : override the path's angular size (rad)
    """
    n = int(round(dur * sr))
    t = np.arange(n) / sr
    theta = TWO_PI * f_mod * t
    pf = get_path(path)
    kw = dict(path_kw or {})
    if extent is not None:
        kw["extent"] = extent
    az, el = pf(theta, **kw)
    az = az + orient

    mono = amp * np.sin(TWO_PI * f_carrier * t)
    L, R, _ = _spatialize(mono, az, el, f_carrier, sr, shadow, itd_gain=itd_gain)
    g = 0.6 / max(distance, 0.05)
    L *= g
    R *= g
    return fade(L, fade_ms, sr), fade(R, fade_ms, sr)


class Spatializer:
    """Stateful, block-by-block spatializer for the real-time engine. Keeps the
    carrier phase and the delay-line history continuous across blocks."""

    def __init__(self, sr=SR):
        self.sr = sr
        self.cphase = 0.0
        self.mphase = 0.0
        maxd = int(np.ceil((_D0 + 0.0012) * sr)) + 4
        self.history = np.zeros(maxd, dtype=np.float64)

    def process(self, frames, *, f_carrier, f_mod, path_fn, extent,
                orient=0.0, shadow=1.0, itd_gain=1.0, amp=0.6, path_kw=None):
        n = frames
        cinc = TWO_PI * f_carrier / self.sr
        minc = TWO_PI * f_mod / self.sr
        k = np.arange(1, n + 1)
        cph = self.cphase + cinc * k
        mph = self.mphase + minc * k
        self.cphase = float(cph[-1] % TWO_PI)
        self.mphase = float(mph[-1] % TWO_PI)

        kw = dict(path_kw or {})
        kw["extent"] = extent
        az, el = path_fn(mph, **kw)
        az = az + orient

        mono = amp * np.sin(cph)
        L, R, self.history = _spatialize(mono, az, el, f_carrier, self.sr,
                                         shadow, self.history, itd_gain=itd_gain)
        return L, R
