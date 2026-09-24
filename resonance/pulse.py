"""Percussive strike envelope, phase-locked to a voice's spatial motion.

Pure SAM (classic phase mode) has a perfectly flat envelope in each ear: all of
its f_mod energy is interaural, carried only by binaural processing. The
strongest known 40 Hz brain response — the auditory steady-state response — is
driven by the *envelope*, hardest by sharp onsets (click trains beat smooth AM).
The 40 Hz GENUS work (Martorell et al. 2019) used 1 ms click trains.

So each voice can be *struck*: `hits` strikes per trip around its path, landing
at `hit_at` degrees along it. The rhythm then drives the envelope pathway while
the motion drives the binaural one. hits > 1 decouples them: e.g. a 10 Hz orbit
(slow enough to hear as motion) with 4 hits = a 40 Hz strike train walking the
four compass points around your head.

Envelope per strike (period T): 1 ms raised-cosine attack, then exponential
decay (`decay_ms`). The attack starts from the previous strike's residual, so
the envelope is continuous even when the decay is longer than the period.
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2 * np.pi
ATTACK_S = 0.001
MAX_MAKEUP = 1.8        # loudness compensation cap (keeps peaks sane)
_GRID = np.linspace(0.0, 1.0, 256, endpoint=False)


def _shape(u, T, tau):
    """Envelope at fractional strike position u in [0, 1)."""
    a = min(ATTACK_S, 0.3 * T)
    t = u * T
    r = np.exp(-(T - a) / tau)                       # residual at end of period
    rise = r + (1.0 - r) * 0.5 * (1.0 - np.cos(np.pi * np.minimum(t, a) / a))
    fall = np.exp(-np.maximum(t - a, 0.0) / tau)
    return np.where(t < a, rise, fall)


def strike_gain(mph, f_mod, pulse, decay_ms, hits=1, hit_at_deg=0.0):
    """Gain array for mod-phase array `mph` (radians, as the synth accumulates it).
    pulse 0 = continuous tone (returns None: no work), 1 = fully struck."""
    if pulse <= 1e-4:
        return None
    hits = max(1, int(round(hits)))
    T = 1.0 / max(f_mod * hits, 1e-6)
    tau = max(decay_ms, 0.2) / 1000.0
    u = ((mph - np.deg2rad(hit_at_deg)) * hits / TWO_PI) % 1.0
    g = (1.0 - pulse) + pulse * _shape(u, T, tau)
    ref = (1.0 - pulse) + pulse * _shape(_GRID, T, tau)
    makeup = min(1.0 / max(np.sqrt(np.mean(ref ** 2)), 1e-6), MAX_MAKEUP)
    return g * makeup
