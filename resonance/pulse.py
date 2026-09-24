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

`nest` groups the strikes into bursts, one per path cycle, centred on `hit_at`:
theta-gamma nesting (phase-amplitude coupling, as in the hippocampus). A 6.67 Hz
orbit x 6 hits with nest 0.8 = 40 Hz strikes clustered into 6.67 Hz bursts.

`click` crossfades each strike's timbre from the struck carrier tone to a
broadband noise burst (a real click, closer to the GENUS stimulus).

Envelope per strike (period T): 1 ms raised-cosine attack, then exponential
decay (`decay_ms`). The attack starts from the previous strike's residual, so
the envelope is continuous even when the decay is longer than the period.
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2 * np.pi
ATTACK_S = 0.001
MAX_MAKEUP = 1.8        # loudness compensation cap (keeps peaks sane)
CLICK_STD = 0.45        # noise-burst level relative to a unit sine (by ear, ~equal loudness)
_GRID = np.linspace(0.0, 1.0, 512, endpoint=False)


def _shape(u, T, tau):
    """Envelope at fractional strike position u in [0, 1)."""
    a = min(ATTACK_S, 0.3 * T)
    t = u * T
    r = np.exp(-(T - a) / tau)                       # residual at end of period
    rise = r + (1.0 - r) * 0.5 * (1.0 - np.cos(np.pi * np.minimum(t, a) / a))
    fall = np.exp(-np.maximum(t - a, 0.0) / tau)
    return np.where(t < a, rise, fall)


def _nest_window(phi, nest):
    """Burst window over one path cycle (phi = phase from hit point, rad)."""
    if nest <= 1e-4:
        return 1.0
    return (1.0 - nest) + nest * (0.5 * (1.0 + np.cos(phi))) ** 2


def strikes(mph, f_mod, pulse, decay_ms, hits=1, hit_at_deg=0.0, nest=0.0):
    """For mod-phase array `mph` (radians, as the synth accumulates it) return
    (tone_gain, click_env), loudness-compensated. tone_gain multiplies the
    carrier (1 - pulse of it stays continuous); click_env is the pure struck
    envelope for the noise burst. Returns None when pulse is 0 (no work)."""
    if pulse <= 1e-4:
        return None
    hits = max(1, int(round(hits)))
    T = 1.0 / max(f_mod * hits, 1e-6)
    tau = max(decay_ms, 0.2) / 1000.0
    phi = mph - np.deg2rad(hit_at_deg)
    # burst window peaks on the strike at the hit point, symmetric around it
    struck = _shape((phi * hits / TWO_PI) % 1.0, T, tau) * _nest_window(phi, nest)
    ref = pulse * _shape((_GRID * hits) % 1.0, T, tau) * _nest_window(TWO_PI * _GRID, nest)
    makeup = min(1.0 / max(np.sqrt(np.mean(((1.0 - pulse) + ref) ** 2)), 1e-6),
                 MAX_MAKEUP)
    return ((1.0 - pulse) + pulse * struck) * makeup, pulse * struck * makeup


def strike_gain(mph, f_mod, pulse, decay_ms, hits=1, hit_at_deg=0.0, nest=0.0):
    """Tone gain only (back-compat helper)."""
    s = strikes(mph, f_mod, pulse, decay_ms, hits, hit_at_deg, nest)
    return None if s is None else s[0]


def make_shaper(p, rng, amp):
    """Source shaper for one voice: fn(mph, mono) -> mono with strikes applied.
    `p` holds f_mod, pulse, decay, hits, hit_at, nest, click."""
    def shape(mph, mono):
        s = strikes(mph, p["f_mod"], p["pulse"], p["decay"], p.get("hits", 1),
                    p.get("hit_at", 0.0), p.get("nest", 0.0))
        if s is None:
            return mono
        tone_g, click_env = s
        c = float(np.clip(p.get("click", 0.0), 0.0, 1.0))
        out = mono * tone_g * (1.0 - c)
        if c > 1e-4:
            out = out + c * amp * CLICK_STD * rng.standard_normal(len(mono)) * click_env
        return out
    return shape
