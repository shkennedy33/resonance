"""Offline multi-voice composition — render a stack of SAM voices to (L, R).

A *voice spec* is a dict describing one independent SAM generator:
    engine   : "spatial" (geometric ITD/ILD path) or "classic" (phase-offset)
    path     : path name (spatial engine)
    mode     : arc mode phase/natural/circular/figure8 (classic engine)
    carrier  : Hz
    f_mod    : Hz (entrainment target)
    arc      : degrees (spatial: figure size; classic: peak phase deviation)
    bias     : degrees (aim / hemisphere)
    depth    : spatial motion depth, deg of interaural phase swing at 90 deg
               (150 ~ classic engine; pitch-independent). Legacy: itd_gain
    ild      : spatial level cue, dB at 90 deg. Legacy: shadow
    pulse, decay, hits, hit_at : percussive strike layer (see pulse.py)
    gain     : per-voice mix level (0..1)

Example:
    render_voices([
        {"engine":"spatial","path":"orbit","carrier":250,"f_mod":6,"bias":-40,"gain":0.8},
        {"engine":"spatial","path":"spinner","carrier":400,"f_mod":40,"bias":+40,"gain":0.7},
    ], dur=60)
"""
from __future__ import annotations

import numpy as np

from .core import SR, mix, normalize, fade
from .spatial import render_path, itd_gain_for_depth, shadow_for_ild
from .generators import sam as classic_sam, pink_noise
from .core import timeline
from .pulse import strike_gain


def _render_voice(spec, dur, sr):
    engine = spec.get("engine", "spatial")
    f_mod = spec.get("f_mod", 40.0)
    env = lambda mph: strike_gain(mph, f_mod, spec.get("pulse", 0.0),
                                  spec.get("decay", 6.0), spec.get("hits", 1),
                                  spec.get("hit_at", 0.0))
    if engine == "spatial":
        fc = spec.get("carrier", 300.0)
        itd_gain = spec["itd_gain"] if "itd_gain" in spec and "depth" not in spec \
            else itd_gain_for_depth(spec.get("depth", 150.0), fc)
        shadow = spec["shadow"] if "shadow" in spec and "ild" not in spec \
            else shadow_for_ild(spec.get("ild", 6.0), fc)
        L, R = render_path(
            spec.get("path", "pendulum"),
            f_carrier=spec.get("carrier", 300.0),
            f_mod=spec.get("f_mod", 40.0),
            dur=dur,
            extent=np.deg2rad(spec["arc"]) if "arc" in spec else None,
            orient=np.deg2rad(spec.get("bias", 0.0)),
            shadow=shadow, itd_gain=itd_gain,
            amp=0.6, sr=sr, fade_ms=60.0, envelope=env)
    else:  # classic
        L, R = classic_sam(
            spec.get("carrier", 300.0), spec.get("f_mod", 40.0), dur,
            arc_deg=spec.get("arc", 75.0), mode=spec.get("mode", "phase"),
            level_depth=spec.get("level_depth", 0.35),
            bias_deg=spec.get("bias", 0.0), amp=0.6, sr=sr, fade_ms=60.0)
        g = env(2 * np.pi * f_mod * timeline(dur, sr)[: len(L)])
        if g is not None:
            L, R = L * g, R * g
    g = spec.get("gain", 0.8)
    return L * g, R * g


def render_voices(specs, dur=30.0, *, sr=SR, noise=0.0, master=0.9,
                  normalize_out=True):
    """Render and mix a list of voice specs. Returns (L, R)."""
    layers = [_render_voice(s, dur, sr) for s in specs]
    L, R = mix(*layers)
    if noise > 1e-4:
        nL, nR = pink_noise(dur, amp=0.35 * noise, sr=sr)
        L, R = mix((L, R), (nL, nR))
    if normalize_out:
        L, R = normalize(L, R, peak=master)
    return fade(L, 80.0, sr), fade(R, 80.0, sr)
