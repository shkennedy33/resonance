"""Presets — save/load the whole voice rack as a small JSON file.

A *state* is the full sound of the engine at rest:
    {"voices":  [voice spec, ...],          # same vocabulary as compose.py
     "globals": {"volume": .., "noise": ..}}

A voice spec is {"engine", "path", "mode", "muted", <every VOICE_PARAMS key>}.
Because it matches compose.py's spec vocabulary, any preset can also be rendered
offline with `render_voices(state["voices"])`.

Presets live in <repo>/presets/<name>.json. Hand-editing them is encouraged;
missing keys fall back to the engine defaults.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .engine import (VOICE_PARAMS, GLOBAL_PARAMS, MODES, ENGINE_MODES, SR)
from .paths import PATH_NAMES

ROOT = Path(__file__).resolve().parent.parent
PRESET_DIR = ROOT / "presets"

_VKEYS = [p["key"] for p in VOICE_PARAMS]
_GKEYS = [p["key"] for p in GLOBAL_PARAMS]
VOICE_DEFAULTS = {p["key"]: p["default"] for p in VOICE_PARAMS}
GLOBAL_DEFAULTS = {p["key"]: p["default"] for p in GLOBAL_PARAMS}


# ---- state <-> engine ----------------------------------------------------- #
def voice_spec(v):
    spec = {"engine": v.engine_mode, "path": v.path_name, "mode": v.mode,
            "muted": v.muted}
    spec.update({k: round(float(v.target[k]), 4) for k in _VKEYS})
    return spec


def capture(eng):
    """Snapshot the engine's current *targets* as a state dict."""
    return {"voices": [voice_spec(v) for v in eng.voices if not v.leaving],
            "globals": {k: round(float(eng.gtarget[k]), 4) for k in _GKEYS}}


def normalize_state(state):
    """Fill defaults and validate names, so everything downstream can trust it."""
    voices = []
    for i, s in enumerate(state.get("voices") or []):
        spec = {"engine": s.get("engine", "spatial"),
                "path": s.get("path", PATH_NAMES[0]),
                "mode": s.get("mode", "phase"),
                "muted": bool(s.get("muted", False))}
        for k in _VKEYS:
            spec[k] = float(s.get(k, VOICE_DEFAULTS[k]))
        if spec["engine"] not in ENGINE_MODES:
            raise ValueError(f"voice {i+1}: engine must be one of {ENGINE_MODES}")
        if spec["path"] not in PATH_NAMES:
            raise ValueError(f"voice {i+1}: unknown path {spec['path']!r} "
                             f"(have: {', '.join(PATH_NAMES)})")
        if spec["mode"] not in MODES:
            raise ValueError(f"voice {i+1}: mode must be one of {MODES}")
        voices.append(spec)
    if not voices:
        raise ValueError("a state needs at least one voice")
    g = dict(GLOBAL_DEFAULTS)
    g.update({k: float(v) for k, v in (state.get("globals") or {}).items()
              if k in g})
    return {"voices": voices, "globals": g}


# ---- files ---------------------------------------------------------------- #
def _slug(name):
    s = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip()).strip("-").lower()
    if not s:
        raise ValueError("preset name is empty")
    return s


def list_presets():
    return sorted(p.stem for p in PRESET_DIR.glob("*.json"))


def preset_path(name):
    p = Path(name)
    if p.suffix == ".json" and p.exists():
        return p
    return PRESET_DIR / f"{_slug(name)}.json"


def load_preset(name):
    p = preset_path(name)
    if not p.exists():
        raise FileNotFoundError(f"no preset {name!r} (have: {', '.join(list_presets())})")
    return normalize_state(json.loads(p.read_text()))


def save_preset(eng_or_state, name, note=None):
    state = eng_or_state if isinstance(eng_or_state, dict) else capture(eng_or_state)
    state = normalize_state(state)
    if note:
        state = {"note": note, **state}
    p = preset_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2) + "\n")
    return p


def render_state(state, dur=30.0, sr=SR, **kw):
    """Offline render of a preset via compose.render_voices."""
    from .compose import render_voices
    state = normalize_state(state)
    live = [s for s in state["voices"] if not s["muted"]]
    return render_voices(live, dur=dur, sr=sr, noise=state["globals"]["noise"], **kw)
