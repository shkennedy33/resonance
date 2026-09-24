"""Sessions — a timeline of scenes the engine glides through.

A session file (<repo>/sessions/<name>.json):

    {
      "name": "descent",
      "note": "free text",
      "scenes": [
        {"at": "0:00",                "preset": "alpha-halo",  "label": "settle"},
        {"at": "3:00", "glide": "2:00", "preset": "theta-orbit", "label": "descend",
         "globals": {"noise": 0.25}},
        {"at": "9:00", "glide": 60, "voices": [ {...voice spec...} ]}
      ],
      "end": "20:00",
      "fade_out": 30
    }

- `at`     when the scene starts gliding in (seconds or "m:ss")
- `glide`  how long the glide takes (default 20 s; first scene is usually 0)
- a scene's sound is a `preset` name, inline `voices`, or both (inline wins);
  `globals` overrides are layered on top
- `end` + `fade_out`: master volume fades to silence, then the session is done

Between glides the sequencer writes nothing, so live knob tweaks stick until
the next scene arrives — perform over the script.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .presets import ROOT, load_preset, normalize_state, capture

SESSION_DIR = ROOT / "sessions"
DEFAULT_GLIDE = 20.0


def parse_time(v):
    """Seconds from 90, 90.0, "90", "1:30" or "1:02:03"."""
    if isinstance(v, (int, float)):
        return float(v)
    parts = [float(p) for p in str(v).strip().split(":")]
    t = 0.0
    for p in parts:
        t = t * 60 + p
    return t


def fmt_time(t):
    t = max(0, int(t))
    return f"{t // 60}:{t % 60:02d}"


class Scene:
    def __init__(self, at, glide, state, label):
        self.at, self.glide, self.state, self.label = at, glide, state, label


class Session:
    def __init__(self, name, scenes, end, fade_out=20.0, note=""):
        self.name = name
        self.scenes = sorted(scenes, key=lambda s: s.at)
        self.end = end
        self.fade_out = fade_out
        self.note = note
        self.reset()

    @property
    def length(self):
        return self.end + self.fade_out

    def reset(self):
        self._next = 0
        self._fading = False

    def scene_label(self, t):
        cur = None
        for i, s in enumerate(self.scenes):
            if s.at <= t:
                cur = (i, s)
        if t >= self.end:
            return "fade out"
        if cur is None:
            return ""
        i, s = cur
        gliding = t < s.at + s.glide and s.glide > 0
        return f"{i+1}/{len(self.scenes)} {s.label}" + (" (gliding)" if gliding else "")

    def due(self, t, eng):
        """Morphs to launch at time t: yields (state, glide)."""
        while self._next < len(self.scenes) and self.scenes[self._next].at <= t:
            s = self.scenes[self._next]
            self._next += 1
            # if several scenes are already due (seek/late start), only the last matters
            if self._next < len(self.scenes) and self.scenes[self._next].at <= t:
                continue
            yield s.state, s.glide
        if t >= self.end and not self._fading:
            self._fading = True
            state = capture(eng)
            state["globals"]["volume"] = 0.0
            yield normalize_state(state), self.fade_out

    def finished(self, t):
        return t >= self.length + 0.3


# ---- loading -------------------------------------------------------------- #
def list_sessions():
    return sorted(p.stem for p in SESSION_DIR.glob("*.json"))


def session_path(name):
    p = Path(name)
    if p.suffix == ".json" and p.exists():
        return p
    return SESSION_DIR / f"{name}.json"


def _scene_state(raw):
    if "preset" in raw:
        state = load_preset(raw["preset"])
    else:
        state = {"voices": [], "globals": {}}
    if "voices" in raw:
        state["voices"] = raw["voices"]
    g = dict(state.get("globals") or {})
    g.update(raw.get("globals") or {})
    state["globals"] = g
    return normalize_state(state)


def load_session(name):
    p = session_path(name)
    if not p.exists():
        raise FileNotFoundError(f"no session {name!r} (have: {', '.join(list_sessions())})")
    raw = json.loads(p.read_text())
    scenes = []
    for i, r in enumerate(raw["scenes"]):
        try:
            state = _scene_state(r)
        except (ValueError, FileNotFoundError) as e:
            raise ValueError(f"{p.name} scene {i+1}: {e}") from None
        scenes.append(Scene(parse_time(r.get("at", 0)),
                            parse_time(r.get("glide", 0 if i == 0 else DEFAULT_GLIDE)),
                            state, r.get("label", r.get("preset", f"scene {i+1}"))))
    if not scenes:
        raise ValueError(f"{p.name}: no scenes")
    last = scenes[-1]
    end = parse_time(raw["end"]) if "end" in raw else last.at + last.glide + 300
    return Session(raw.get("name", p.stem), scenes, end,
                   parse_time(raw.get("fade_out", 20)), raw.get("note", ""))


# ---- offline render ------------------------------------------------------- #
def render_session(session, out_path=None, progress=None):
    """Render a whole session to (L, R) by driving the real engine offline —
    the file is sample-identical in behavior to what you'd hear live."""
    from .engine import Engine, BLOCK
    from .core import write

    eng = Engine(device=-1)          # never opens a stream
    eng.play_session(session)
    eng.gcur["volume"] = 0.0         # same soft attack as live start()
    total = int(np.ceil((session.length + 1.0) * eng.sr / BLOCK)) * BLOCK
    out = np.zeros((total, 2), dtype=np.float32)
    buf = np.zeros((BLOCK, 2), dtype=np.float32)
    i = 0
    while i < total and not eng.session_done:
        eng._callback(buf, BLOCK, None, None)
        out[i:i + BLOCK] = buf
        i += BLOCK
        if progress and (i // BLOCK) % 2000 == 0:
            progress(eng.session_t, session.length)
    L, R = out[:i, 0].astype(np.float64), out[:i, 1].astype(np.float64)
    if out_path:
        write(out_path, L, R, sr=eng.sr)
    return L, R
