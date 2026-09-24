"""Real-time multi-voice SAM synthesis engine.

A phase-continuous, clickless audio engine you can drive live. The audio thread
(PortAudio callback) synthesizes each block from running phase accumulators, so
you can turn any knob mid-tone and it glides instead of clicking.

Structure
---------
    Engine
      ├─ voices: list[Voice]     each an independent SAM generator (its own
      │                          carrier, f_mod, path/mode, aim, gain, mute)
      ├─ globals: master volume + pink-noise bed (shared)
      └─ callback: sum all voices -> + noise -> master vol -> clip

This is the patent's "multiple paths at different frequencies to stimulate
multiple cortical regions" — e.g. a theta voice orbiting left + a gamma voice
darting right, at once.

Each Voice runs one of two synthesis engines:
    "spatial" — geometric ITD/ILD along a parametric path (externalized, real)
    "classic" — phase-offset SAM (exaggerated / hyper-real interaural swing)
"""
from __future__ import annotations

import numpy as np
import sounddevice as sd
from scipy.signal import lfilter, lfilter_zi

from .spatial import Spatializer
from .paths import PATH_NAMES, get_path

SR = 48_000
BLOCK = 1024
MAX_VOICES = 4

MODES = ["phase", "natural", "circular", "figure8"]     # classic arc modes
ENGINE_MODES = ["spatial", "classic"]
BANDS = {"delta": 2.5, "theta": 6.0, "alpha": 10.0, "beta": 20.0, "gamma": 40.0}

# Paul Kellet's refined pink-noise IIR (filters white -> ~ -3 dB/oct)
_PINK_B = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
_PINK_A = np.array([1.0, -2.494956002, 2.017265875, -0.522189400])
_NOISE_MAKEUP = 3.5

TWO_PI = 2 * np.pi

# Per-voice continuous parameters (order = TUI display order)
VOICE_PARAMS = [
    {"key": "carrier",     "label": "carrier f_s",  "min": 40.0,  "max": 1000.0, "step": 5.0,  "unit": "Hz",  "default": 300.0},
    {"key": "f_mod",       "label": "f_mod entrain", "min": 0.5,   "max": 70.0,   "step": 0.5,  "unit": "Hz",  "default": 40.0},
    {"key": "arc",         "label": "arc / size",    "min": 0.0,   "max": 90.0,   "step": 5.0,  "unit": "deg", "default": 75.0},
    {"key": "bias",        "label": "aim (bias)",    "min": -90.0, "max": 90.0,   "step": 5.0,  "unit": "deg", "default": 0.0},
    {"key": "level_depth", "label": "level depth",   "min": 0.0,   "max": 1.0,    "step": 0.05, "unit": "",    "default": 0.35},
    {"key": "itd_gain",    "label": "itd gain",      "min": 0.5,   "max": 3.0,    "step": 0.25, "unit": "x",   "default": 1.5},
    {"key": "shadow",      "label": "ild shadow",    "min": 0.0,   "max": 3.0,    "step": 0.25, "unit": "x",   "default": 1.2},
    {"key": "gain",        "label": "voice gain",    "min": 0.0,   "max": 1.0,    "step": 0.05, "unit": "",    "default": 0.8},
]
_VSPEC = {p["key"]: p for p in VOICE_PARAMS}

# Global (engine-level) params
GLOBAL_PARAMS = [
    {"key": "volume", "label": "master vol", "min": 0.0, "max": 1.0, "step": 0.05, "unit": "", "default": 0.45},
    {"key": "noise",  "label": "pink noise", "min": 0.0, "max": 1.0, "step": 0.05, "unit": "", "default": 0.0},
]
_GSPEC = {p["key"]: p for p in GLOBAL_PARAMS}
GLOBAL_KEYS = set(_GSPEC)


def resolve_device(prefer=("pulse", "pipewire")):
    """Pick an output device that routes through the desktop audio graph."""
    try:
        devs = sd.query_devices()
    except Exception:
        return None
    for name in prefer:
        for i, d in enumerate(devs):
            if name in d["name"].lower() and d["max_output_channels"] >= 2:
                return i
    return None


class Voice:
    """One independent SAM generator with its own phases, params, and gain."""

    def __init__(self, sr=SR, tone_amp=0.6, **overrides):
        self.sr = sr
        self.tone_amp = tone_amp
        self.target = {p["key"]: float(p["default"]) for p in VOICE_PARAMS}
        for k, v in overrides.items():
            if k in self.target:
                self.target[k] = float(v)
        self.cur = dict(self.target)
        self.mode = overrides.get("mode", "phase")
        self.engine_mode = overrides.get("engine_mode", "spatial")
        self.path_name = overrides.get("path_name", PATH_NAMES[0])
        self.spat = Spatializer(sr)
        self.cphase = 0.0
        self.mphase = 0.0
        self.muted = False
        self._mg = 1.0                     # smoothed mute gain (0..1), clickless

    # ---- control ---------------------------------------------------------- #
    def set(self, key, value):
        s = _VSPEC[key]
        self.target[key] = float(np.clip(value, s["min"], s["max"]))

    def nudge(self, key, direction):
        self.set(key, self.target[key] + direction * _VSPEC[key]["step"])

    def cycle_mode(self, d=1):
        self.mode = MODES[(MODES.index(self.mode) + d) % len(MODES)]

    def cycle_path(self, d=1):
        self.path_name = PATH_NAMES[(PATH_NAMES.index(self.path_name) + d) % len(PATH_NAMES)]

    def toggle_engine(self):
        self.engine_mode = ENGINE_MODES[(ENGINE_MODES.index(self.engine_mode) + 1) % len(ENGINE_MODES)]

    def set_band(self, name):
        if name in BANDS:
            self.set("f_mod", BANDS[name])

    def summary(self):
        t = self.target
        return {"carrier": t["carrier"], "f_mod": t["f_mod"], "path": self.path_name,
                "engine": self.engine_mode, "mode": self.mode, "gain": t["gain"],
                "muted": self.muted}

    def detail(self):
        return {"target": dict(self.target), "mode": self.mode,
                "engine_mode": self.engine_mode, "path": self.path_name,
                "muted": self.muted}

    # ---- synthesis -------------------------------------------------------- #
    def synth(self, n, alpha):
        old = dict(self.cur)
        for k in self.cur:
            self.cur[k] += (self.target[k] - self.cur[k]) * alpha
        p = self.cur

        if self.engine_mode == "spatial":
            L, R = self._spatial(n, p)
        else:
            L, R = self._classic(n, p)

        old_mg = self._mg
        self._mg += ((0.0 if self.muted else 1.0) - self._mg) * alpha
        gramp = np.linspace(old["gain"] * old_mg, p["gain"] * self._mg, n)
        return L * gramp, R * gramp

    def _classic(self, n, p):
        k = np.arange(1, n + 1)
        cph = self.cphase + (TWO_PI * p["carrier"] / self.sr) * k
        mph = self.mphase + (TWO_PI * p["f_mod"] / self.sr) * k
        self.cphase = float(cph[-1] % TWO_PI)
        self.mphase = float(mph[-1] % TWO_PI)
        phi_p = np.deg2rad(p["arc"])
        bias = np.deg2rad(p["bias"])
        depth = p["level_depth"]
        if self.mode == "phase":
            pan = 0.0
        elif self.mode == "natural":
            pan = depth * np.sin(mph)
        elif self.mode == "circular":
            pan = depth * np.cos(mph)
        else:
            pan = depth * np.sin(2 * mph)
        swing = phi_p * np.sin(mph)
        L = self.tone_amp * (1.0 + pan) * np.sin(cph + swing + bias / 2)
        R = self.tone_amp * (1.0 - pan) * np.sin(cph - swing - bias / 2)
        return L, R

    def _spatial(self, n, p):
        pf = get_path(self.path_name)
        return self.spat.process(
            n, f_carrier=p["carrier"], f_mod=p["f_mod"], path_fn=pf,
            extent=np.deg2rad(p["arc"]), orient=np.deg2rad(p["bias"]),
            shadow=p["shadow"], itd_gain=p["itd_gain"], amp=self.tone_amp)


class Engine:
    def __init__(self, sr=SR, block=BLOCK, device=None, tone_amp=0.6):
        self.sr = sr
        self.block = block
        self.device = device if device is not None else resolve_device()
        self.tone_amp = tone_amp

        self.voices = [Voice(sr, tone_amp=tone_amp)]
        self.sel = 0
        self.gtarget = {p["key"]: float(p["default"]) for p in GLOBAL_PARAMS}
        self.gcur = dict(self.gtarget)
        self.alpha = self._smoothing_alpha(60.0)

        self.rng = np.random.default_rng()
        self.zi_l = lfilter_zi(_PINK_B, _PINK_A) * 0.0
        self.zi_r = lfilter_zi(_PINK_B, _PINK_A) * 0.0
        self.peak = 0.0
        self.running = False
        self._stream = None

    def _smoothing_alpha(self, ms):
        return float(1.0 - np.exp(-(self.block / self.sr) / (ms / 1000.0)))

    # ---- voice rack management ------------------------------------------- #
    @property
    def selected(self):
        return self.voices[self.sel]

    def add_voice(self):
        if len(self.voices) >= MAX_VOICES:
            return
        # seed a useful contrast: alternate aim, step through bands & paths
        i = len(self.voices)
        bands = list(BANDS.values())
        v = Voice(self.sr, tone_amp=self.tone_amp,
                  f_mod=bands[i % len(bands)],
                  bias=(30.0 if i % 2 else -30.0),
                  path_name=PATH_NAMES[i % len(PATH_NAMES)],
                  gain=0.7)
        self.voices.append(v)
        self.sel = len(self.voices) - 1

    def remove_voice(self):
        if len(self.voices) <= 1:
            return
        self.voices.pop(self.sel)
        self.sel = min(self.sel, len(self.voices) - 1)

    def select(self, direction):
        self.sel = (self.sel + direction) % len(self.voices)

    def toggle_mute(self):
        self.selected.muted = not self.selected.muted

    # ---- control (delegates to selected voice, or globals) --------------- #
    def set(self, key, value):
        if key in GLOBAL_KEYS:
            s = _GSPEC[key]
            self.gtarget[key] = float(np.clip(value, s["min"], s["max"]))
        else:
            self.selected.set(key, value)

    def nudge(self, key, direction):
        if key in GLOBAL_KEYS:
            self.set(key, self.gtarget[key] + direction * _GSPEC[key]["step"])
        else:
            self.selected.nudge(key, direction)

    def nudge_global(self, key, direction):
        self.set(key, self.gtarget[key] + direction * _GSPEC[key]["step"])

    def cycle_mode(self, d=1): self.selected.cycle_mode(d)
    def cycle_path(self, d=1): self.selected.cycle_path(d)
    def toggle_engine(self): self.selected.toggle_engine()
    def set_band(self, name): self.selected.set_band(name)

    # proxies so single-voice callers (and tests) keep working
    @property
    def cur(self): return self.selected.cur
    @cur.setter
    def cur(self, v): self.selected.cur = v
    @property
    def target(self): return self.selected.target
    @target.setter
    def target(self, v): self.selected.target = v
    @property
    def mode(self): return self.selected.mode
    @mode.setter
    def mode(self, v): self.selected.mode = v
    @property
    def engine_mode(self): return self.selected.engine_mode
    @engine_mode.setter
    def engine_mode(self, v): self.selected.engine_mode = v
    @property
    def path_name(self): return self.selected.path_name
    @path_name.setter
    def path_name(self, v): self.selected.path_name = v

    def snapshot(self):
        return {"voices": [v.summary() for v in self.voices], "sel": self.sel,
                "detail": self.selected.detail(), "globals": dict(self.gtarget),
                "peak": self.peak, "running": self.running}

    # ---- audio thread ---------------------------------------------------- #
    def _callback(self, outdata, frames, time_info, status):
        a = self.alpha
        n = frames
        gold = dict(self.gcur)
        for kk in self.gcur:
            self.gcur[kk] += (self.gtarget[kk] - self.gcur[kk]) * a

        accL = np.zeros(n, dtype=np.float64)
        accR = np.zeros(n, dtype=np.float64)
        for v in self.voices:
            L, R = v.synth(n, a)
            accL += L
            accR += R

        if max(gold["noise"], self.gcur["noise"]) > 1e-4:
            nl, self.zi_l = lfilter(_PINK_B, _PINK_A, self.rng.standard_normal(n), zi=self.zi_l)
            nr, self.zi_r = lfilter(_PINK_B, _PINK_A, self.rng.standard_normal(n), zi=self.zi_r)
            nramp = np.linspace(gold["noise"], self.gcur["noise"], n)
            accL += _NOISE_MAKEUP * 0.3 * nramp * nl
            accR += _NOISE_MAKEUP * 0.3 * nramp * nr

        vramp = np.linspace(gold["volume"], self.gcur["volume"], n)
        L = np.ascontiguousarray(accL * vramp)
        R = np.ascontiguousarray(accR * vramp)
        np.clip(L, -1.0, 1.0, out=L)
        np.clip(R, -1.0, 1.0, out=R)
        outdata[:, 0] = L
        outdata[:, 1] = R
        self.peak = float(max(np.abs(L).max(), np.abs(R).max()))

    # ---- lifecycle ------------------------------------------------------- #
    def start(self):
        if self.running:
            return
        self.gcur["volume"] = 0.0   # soft attack
        self._stream = sd.OutputStream(
            samplerate=self.sr, blocksize=self.block, channels=2,
            dtype="float32", device=self.device, callback=self._callback,
            latency="low")
        self._stream.start()
        self.running = True

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.running = False
