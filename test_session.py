"""Headless test of presets, morphs and sessions.

Renders the `tour` session through the real engine (offline) and checks:
  - finite, in range, correct total length
  - no clicks anywhere: worst sample-to-sample jump stays near the steady-state max
    (crossfades between different paths/engines are the risky moments)
  - leaving voices are actually removed after each morph (no silent pile-up)
  - preset round-trip: capture -> save -> load gives the same state
  - log-frequency glide: halfway through a 6->40 Hz glide sits near sqrt(6*40)
"""
import json
import numpy as np

from resonance.engine import Engine, BLOCK, SR
from resonance.presets import capture, load_preset, save_preset, normalize_state, PRESET_DIR
from resonance.session import load_session, render_session


def drive(eng, seconds):
    buf = np.zeros((BLOCK, 2), dtype=np.float32)
    for _ in range(int(seconds * SR / BLOCK)):
        eng._callback(buf, BLOCK, None, None)


def main():
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"{name:<34}: {bool(cond)}  {info}")

    # --- tour render ---
    s = load_session("tour")
    L, R = render_session(s)
    dur = len(L) / SR
    check("tour length", abs(dur - s.length) < 1.0, f"({dur:.1f}s vs {s.length:.1f}s)")
    check("finite & in range", np.isfinite(L).all() and np.abs(np.r_[L, R]).max() <= 1.0,
          f"(peak {np.abs(np.r_[L, R]).max():.3f})")
    # click detector: per-second worst |d/dt|, compare transition seconds to median second
    d = np.maximum(np.abs(np.diff(L)), np.abs(np.diff(R)))
    per_sec = d[: (len(d) // SR) * SR].reshape(-1, SR).max(axis=1)
    loud = per_sec[per_sec > 1e-3]
    worst = loud.max() / np.median(loud)
    check("no clicks at transitions", worst < 1.6, f"(worst second {worst:.2f}x median)")
    tail = np.abs(np.r_[L[-SR // 2:], R[-SR // 2:]]).max()
    check("ends in silence", tail < 1e-3, f"(last 0.5s peak {tail:.1e})")

    # --- voice cleanup after morphs ---
    eng = Engine(device=-1)
    eng.morph_to(load_preset("theta-gamma"), 1.0); drive(eng, 2.0)
    n2 = len(eng.voices)
    eng.morph_to(load_preset("gamma-focus"), 1.0); drive(eng, 2.0)
    n1 = len(eng.voices)
    check("rack grows/shrinks cleanly", (n2, n1) == (2, 1), f"(2-voice -> {n2}, 1-voice -> {n1})")

    # --- log glide midpoint ---
    eng = Engine(device=-1)
    a = load_preset("theta-orbit")
    eng.morph_to(a, 0); drive(eng, 0.5)
    b = json.loads(json.dumps(a)); b["voices"][0]["f_mod"] = 40.0
    eng.morph_to(b, 10.0); drive(eng, 5.0)
    mid = eng.voices[0].target["f_mod"]
    check("log-frequency glide", abs(mid - np.sqrt(6 * 40)) < 1.0, f"(mid {mid:.2f} Hz, want {np.sqrt(240):.2f})")

    # --- preset round trip ---
    eng = Engine(device=-1)
    eng.morph_to(load_preset("theta-gamma"), 0); drive(eng, 0.5)
    p = save_preset(eng, "_roundtrip_test")
    back = load_preset("_roundtrip_test"); p.unlink()
    check("preset round-trip", back == normalize_state(capture(eng)))

    print("\nRESULT:", "ALL PASS ✅" if ok else "FAIL ❌")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
