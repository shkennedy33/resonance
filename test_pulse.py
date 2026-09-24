"""Headless test of the percussive strike layer (resonance/pulse.py).

Checks:
  - envelope drive: pure SAM has ~no 40 Hz in a single ear's envelope; struck
    voices have a lot (this is the channel the auditory steady-state response,
    and GENUS-style click trains, run on)
  - hits decouple rhythm from motion: 10 Hz orbit x 4 hits -> envelope peak at 40 Hz
  - envelope is continuous across strike wrap even when decay > period
  - sweeping pulse/decay live doesn't click; output stays in range
"""
import numpy as np
from scipy.signal import hilbert

from resonance.engine import Engine, BLOCK, SR
from resonance.presets import load_preset
from resonance.pulse import strike_gain


def render(state, secs=3.0, mutate=None):
    e = Engine(device=-1)
    e.morph_to(state, 0); e.gcur["volume"] = e.gtarget["volume"]
    buf = np.zeros((BLOCK, 2), np.float32); out = []
    for b in range(int(secs * SR / BLOCK)):
        if mutate:
            mutate(b, e)
        e._callback(buf, BLOCK, None, None); out.append(buf.copy())
    return np.concatenate(out)[SR // 2:].astype(float)


def env_spectrum(x):
    env = np.abs(hilbert(x))
    sp = np.abs(np.fft.rfft(env - env.mean())) / len(env)
    fr = np.fft.rfftfreq(len(env), 1 / SR)
    return fr, sp / env.mean()          # modulation depth relative to mean level


def at(fr, sp, f):
    return sp[np.argmin(np.abs(fr - f))]


def main():
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"{name:<36}: {bool(cond)}  {info}")

    classic = load_preset("classic-gamma")
    fr, sp = env_spectrum(render(classic)[:, 0])
    m_classic = at(fr, sp, 40)
    fr, sp = env_spectrum(render(load_preset("gamma-strike"))[:, 0])
    m_strike = at(fr, sp, 40)
    check("classic SAM: flat per-ear envelope", m_classic < 0.02, f"(40 Hz env depth {m_classic:.3f})")
    check("struck: strong 40 Hz envelope", m_strike > 0.25, f"(40 Hz env depth {m_strike:.3f}, "
          f"{m_strike / max(m_classic, 1e-6):.0f}x classic)")

    fr, sp = env_spectrum(render(load_preset("gamma-walk"))[:, 0])
    band = (fr > 5) & (fr < 60)
    peak = fr[band][np.argmax(sp[band])]
    check("walk: 10 Hz orbit x4 -> 40 Hz hits", abs(peak - 40) < 0.5, f"(envelope peak {peak:.1f} Hz)")

    # continuity at strike wrap with decay longer than the period
    f = 40.0; mph = 2 * np.pi * f * np.arange(SR) / SR
    g = strike_gain(mph, f, 1.0, 40.0)
    steps = np.abs(np.diff(g)); per_attack = steps.max()
    wrap = np.where(np.diff(((mph * 1) / (2 * np.pi)) % 1.0) < 0)[0]
    wrap_jump = steps[wrap].max()
    check("envelope continuous at wrap", wrap_jump <= per_attack * 1.01 and g.min() > 0.2,
          f"(wrap step {wrap_jump:.3f}, attack step {per_attack:.3f}, floor {g.min():.2f})")

    # live sweep of pulse & decay: no clicks beyond normal strike attacks
    st = load_preset("gamma-strike"); st["voices"][0]["pulse"] = 0.0
    def sweep(b, e):
        e.set("pulse", min(1.0, b / 100)); e.set("decay", 2 + (b % 60) * 0.5)
    x = render(st, 5.0, sweep)
    ref = render(load_preset("gamma-strike"), 3.0)
    worst = np.abs(np.diff(x, axis=0)).max() / np.abs(np.diff(ref, axis=0)).max()
    check("live pulse/decay sweep clickless", worst < 1.3 and np.abs(x).max() <= 1.0,
          f"(worst step {worst:.2f}x steady struck, peak {np.abs(x).max():.2f})")

    print("\nRESULT:", "ALL PASS ✅" if ok else "FAIL ❌")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
