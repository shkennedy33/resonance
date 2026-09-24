"""Headless test of the percussive strike layer (resonance/pulse.py).

Checks:
  - envelope drive: pure SAM has ~no 40 Hz in a single ear's envelope; struck
    voices have a lot (this is the channel the auditory steady-state response,
    and GENUS-style click trains, run on)
  - hits decouple rhythm from motion: 10 Hz orbit x 4 hits -> envelope peak at 40 Hz
  - envelope is continuous across strike wrap even when decay > period
  - sweeping pulse/decay live doesn't click; output stays in range
  - nesting: theta-gamma-nest envelope carries BOTH 40 Hz strikes and 6.67 Hz bursts
  - click timbre is broadband AND localized: a click struck hard-right reaches
    the right ear first by the head-model ITD
"""
import json

import numpy as np
from scipy.signal import hilbert

from resonance.engine import Engine, BLOCK, SR
from resonance.presets import load_preset
from resonance.pulse import strike_gain
from resonance.spatial import itd_gain_for_depth, ITD_90


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

    # --- nesting: phase-amplitude coupling shows up as a theta line in the envelope
    nest = load_preset("theta-gamma-nest")
    flat = json.loads(json.dumps(nest)); flat["voices"][0]["nest"] = 0.0
    for st in (nest, flat):     # level cue off: the orbit's ILD alone rises & falls at 6.67 Hz
        st["globals"]["noise"] = 0.0; st["voices"][0]["ild"] = 0.0
    fr, sp = env_spectrum(render(nest, 4.0)[:, 0]); th_n, g_n = at(fr, sp, 6.667), at(fr, sp, 40)
    fr, sp = env_spectrum(render(flat, 4.0)[:, 0]); th_f = at(fr, sp, 6.667)
    check("nest: theta bursts + gamma strikes", th_n > 3 * th_f and g_n > 0.1,
          f"(6.67 Hz env {th_f:.3f} -> {th_n:.3f}; 40 Hz {g_n:.3f})")

    # --- click: broadband
    def hf_share(x):
        P = np.abs(np.fft.rfft(x)) ** 2; f = np.fft.rfftfreq(len(x), 1 / SR)
        return P[f > 4000].sum() / P.sum()
    tone = load_preset("gamma-walk"); clk = load_preset("click-walk")
    h_t, h_c = hf_share(render(tone)[:, 0]), hf_share(render(clk)[:, 0])
    check("click: broadband energy", h_c > 0.2 and h_t < 0.02, f"(>4 kHz share tone {h_t:.3f}, click {h_c:.2f})")

    # --- click: localized. single hit per cycle at the hard-right point of a pendulum
    st = json.loads(json.dumps(clk)); v = st["voices"][0]
    v.update(path="pendulum", f_mod=5.0, hits=1, hit_at=90.0, arc=90.0, ild=0.0)
    x = render(st, 3.0)
    lagmax = int(0.004 * SR)
    xc = [np.dot(x[lagmax:-lagmax, 0], np.roll(x[:, 1], k)[lagmax:-lagmax]) for k in range(-lagmax, lagmax + 1)]
    lag_ms = (np.argmax(xc) - lagmax) / SR * 1000     # >0: right ear leads
    want = itd_gain_for_depth(v["depth"], v["carrier"]) * ITD_90 * 1000
    check("click: lands in space (right ear first)", abs(lag_ms - want) < 0.15,
          f"(right leads by {lag_ms:.2f} ms, head model {want:.2f} ms)")

    print("\nRESULT:", "ALL PASS ✅" if ok else "FAIL ❌")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
