"""Prove multi-voice SAM: independent carriers/entrainment-freqs coexisting and
spatially separable, both offline (compose) and in the real-time rack (Engine).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import resonance as rz
from resonance.compose import render_voices, _render_voice
from resonance.engine import Engine, BLOCK, SR

OUT = "out"


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def offline_proof():
    # a spatial chord: theta orbiting LEFT (220 Hz) + gamma darting RIGHT (400 Hz)
    A = {"engine": "spatial", "path": "pendulum", "carrier": 220, "f_mod": 6,
         "bias": -45, "shadow": 1.6, "itd_gain": 2.0, "gain": 0.8}
    B = {"engine": "spatial", "path": "spinner", "carrier": 400, "f_mod": 40,
         "bias": +45, "shadow": 1.6, "itd_gain": 2.0, "gain": 0.8}
    L, R = render_voices([A, B], dur=20.0)

    # per-voice lateralization (render each alone, no normalization)
    aL, aR = _render_voice(A, 20.0, SR)
    bL, bR = _render_voice(B, 20.0, SR)
    balA = 20 * np.log10(rms(aL) / rms(aR))   # + => left-heavy
    balB = 20 * np.log10(rms(bL) / rms(bR))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    fig.suptitle("Multi-voice SAM — theta@220Hz (left) + gamma@400Hz (right)",
                 fontweight="bold")

    win = np.hanning(len(L))
    sp = np.abs(np.fft.rfft(L * win)); sp /= sp.max() + 1e-12
    fr = np.fft.rfftfreq(len(L), 1 / SR)
    db = 20 * np.log10(sp + 1e-12)
    m = fr <= 600
    ax[0].plot(fr[m], db[m], color="#2c3e50", lw=1.0)
    for kk in range(-3, 4):
        ax[0].axvline(220 + kk * 6, ls=":", c="#8e44ad", lw=.7, alpha=.6)
        ax[0].axvline(400 + kk * 40, ls=":", c="#27ae60", lw=.7, alpha=.6)
    ax[0].set_ylim(-80, 3)
    ax[0].set_title("Spectrum: two carriers, each w/ own sideband spacing\n"
                    "purple=6Hz (theta) · green=40Hz (gamma)")
    ax[0].set_xlabel("Hz"); ax[0].set_ylabel("dB")

    ax[1].bar([0, 1], [rms(aL), rms(bL)], width=0.35, label="Left", color="#c0392b")
    ax[1].bar([0.4, 1.4], [rms(aR), rms(bR)], width=0.35, label="Right", color="#2980b9")
    ax[1].set_xticks([0.2, 1.2])
    ax[1].set_xticklabels([f"voice A (theta)\naim -45°  bal {balA:+.1f} dB",
                           f"voice B (gamma)\naim +45°  bal {balB:+.1f} dB"])
    ax[1].set_title("Per-voice L/R energy — voices land on opposite sides")
    ax[1].set_ylabel("rms"); ax[1].legend()

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = f"{OUT}/multivoice_proof.png"
    fig.savefig(p, dpi=110); plt.close(fig)

    # numeric checks
    cA = sp[np.argmin(np.abs(fr - 220))]
    cB = sp[np.argmin(np.abs(fr - 400))]
    both_carriers = cA > 0.1 and cB > 0.1
    separated = balA > 1.0 and balB < -1.0   # A left-heavy, B right-heavy

    # render listenable chords
    render_voices([A, B], dur=15.0)  # (already have L,R above; write them)
    rz.write(f"{OUT}/chord_theta_gamma.wav", *render_voices([A, B], dur=15.0))
    C = {"engine": "spatial", "path": "orbit", "carrier": 300, "f_mod": 10,
         "bias": 0, "gain": 0.6}
    rz.write(f"{OUT}/chord_triad.wav", *render_voices([A, B, C], dur=15.0))

    return p, dict(both_carriers=bool(both_carriers), separated=bool(separated),
                   balA=balA, balB=balB)


def realtime_rack_check():
    eng = Engine()
    eng.add_voice(); eng.add_voice()          # 3 voices total
    setups = [(220, 6), (400, 40), (300, 10)]
    for v, (c, f) in zip(eng.voices, setups):
        v.set("carrier", c); v.set("f_mod", f)
        v.cur = dict(v.target)                 # snap (skip slew for clean measure)
    eng.gcur["volume"] = 0.45

    buf = np.zeros((BLOCK, 2), dtype=np.float32)
    L = []
    for _ in range(210):
        eng._callback(buf, BLOCK, None, None)
        L.append(buf[:, 0].copy())
    L = np.concatenate(L)[10 * BLOCK:]         # drop priming

    finite = bool(np.all(np.isfinite(L)) and np.max(np.abs(L)) <= 1.0001)
    d = np.abs(np.diff(L)); ei = np.arange(BLOCK, len(L), BLOCK) - 1
    clickless = bool(d[ei].max() <= 3 * np.percentile(d, 99))
    win = np.hanning(len(L)); sp = np.abs(np.fft.rfft(L * win)); sp /= sp.max() + 1e-12
    fr = np.fft.rfftfreq(len(L), 1 / SR)
    peaks_present = all(sp[np.argmin(np.abs(fr - c))] > 0.08 for c, _ in setups)
    return dict(voices=len(eng.voices), finite=finite, clickless=clickless,
                all_carriers=bool(peaks_present))


if __name__ == "__main__":
    p, off = offline_proof()
    rt = realtime_rack_check()
    print("=== multi-voice verification ===")
    print(f"[offline] both carriers present : {off['both_carriers']}")
    print(f"[offline] voices spatially split : {off['separated']}  "
          f"(A {off['balA']:+.1f} dB left, B {off['balB']:+.1f} dB {'left' if off['balB']>0 else 'right'})")
    print(f"[realtime] rack voices           : {rt['voices']}")
    print(f"[realtime] finite & clickless    : {rt['finite'] and rt['clickless']}")
    print(f"[realtime] all 3 carriers present: {rt['all_carriers']}")
    print(f"proof figure : {p}")
    print("demo WAVs    : out/chord_theta_gamma.wav, out/chord_triad.wav")
    ok = (off["both_carriers"] and off["separated"] and rt["finite"]
          and rt["clickless"] and rt["all_carriers"])
    print("\nRESULT:", "ALL PASS ✅" if ok else "FAIL ❌")
