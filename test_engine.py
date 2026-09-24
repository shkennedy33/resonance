"""Headless test of the real-time Engine: drive its audio callback by hand and
prove the synthesis is sane, phase-continuous, and clickless across blocks —
without opening a live audio stream (works in any environment).

Renders a block-by-block run where we change knobs mid-stream, then checks:
  - no NaN/Inf, stays within [-1, 1]
  - phase continuity across block boundaries (no discontinuity/click)
  - the interaural phase difference tracks f_mod at the right depth
  - a knob change (arc) glides rather than jumps
"""
import numpy as np
from scipy.signal import hilbert

from resonance.engine import Engine, SR, BLOCK


def run_blocks(eng, n_blocks, mutate=None):
    buf = np.zeros((BLOCK, 2), dtype=np.float32)
    L, R = [], []
    for b in range(n_blocks):
        if mutate:
            mutate(b, eng)
        eng._callback(buf, BLOCK, None, None)
        L.append(buf[:, 0].copy())
        R.append(buf[:, 1].copy())
    return np.concatenate(L), np.concatenate(R)


def main():
    eng = Engine()
    eng.engine_mode = "classic"
    eng.mode = "phase"
    # settle params to targets instantly for a clean measurement segment
    eng.cur = dict(eng.target)
    eng.set("carrier", 300); eng.set("f_mod", 40); eng.set("arc", 75)
    eng.cur = dict(eng.target)

    # steady segment for measurement
    L, R = run_blocks(eng, 200)  # ~4.3 s
    ok = True

    # 1. sanity
    finite = np.all(np.isfinite(L)) and np.all(np.isfinite(R))
    inrange = np.max(np.abs(L)) <= 1.0001 and np.max(np.abs(R)) <= 1.0001
    print(f"finite & in-range        : {finite and inrange}  "
          f"(peak {max(np.abs(L).max(), np.abs(R).max()):.3f})")
    ok &= finite and inrange

    # 2. block-boundary continuity (click detector): max |sample-to-sample|
    #    should not spike at block edges vs. within blocks.
    d = np.abs(np.diff(L))
    edge_idx = np.arange(BLOCK, len(L), BLOCK) - 1
    edge_jumps = d[edge_idx]
    typical = np.percentile(d, 99)
    worst_edge = edge_jumps.max()
    clickless = worst_edge <= 3 * typical
    print(f"block-edge continuity    : {clickless}  "
          f"(worst edge jump {worst_edge:.4f} vs 99pct {typical:.4f})")
    ok &= clickless

    # 3. interaural phase difference tracks f_mod at ±2*arc
    phL = np.unwrap(np.angle(hilbert(L)))
    phR = np.unwrap(np.angle(hilbert(R)))
    ipd = np.rad2deg(phL - phR)
    core = ipd[2000:-2000]
    depth = (core.max() - core.min()) / 2
    sp = np.abs(np.fft.rfft((core - core.mean()) * np.hanning(len(core))))
    fr = np.fft.rfftfreq(len(core), 1 / SR)
    ipd_f = fr[np.argmax(sp)]
    dep_ok = abs(depth - 150) < 6
    frq_ok = abs(ipd_f - 40) < 0.5
    print(f"IPD depth (want ±150°)   : {dep_ok}  (measured ±{depth:.0f}°)")
    print(f"IPD rate  (want 40 Hz)   : {frq_ok}  (measured {ipd_f:.2f} Hz)")
    ok &= dep_ok and frq_ok

    # 4. glide test: snap arc target from 75 -> 20 and confirm it eases (many
    #    small steps), not a single jump.
    eng.set("arc", 20)
    arcs = []
    for _ in range(60):  # ~1.3 s of blocks
        buf = np.zeros((BLOCK, 2), dtype=np.float32)
        eng._callback(buf, BLOCK, None, None)
        arcs.append(eng.cur["arc"])
    steps = np.abs(np.diff(arcs))
    glided = steps.max() < 20 and abs(arcs[-1] - 20) < 1.0
    print(f"knob glide (75->20)      : {glided}  "
          f"(largest single step {steps.max():.2f}°, settled at {arcs[-1]:.1f}°)")
    ok &= glided

    # 5. spatial engine: finite, clickless across blocks, real SAM sidebands
    eng.engine_mode = "spatial"
    eng.path_name = "pendulum"
    for kk, vv in (("carrier", 300), ("f_mod", 40), ("arc", 75), ("depth", 150)):
        eng.set(kk, vv); eng.cur[kk] = float(vv)
    Ls_full, _ = run_blocks(eng, 210)
    Ls = Ls_full[10 * BLOCK:]          # drop delay-line priming (startup transient)
    s_finite = bool(np.all(np.isfinite(Ls)) and np.max(np.abs(Ls)) <= 1.0001)
    ds = np.abs(np.diff(Ls))
    ei = np.arange(BLOCK, len(Ls), BLOCK) - 1
    s_click = bool(ds[ei].max() <= 3 * np.percentile(ds, 99))
    win = np.hanning(len(Ls))
    sp = np.abs(np.fft.rfft(Ls * win)); sp /= sp.max() + 1e-12
    fr = np.fft.rfftfreq(len(Ls), 1 / SR)
    sb = bool(20 * np.log10(sp[(fr > 300) & (fr < 300 + 60)].max() + 1e-12) > -40)
    print(f"spatial engine           : {s_finite and s_click and sb}  "
          f"(finite&in-range {s_finite}, clickless {s_click}, SAM sidebands {sb})")
    ok &= s_finite and s_click and sb

    print("\nRESULT:", "ALL PASS ✅" if ok else "FAIL ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
