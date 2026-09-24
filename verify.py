"""Render a 40 Hz gamma SAM tone and prove it matches the patent equations.
Also renders short demo files for each modality so we can listen later.

Run:  .venv/bin/python verify.py
"""
import numpy as np
import resonance as rz
from resonance.analyze import analyze_sam, interaural_phase

SR = rz.SR
OUT = "out"

def main():
    # ---- 40 Hz gamma SAM, 300 Hz carrier, wide arc ----
    f_carrier, f_mod, arc = 300.0, 40.0, 75.0
    L, R = rz.sam(f_carrier, f_mod, 20.0, arc_deg=arc, mode="phase")

    # numerical checks
    ipd = np.rad2deg(interaural_phase(L, R))
    # drop the fade regions before measuring
    edge = int(0.2 * SR)
    ipd_core = ipd[edge:-edge]
    peak_measured = (ipd_core.max() - ipd_core.min()) / 2
    peak_predicted = 2 * arc
    # frequency of the IPD swing (should equal f_mod)
    core = ipd_core - ipd_core.mean()
    spec = np.abs(np.fft.rfft(core * np.hanning(len(core))))
    freqs = np.fft.rfftfreq(len(core), 1 / SR)
    ipd_freq = freqs[np.argmax(spec)]

    print("=== 40 Hz gamma SAM verification ===")
    print(f"carrier f_s      : {f_carrier} Hz")
    print(f"modulation f_m   : {f_mod} Hz   (target = gamma)")
    print(f"arc phi_p        : {arc} deg")
    print(f"IPD swing peak   : predicted ±{peak_predicted:.1f}°, "
          f"measured ±{peak_measured:.1f}°  "
          f"({'OK' if abs(peak_measured-peak_predicted) < 5 else 'MISMATCH'})")
    print(f"IPD swing rate   : predicted {f_mod} Hz, measured {ipd_freq:.2f} Hz  "
          f"({'OK' if abs(ipd_freq-f_mod) < 1 else 'MISMATCH'})")

    p = analyze_sam(L, R, f_carrier=f_carrier, f_mod=f_mod, arc_deg=arc,
                    title="SAM — 300 Hz carrier, 40 Hz gamma spatial swing, 75° arc",
                    path=f"{OUT}/sam_gamma40.png")
    print(f"analysis figure  : {p}")

    # ---- render listenable demos of every modality (10 s each) ----
    demos = {
        "sam_gamma40_phase":   rz.sam(300, 40, 10, arc_deg=75, mode="phase"),
        "sam_theta6_natural":  rz.sam(300, 6, 10, arc_deg=80, mode="natural"),
        "sam_alpha10_circular":rz.sam(300, 10, 10, arc_deg=70, mode="circular"),
        "binaural_alpha10":    rz.binaural(200, 10, 10),
        "isochronic_theta6":   rz.isochronic(300, 6, 10),
        "monaural_beta20":     rz.monaural(300, 20, 10),
    }
    for name, (l, r) in demos.items():
        l, r = rz.normalize(l, r)
        rz.write(f"{OUT}/{name}.wav", l, r)
    print(f"demo files       : {len(demos)} WAVs in {OUT}/")

if __name__ == "__main__":
    main()
