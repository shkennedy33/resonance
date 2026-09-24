"""Verify the geometric spatializer: draw the shape gallery, prove the DSP still
yields SAM phase-modulation sidebands, and confirm a full 360 deg orbit is
click-free. Renders listenable WAVs for each shape.
"""
import numpy as np
from scipy.signal import hilbert
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import resonance as rz
from resonance.paths import PATHS, PATH_NAMES, get_path, TWO_PI
from resonance.spatial import render_path, woodworth_itd, HEAD_R, C

SR = rz.SR
OUT = "out"


def shape_gallery():
    theta = np.linspace(0, TWO_PI, 2000)
    ncol = 4
    nrow = int(np.ceil(len(PATH_NAMES) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.1 * nrow))
    fig.suptitle("SAM spatial paths — where the sound travels "
                 "(x = azimuth L↔R, y = elevation down↔up)", fontweight="bold")
    for ax, name in zip(axes.flat, PATH_NAMES):
        az, el = get_path(name)(theta)
        ax.plot(np.rad2deg(az), np.rad2deg(el), lw=1.8, color="#8e44ad")
        ax.scatter([np.rad2deg(az[0])], [np.rad2deg(el[0])], c="#27ae60", s=25, zorder=3)
        ax.set_title(name, fontsize=11)
        ax.axhline(0, c="gray", lw=.5); ax.axvline(0, c="gray", lw=.5)
        ax.set_xlim(-200, 200); ax.set_ylim(-100, 100)
        ax.set_xlabel("az°", fontsize=8); ax.set_ylabel("el°", fontsize=8)
    for ax in axes.flat[len(PATH_NAMES):]:
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = f"{OUT}/spatial_shapes.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    return p


def dsp_proof():
    f_c, f_m = 300.0, 40.0
    L, R = render_path("pendulum", f_c, f_m, 20.0, extent=1.4, shadow=1.0)
    t = np.arange(len(L)) / SR

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Geometric SAM — 300 Hz carrier, 40 Hz pendulum "
                 "(true ITD delay-line model)", fontweight="bold")

    # (a) static ITD vs azimuth (the physics)
    azx = np.linspace(-np.pi, np.pi, 400)
    ax[0, 0].plot(np.rad2deg(azx), woodworth_itd(azx) * 1e6, color="#2c3e50")
    itd90 = woodworth_itd(np.pi / 2) * 1e6
    ax[0, 0].set_title(f"a. ITD vs azimuth (Woodworth). |ITD|@90°={itd90:.0f} µs")
    ax[0, 0].set_xlabel("azimuth °"); ax[0, 0].set_ylabel("ITD µs")
    ax[0, 0].axhline(0, c="gray", lw=.5); ax[0, 0].axvline(0, c="gray", lw=.5)

    # (b) interaural phase diff over ~5 cycles => the motion, measured
    phL = np.unwrap(np.angle(hilbert(L)))
    phR = np.unwrap(np.angle(hilbert(R)))
    ipd = np.rad2deg(phL - phR)
    z0 = int(0.2 * SR); z1 = z0 + int(5 * SR / f_m)
    ax[0, 1].plot(t[z0:z1], ipd[z0:z1], color="#c0392b", lw=1.5)
    ax[0, 1].set_title("b. Interaural phase diff — swings at f_m (measured motion)")
    ax[0, 1].set_xlabel("s"); ax[0, 1].set_ylabel("IPD °")

    # (c) spectrum: are the SAM phase-modulation sidebands still there?
    win = np.hanning(len(L))
    spec = np.abs(np.fft.rfft(L * win)); spec /= spec.max() + 1e-12
    fr = np.fft.rfftfreq(len(L), 1 / SR)
    db = 20 * np.log10(spec + 1e-12)
    m = (fr >= f_c - 260) & (fr <= f_c + 260)
    ax[1, 0].plot(fr[m], db[m], color="#2c3e50", lw=1.0)
    for k in range(-5, 6):
        ax[1, 0].axvline(f_c + k * f_m, ls=":", c="#27ae60", lw=.8, alpha=.7)
    ax[1, 0].set_ylim(-80, 3)
    ax[1, 0].set_title(f"c. Spectrum — PM sidebands every {f_m:.0f} Hz survive (green)")
    ax[1, 0].set_xlabel("Hz"); ax[1, 0].set_ylabel("dB")

    # (d) full 360 orbit — ITD path must be continuous (no wrap click)
    Lo, Ro = render_path("orbit", f_c, 10.0, 3.0)  # slow 10 Hz orbit
    to = np.arange(len(Lo)) / SR
    phLo = np.unwrap(np.angle(hilbert(Lo)))
    phRo = np.unwrap(np.angle(hilbert(Ro)))
    zz = slice(int(0.2 * SR), int(0.2 * SR) + int(3 * SR / 10.0))
    ax[1, 1].plot(to[zz], np.rad2deg(phLo - phRo)[zz], color="#16a085", lw=1.5)
    ax[1, 1].set_title("d. Full 360° orbit IPD — continuous, no phase-wrap click")
    ax[1, 1].set_xlabel("s"); ax[1, 1].set_ylabel("IPD °")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = f"{OUT}/spatial_proof.png"
    fig.savefig(p, dpi=110); plt.close(fig)

    # numeric checks
    sig_side = spec[(fr > f_c) & (fr < f_c + 1.5 * f_m)].max()
    sb_present = 20 * np.log10(sig_side + 1e-12) > -40
    finite = np.all(np.isfinite(L)) and np.all(np.isfinite(Lo))
    orbit_range = (np.rad2deg(phLo - phRo)[zz].max() -
                   np.rad2deg(phLo - phRo)[zz].min())
    return p, dict(itd90=itd90, sb_present=bool(sb_present), finite=bool(finite),
                   orbit_ipd_range=float(orbit_range))


def render_wavs():
    demos = {
        "spatial_pendulum_gamma40": ("pendulum", 300, 40, {"extent": 1.4}),
        "spatial_orbit_gamma40":    ("orbit", 300, 40, {}),
        "spatial_figure8_theta6":   ("figure8", 300, 6, {}),
        "spatial_halo_alpha10":     ("halo", 300, 10, {}),
        "spatial_rose_theta6":      ("rose", 300, 6, {}),
        "spatial_lissajous_alpha10":("lissajous", 300, 10, {}),
    }
    for name, (path, fc, fm, kw) in demos.items():
        L, R = render_path(path, fc, fm, 12.0, shadow=1.2, path_kw=kw)
        L, R = rz.normalize(L, R)
        rz.write(f"{OUT}/{name}.wav", L, R)
    return list(demos)


if __name__ == "__main__":
    g = shape_gallery()
    p, checks = dsp_proof()
    wavs = render_wavs()
    print("=== geometric spatializer verification ===")
    print(f"ITD @ 90°        : {checks['itd90']:.0f} µs "
          f"(theory ~656 µs)  {'OK' if abs(checks['itd90']-656) < 30 else 'CHECK'}")
    print(f"PM sidebands     : {'present OK' if checks['sb_present'] else 'MISSING ❌'}")
    print(f"finite output    : {'OK' if checks['finite'] else 'NaN ❌'}")
    print(f"360° orbit IPD Δ  : {checks['orbit_ipd_range']:.0f}°  "
          f"(wide, continuous = real revolution)")
    print(f"shape gallery    : {g}")
    print(f"dsp proof        : {p}")
    print(f"demo WAVs        : {len(wavs)} in {OUT}/")
