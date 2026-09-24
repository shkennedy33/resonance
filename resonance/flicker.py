"""40 Hz light flicker, phase-locked to the audio strikes (GENUS-style audio+visual).

Runs as its own process (own main thread, own GIL) so frame timing never fights
the audio callback. Launched by the engine (TUI key `L`), or by hand:
    python -m resonance.flicker <shared-memory-name>

Timing
------
Each frame, predict when it will actually light up the panel (now + display
latency), extrapolate the lead voice's phase to that instant from the audio
timing packet (avsync.py), and set the frame's brightness to the flicker
waveform's AVERAGE over the frame's duration (box-filtered, in linear light).

Why band-limit: at 100 Hz refresh, 40 Hz is 2.5 frames per cycle. Naive on/off
sampling stutters (1-1-0-1-0...) with a strong 20 Hz sub-flicker; even frame
averaging leaves the square wave's 120 Hz harmonic aliasing to 20 Hz. 40 Hz is
below the display's 50 Hz Nyquist, so we keep only the waveform's components
under fps/2 and render those exactly: a clean 40 Hz. Honest consequence: on a
100 Hz display a 40 Hz flicker is necessarily sine-shaped; `square` vs `strike`
only differ on 240 Hz+ displays. Theta nesting (all < 50 Hz) survives intact.

Modes: square = GENUS-style 50% duty at the strike rate; strike = follows the
audio strike envelope (short flashes). Both honour `nest` (theta bursts).

Keys (in the flicker window)
    f fullscreen   m mode   up/down brightness   left/right latency -/+1 ms
    q / esc close
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from .pulse import _shape, _nest_window, TWO_PI

MODES = ["square", "strike"]
GAMMA = 2.2
_SUB = (np.arange(24) + 0.5) / 24          # sub-samples within one frame


def waveform(phi, st, mode):
    """Linear-light brightness 0..1 at lead-voice phase(s) phi (radians)."""
    hits = max(1, int(round(st["hits"])))
    f = max(st["f_mod"] * hits, 1e-6)
    ph = phi - np.deg2rad(st["hit_at"])
    u = (ph * hits / TWO_PI) % 1.0
    if mode == "square":
        w = (u < 0.5).astype(float)
    else:
        w = _shape(u, 1.0 / f, max(st["decay"], 0.2) / 1000.0)
    return w * _nest_window(ph, st["nest"])


_CACHE = {}


def _spectrum(st, mode, frame_dt):
    """Band-limited, frame-held Fourier series of one path cycle of the waveform.
    Harmonics at or above the display's Nyquist (fps/2) are removed — they
    can't be shown and would alias into sub-flicker (at 100 Hz refresh a square
    wave's 120 Hz harmonic lands at 20 Hz). Remaining harmonics are scaled by
    the frame-hold (box) response. Normalized to span 0..1 over the cycle."""
    key = (mode, round(st["f_mod"], 4), round(st["hits"]), round(st["hit_at"], 2),
           round(st["nest"], 3), round(st["decay"], 2), round(frame_dt, 4))
    c = _CACHE.get(key)
    if c is not None:
        return c
    N = 1024
    phi = TWO_PI * np.arange(N) / N
    X = np.fft.rfft(waveform(phi, st, mode)) / N
    k = np.arange(len(X))
    fk = k * st["f_mod"]
    X[fk >= 0.5 / frame_dt] = 0.0
    X[1:] *= 2.0 * np.sinc(fk[1:] * frame_dt)      # one-sided x frame-hold
    ks = np.nonzero(X)[0]
    rec = np.real(np.exp(1j * np.outer(phi, ks)) @ X[ks])
    lo, hi = rec.min(), rec.max()
    c = (ks, X[ks], lo, max(hi - lo, 1e-9))
    if len(_CACHE) > 64:
        _CACHE.clear()
    _CACHE[key] = c
    return c


def frame_brightness(st, t_show, frame_dt, mode="square"):
    """Linear brightness (0..1) to display for the frame lit over
    [t_show, t_show + frame_dt): the band-limited waveform, frame-averaged."""
    ks, X, lo, span = _spectrum(st, mode, frame_dt)
    t_mid = t_show + 0.5 * frame_dt
    phi = st["mphase"] + TWO_PI * st["f_mod"] * (t_mid - st["dac_mono"])
    v = float(np.real(np.sum(X * np.exp(1j * ks * phi))))
    return min(max((v - lo) / span, 0.0), 1.0)


def to_pixel(b):
    return int(round(255 * max(0.0, min(1.0, b)) ** (1.0 / GAMMA)))


# ---- window ---------------------------------------------------------------- #
def run(shm_name, mode="square", level=1.0, latency_ms=None):
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    from .avsync import AVReader

    reader = AVReader(shm_name)
    parent = os.getppid()
    pygame.init()
    pygame.display.set_caption("resonance — light")
    screen = pygame.display.set_mode((64, 36), pygame.SCALED | pygame.RESIZABLE, vsync=1)
    try:
        refresh = float(pygame.display.get_current_refresh_rate()) or 60.0
    except Exception:
        refresh = 60.0
    frame_dt = 1.0 / refresh
    offset = 0.0 if latency_ms is None else latency_ms / 1000.0
    font = pygame.font.Font(None, 9)
    hud_until = time.monotonic() + 4.0
    started = time.monotonic()
    last_flip = None

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif k == pygame.K_f:
                    pygame.display.toggle_fullscreen()
                elif k == pygame.K_m:
                    mode = MODES[(MODES.index(mode) + 1) % len(MODES)]
                elif k == pygame.K_UP:
                    level = min(1.0, level + 0.05)
                elif k == pygame.K_DOWN:
                    level = max(0.0, level - 0.05)
                elif k == pygame.K_RIGHT:
                    offset += 0.001
                elif k == pygame.K_LEFT:
                    offset -= 0.001
                hud_until = time.monotonic() + 3.0

        if os.getppid() != parent:
            break                             # engine process went away: close
        now = time.monotonic()
        st = reader.read()
        if st is None or st["running"] < 0.5 or now - st["writer_alive"] > 0.5:
            b = 0.0                           # paused / no audio yet: dark
        else:
            # frame shows ~one refresh after flip (compositor), plus user trim
            t_show = now + frame_dt + offset
            b = frame_brightness(st, t_show, frame_dt, mode)
        ramp = min(1.0, (now - started) / 2.0)   # 2 s fade-in on open
        v = to_pixel(b * level * ramp)
        screen.fill((v, v, v))
        rate = st["f_mod"] * max(1, round(st["hits"])) if st else 0.0
        too_slow = st is not None and rate >= 0.5 / frame_dt
        if now < hud_until or too_slow:
            txt = (f"{mode}  lvl {level:.2f}  trim {offset*1000:+.0f}ms  "
                   f"{1/frame_dt:.0f}fps  rate {rate:.1f}Hz" if st else "waiting for audio")
            screen.blit(font.render(txt, False, (200, 40, 40)), (1, 1))
            if too_slow:     # band-limiting removes it: steady light, not garbage
                screen.blit(font.render(f"display too slow for {rate:.0f} Hz", False,
                                        (200, 40, 40)), (1, 10))
        pygame.display.flip()
        t = time.monotonic()
        if last_flip is not None:
            frame_dt += ((t - last_flip) - frame_dt) * 0.02   # track real refresh
            frame_dt = min(max(frame_dt, 0.25 / refresh), 4.0 / refresh)
        last_flip = t

    pygame.quit()
    reader.close()


if __name__ == "__main__":
    run(sys.argv[1], latency_ms=float(os.environ.get("RZ_LIGHT_TRIM_MS", 0)))
