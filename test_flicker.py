"""Headless test of the light flicker (no window needed).

Checks, simulating a 100 Hz display against a live-style engine run:
  - area-sampled 40 Hz at 100 fps: dominant frame-brightness component is 40 Hz
    (not the 20 Hz sub-flicker naive on/off sampling produces)
  - phase lock: the light's 40 Hz is in phase with the audio strike train
    (both measured against the same monotonic clock via the avsync packet)
  - nesting carries through to the light (theta line in the brightness)
  - avsync seqlock round-trip across the shared-memory boundary
"""
import numpy as np

import resonance.avsync as avsync
from resonance.avsync import AVWriter, AVReader
from resonance.engine import Engine, BLOCK, SR
from resonance.flicker import frame_brightness, waveform
from resonance.presets import load_preset


class Clock:   # simulated monotonic clock: advances one block per callback
    t = 100.0
    def monotonic(self): return self.t
CLK = Clock()
avsync.time = CLK


class TI:   # fake PortAudio time_info: block plays 30 ms after the callback
    def __init__(self, _=None): self.currentTime, self.outputBufferDacTime = 1.0, 1.03


def cb(e, buf):
    e._callback(buf, BLOCK, TI(), None)
    CLK.t += BLOCK / SR


def main():
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"{name:<38}: {bool(cond)}  {info}")

    e = Engine(device=-1)
    e.av = AVWriter()
    e.morph_to(load_preset("gamma-strike"), 0)
    buf = np.zeros((BLOCK, 2), np.float32)
    rd = AVReader(e.av.name)
    for _ in range(30): cb(e, buf)            # let the 0.15 s morph + slew land
    st = rd.read()
    check("avsync round-trip", st and abs(st["f_mod"] - 40) < 1e-6 and abs(st["pulse"] - 1.0) < 0.01,
          f"(f_mod {st['f_mod']:.1f}, pulse {st['pulse']:.1f})")

    # simulate 4 s of a 100 Hz display
    fps, dt = 100.0, 0.01
    t = st["dac_mono"] + np.arange(400) * dt
    for mode in ("square", "strike"):
        b = np.array([frame_brightness(st, ti, dt, mode) for ti in t])
        sp = np.abs(np.fft.rfft(b - b.mean())); fr = np.fft.rfftfreq(len(b), dt)
        peak = fr[np.argmax(sp)]
        sub = sp[np.argmin(np.abs(fr - 20))] / sp.max()
        check(f"{mode}: clean 40 Hz at 100 fps", abs(peak - 40) < 0.3 and sub < 0.05,
              f"(peak {peak:.1f} Hz, 20 Hz sub-flicker {sub:.3f})")
    naive = np.array([waveform(np.array([st["mphase"] + 2*np.pi*40*(ti - st["dac_mono"])]), st, "square")[0] for ti in t])
    sp = np.abs(np.fft.rfft(naive - naive.mean())); fr = np.fft.rfftfreq(len(naive), dt)
    print(f"{'  (naive on/off sampling, for contrast)':<38}   20 Hz sub-flicker {sp[np.argmin(np.abs(fr-20))]/sp.max():.3f}")

    # phase lock: audio strike onsets vs light onsets on the shared clock.
    # motion & level cue off: the swing's ILD also moves each ear's 40 Hz envelope
    lock = load_preset("gamma-strike"); lock["voices"][0].update(depth=0.0, ild=0.0)
    e.morph_to(lock, 0)
    for _ in range(30): cb(e, buf)
    # render audio blocks; map sample k of block j to time dac_j + k/SR
    L = []; T = []
    for j in range(100):
        t_dac = CLK.t + 0.03                  # TRUE speaker time of this block (ground truth)
        cb(e, buf)
        s2 = rd.read()
        L.append(buf[:, 0].copy()); T.append(t_dac + np.arange(BLOCK) / SR)
    L = np.concatenate(L); T = np.concatenate(T)
    env = np.abs(L)
    ph_audio = np.angle(np.sum(env * np.exp(-2j * np.pi * 40 * T)))
    tl = np.linspace(T[0], T[-1], 20000)
    light = np.array([waveform(np.array([s2["mphase"] + 2*np.pi*40*(ti - s2["dac_mono"])]), s2, "strike")[0] for ti in tl])
    ph_light = np.angle(np.sum(light * np.exp(-2j * np.pi * 40 * tl)))
    err_ms = ((ph_light - ph_audio + np.pi) % (2*np.pi) - np.pi) / (2*np.pi*40) * 1000
    check("light phase-locked to audio strikes", abs(err_ms) < 1.5, f"(offset {err_ms:+.2f} ms)")

    # nesting reaches the light
    e.morph_to(load_preset("theta-gamma-nest"), 0)
    for j in range(30): cb(e, buf)
    st = rd.read(); t = st["dac_mono"] + np.arange(600) * dt
    b = np.array([frame_brightness(st, ti, dt, "square") for ti in t])
    sp = np.abs(np.fft.rfft(b - b.mean())); fr = np.fft.rfftfreq(len(b), dt)
    th = sp[np.argmin(np.abs(fr - 6.667))] / sp.max()
    check("nest: theta bursts in the light", th > 0.3, f"(6.67 Hz line {th:.2f} of peak)")

    rd.close(); e.av.close()
    print("\nRESULT:", "ALL PASS ✅" if ok else "FAIL ❌")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
