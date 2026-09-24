# resonance

An open, inspectable suite of brainwave-entrainment audio tools inspired by the
Monroe Institute (Hemi-Sync, **SAM** / Spatial Angle Modulation) — built from
first principles so every signal is mathematically verifiable.

## Why
Monroe's SAM is locked down (no public recordings/specs). But the method is
*patented* (US20130010967A1), so the equations are public. This project
implements them cleanly and **proves** the output matches, then extends into a
full layered-session toolkit.

## Modalities
- **SAM** — Spatial Angle Modulation. A single carrier tone whose apparent
  spatial position swings at your target frequency via interaural phase
  modulation. Unlike binaural beats it has **no ~30 Hz ceiling**, so it reaches
  gamma (40–70 Hz). Modes: `phase` (patent-faithful), `natural` (adds loudness
  cue), `circular` (orbits the head).
- **Binaural beats**, **monaural beats**, **isochronic tones** (soft-gated),
  **pink/brown noise** beds.

## Quick start
```bash
python -m venv .venv && . .venv/bin/activate
pip install numpy scipy soundfile sounddevice matplotlib
python verify.py         # renders + proves a 40 Hz gamma SAM tone; writes out/
```

```python
import resonance as rz
L, R = rz.sam(f_carrier=300, f_mod=40, dur=600, arc_deg=75, mode="phase")  # 10-min gamma
L, R = rz.normalize(L, R)
rz.write("out/gamma_session.flac", L, R)
```

## Verifying a SAM signal
```python
from resonance.analyze import analyze_sam
analyze_sam(L, R, f_carrier=300, f_mod=40, arc_deg=75, path="out/proof.png")
```
Produces a 4-panel figure: waveform phase-shift, interaural-phase swing (the
spatial motion, measured), PM sideband spectrum, and the mono-sum tremolo.

## ⚠️ Safety
Entrainment audio can be intense. **Do not use while driving or operating
machinery.** If you have epilepsy or a seizure history, avoid — flashing/pulsing
stimuli carry risk. Start with short sessions and low volume.

## Status
Early. SAM engine + generators + verification working. Session sequencing,
real-time engine, and CLI are next. See `LOGBOOK.md`.
