# resonance

An open, inspectable brainwave-entrainment instrument inspired by the Monroe
Institute (Hemi-Sync, **SAM** / Spatial Angle Modulation) — built from first
principles so every signal is mathematically verifiable.

## Why
Monroe's SAM is locked down (no public recordings or specs). But the method is
*patented* (US20130010967A1), so the equations are public. This project
implements them, **proves** the output matches, and then goes further: real
3-D head geometry, multiple simultaneous voices, and scripted sessions.

## Quick start
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python live.py --session tour     # 4-minute guided listening tour (headphones!)
python live.py                    # blank rack, play it by hand
python -m resonance list          # what presets & sessions exist
```

## What's in it

**Live engine** (`resonance/engine.py`, `resonance/tui.py`) — a real-time,
phase-continuous synth you play from the terminal. Every knob glides; nothing
clicks. Up to 4 independent **voices**, each with its own carrier, entrainment
rate, spatial path and aim — e.g. a 6 Hz theta tone orbiting left while a 40 Hz
gamma tone swings on the right.

Each voice runs one of two engines:
- **spatial** — geometric. A spherical-head model (true interaural time delay +
  head-shadow level difference) moves the tone along a **path**: `pendulum`,
  `arc_front`, `orbit`, `halo`, `figure8`, `lissajous`, `rose`, `spinner`.
  Externalized, with a loudness cue that follows position. Strength is set by
  `motion depth` (degrees of interaural phase swing, pitch-independent; 150 ≈
  the classic engine) and `level cue` (dB). A real head only gives ~±70° at
  300 Hz — the defaults deliberately exceed physics, because physical was too
  subtle to hear.
- **classic** — the patent's phase-offset equations verbatim. Hyper-real,
  exaggerated swing (±150°). Modes `phase`, `natural`, `circular`, `figure8`.

Unlike binaural beats, SAM is spatial-position modulation, not a beat, so it
has **no ~30 Hz ceiling** — it reaches gamma (40–70 Hz).

Heads-up on perception: the brain's direction-finding is slow (binaural
sluggishness) — it tracks position changes up to roughly 5–10 Hz. Above that,
motion is heard as width/flutter, not travel. So at 40 Hz you won't *hear* a
swing; the entrainment claim is that the brainstem follows anyway. Press `z`
(slow-mo) to hear any path's actual shape at 0.5 Hz.

**Percussive strikes** (`resonance/pulse.py`) — any voice can be *struck*
(`pulse` 0 = smooth tone … 1 = fully struck; `strike decay`; `hits / cycle`;
`hit point` = where along the path the strike lands). Why: pure SAM has a
perfectly flat envelope in each ear — all its 40 Hz lives in the interaural
(binaural) channel. The strongest known 40 Hz brain response, the auditory
steady-state response, runs on the *envelope*, and sharp onsets drive it
hardest; the 40 Hz GENUS studies used click trains. Strikes add that channel,
phase-locked to the motion. `hits` decouples rhythm from motion: a 10 Hz orbit
(slow enough to hear as travel) struck 4× per lap is a 40 Hz strike train
walking the compass around your head.

| preset | 40 Hz envelope depth, one ear |
|---|---|
| `classic-gamma` (patent SAM) | 0.000 |
| `gamma-focus` (spatial, smooth) | 0.130 |
| `gamma-strike` (motion + strikes) | 0.333 |
| `gamma-walk` (10 Hz orbit × 4 hits) | 0.405 |
| `gamma-click` (strikes only, GENUS-style control) | 0.600 |

Science caveat: the mouse results (Martorell 2019, Murdock 2024) are contested
(Soula 2023 failed to replicate), and human trials are early.

**Presets** (`presets/*.json`) — the whole rack saved as a small, hand-editable
JSON file. `w` saves from the TUI, `o` glides to one over 3 s.

**Sessions** (`sessions/*.json`) — a timeline of scenes the engine glides
through: settle in alpha, slide to theta over three minutes, drop to delta,
surface, fade out. Frequencies glide in octaves (log scale) so descents feel
even. Between glides you can still turn knobs — perform over the script.

```json
{
  "name": "descent",
  "scenes": [
    {"at": "0:00",                   "preset": "alpha-halo",  "label": "settle"},
    {"at": "4:00",  "glide": "3:00", "preset": "theta-orbit", "label": "descend"},
    {"at": "12:00", "glide": "2:00", "preset": "delta-floor", "label": "floor",
     "globals": {"noise": 0.3}}
  ],
  "end": "24:00",
  "fade_out": 60
}
```
A scene takes a `preset`, inline `voices`, or both; `globals` overrides layer on
top. Shipped: `tour` (4 min), `focus` (20 min), `descent` (25 min).

**Offline rendering** — any session or preset to a 24-bit FLAC/WAV, rendered by
the same engine you hear live (a 25-minute session takes ~40 s):
```bash
python -m resonance render descent               # -> out/descent.flac
python -m resonance render --preset theta-gamma -d 300
```

**Classic generators** (`resonance/generators.py`) — `sam`, `binaural`,
`monaural`, `isochronic` (soft-gated), `pink_noise`, `brown_noise`, `tone`, all
returning plain numpy `(L, R)` arrays for your own layering:
```python
import resonance as rz
L, R = rz.sam(f_carrier=300, f_mod=40, dur=600, arc_deg=75, mode="phase")
rz.write("out/gamma.flac", *rz.normalize(L, R))
```

## TUI keys
| | |
|---|---|
| `tab`/`v`, `V` | next / previous voice |
| `a`, `x`, `X` | add, mute, delete voice |
| `↑↓` (`jk`), `←→` (`hl`) | pick knob, turn knob |
| `e` | engine spatial ↔ classic |
| `p`/`P`, `m`/`M` | cycle path (spatial) / mode (classic) |
| `1`–`5` | jump to delta / theta / alpha / beta / gamma |
| `-` `=`, `[` `]` | master volume, pink-noise bed |
| `w`, `o`, `s` | save preset, open preset, start/stop session |
| `z` | slow-mo: every voice traces its path at 0.5 Hz so you can hear the shape |
| `space`, `q` | play/pause, quit |

## Proof, not promises
Every claim about the signal is measured:
```bash
python test_engine.py       # realtime: clickless, IPD ±150° @ 40 Hz, knob glides
python test_session.py      # sessions: timing, no clicks at scene changes, clean fades
python test_pulse.py        # strikes: envelope drive, hit decoupling, clickless sweeps
python verify.py            # patent SAM: predicted vs measured IPD -> out/sam_gamma40.png
python verify_spatial.py    # head model: ITD 656 µs @ 90°, orbit continuity
python verify_multivoice.py # voices coexist and separate in space
```
Proof figures live in `out/*.png`.

**Honest science read:** frequency-following responses to rhythmic auditory
input are real, and driving the brain's localization pathway at 40 Hz is a
legitimate route to gamma. "Targeting a hemisphere" by aiming a voice left or
right has a real anatomical kernel but is **unproven** — it's on the list to
test with EEG. The engineering stands on its own either way.

## ⚠️ Safety
Entrainment audio can be intense. **Do not use while driving or operating
machinery.** If you have epilepsy or a seizure history, avoid it. Start with
short sessions at low volume.

## Status & next
Working: live multi-voice engine, both SAM engines, presets, sessions, offline
render, verification suite. Next: closed loop with a 1st-gen Emotiv EPOC
(record EEG during a session, check for frequency-following, then let the brain
steer the sound). See `LOGBOOK.md` for the full build history.
