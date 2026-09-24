# LOGBOOK — resonance

An open, inspectable brainwave-entrainment audio suite inspired by the Monroe
Institute (Hemi-Sync, SAM). Built from first principles so every signal is
verifiable, unlike the locked-down proprietary originals.

---

## 2026-07-04 — Session 1: SAM reverse-engineered & working

**Context / origin.** Sequoyah wanted to replicate Monroe's SAM (Spatial Angle
Modulation) — impossible to find recordings/specs anywhere — plus a broader
suite (binaural, isochronic, layering). Turns out SAM is *patented*
(US20130010967A1 + continuation US20150016613A1), so the full spec is public.

**The SAM secret (from the patent), now implemented verbatim:**
```
S_L(t) = A*sin[2*pi*f_s*t + phi_p*sin(2*pi*f_m*t) + phi_L]
S_R(t) = A*sin[2*pi*f_s*t - phi_p*sin(2*pi*f_m*t) + phi_R]
```
- f_s = carrier (patent likes 300 or 440 Hz), f_m = entrainment target,
  phi_p = peak phase deviation = arc size, phi_L/phi_R = arc center / hemisphere aim.
- Opposite signs on the phi_p term => interaural phase difference swings
  ±2*phi_p at f_m. It's spatial-position modulation, NOT a beat, so it has
  **no 30 Hz ceiling** — that's how it reaches gamma (40-70 Hz) where binaural
  beats physically can't.

**Science read (honest):**
- Solid: auditory steady-state / frequency-following response is real; driving
  the localization pathway via interaural-phase modulation is a legit route to gamma.
- Unproven: "target specific brain regions" via arc placement (phi_L/phi_R).
  Real kernel (contralateral auditory pathway) but big extrapolation. TEST it.
- Marketing velvet: the quantum-microtubule/Orch-OR "consciousness at 40 Hz"
  story. Engineering stands on its own regardless.

**Built this session:**
- `resonance/core.py` — timeline, cosine fade, mix, normalize, 24-bit WAV/FLAC out (48 kHz).
- `resonance/generators.py` — `sam()` (modes: phase / natural / circular),
  `binaural()`, `monaural()`, `isochronic()` (soft raised-cosine gate),
  `pink_noise()`, `brown_noise()`, `tone()`.
- `resonance/analyze.py` — `analyze_sam()` 4-panel proof figure + `spectrum()`.
- `verify.py` — renders 40 Hz gamma SAM, checks IPD math, dumps 6 demo WAVs.

**Verification result (PASSED):** 300 Hz carrier / 40 Hz / 75° arc →
IPD swing predicted ±150°, measured ±150°; swing rate measured 40.00 Hz.
Spectrum shows clean Bessel PM sidebands every 40 Hz around 300 Hz carrier.
Figure: `out/sam_gamma40.png`. Demo WAVs in `out/`.

**Env:** py venv at `.venv` (numpy 2.5.1, scipy 1.18, soundfile, sounddevice,
matplotlib). Box: Arch, PipeWire (48 kHz), ffmpeg present. Run:
`.venv/bin/python verify.py`.

## 2026-07-04 — Session 2: real-time live engine + TUI

**Direction decided (Sequoyah):** (1) primary use = REAL-TIME live knobs;
(2) deepen SAM next; (3) has a **1st-gen Emotiv EPOC** + existing software.

**Found the EEG software:** `/mnt/fast/emokit/` — open-source emokit lib for the
1st-gen EPOC. Key bits: `python/emokit/emotiv.py` (decoder), `doc/
emotiv_protocol.asciidoc`, `linux/epoc.rules` (udev), old recorded CSVs. This is
the path to closed-loop later (14ch, 128 Hz). Phase 2.

**Audio routing:** box exposes `pulse` (dev 15) / `pipewire` (dev 14) via ALSA.
Engine routes through `pulse` so it respects the desktop default sink/headphones.

**Built this session:**
- `resonance/engine.py` — `Engine`: phase-continuous, clickless real-time SAM
  synth in the PortAudio callback. Running carrier+mod phase accumulators; params
  slew toward targets (one-pole ~60 ms) with intra-block ramps on amp params.
  Live params: carrier, f_mod, arc(phi_p), hemi bias, level_depth, pink noise,
  master vol. Arc MODES = phase/natural/circular/figure8 (the "SAM depth" arc
  paths). Band quick-jumps delta/theta/alpha/beta/gamma.
- `resonance/tui.py` + `live.py` — curses control surface (sliders, meter, mode,
  bands). Launch: `.venv/bin/python live.py`.
- `test_engine.py` — headless callback-driver test.

**Verification (ALL PASS):** finite & in-range (peak 0.27); block-edge
continuity clickless (worst edge jump 0.0124 ≈ 99pct in-block 0.0123); IPD ±150°
@ 39.92 Hz; knob glide 75°→20° eases (max step 11.5°, settles 20.0°). Live
stream opens on `pulse` at 48k/2ch/float32 and runs callbacks cleanly (tested
silent). Only unverified thing = subjective listening (needs S).

**HOW TO RUN THE LIVE ENGINE:**
`cd /mnt/fast/projects/resonance && .venv/bin/python live.py`
↑↓ select · ←→ adjust · m mode · 1-5 band · space play/pause · q quit.

## 2026-07-07 — Session 3: geometric spatializer (SAM depth jump)

**Goal:** "keep upping SAM sophistication." Built the two foundational upgrades.

**New modules:**
- `resonance/paths.py` — parametric spatial PATHS as first-class geometry. Each
  is theta -> (az, el). Registry: pendulum, arc_front, orbit(full 360), halo,
  figure8, lissajous, rose, spinner. `extent` = angular size. Plot (az,el) to
  SEE the shape (`out/spatial_shapes.png`).
- `resonance/spatial.py` — spherical-head binaural: true ITD via fractional
  delay lines (Woodworth, head r=8.75cm, max 656µs @90°) + freq-aware ILD
  head-shadow. `render_path()` offline + `Spatializer` class (block-wise,
  history-continuous) for realtime. Params: shadow (ILD exagg), itd_gain
  (1=physical .. 3=hyper-real), orient (aim), extent (size).

**Key honest finding:** geometric model swings IPD ±66° @300Hz carrier vs the
classic phase-offset engine's ±150°. NOT a bug — physics: at 300Hz the head
delay only buys ~±71° of carrier phase. So geometric = externalized/real but
gentler; classic = hyper-real/exaggerated. Unified via `itd_gain` knob. To
strengthen geometric motion: raise carrier (more phase per µs) or itd_gain/shadow.

**Engine integration:** `Engine` now has two engines, toggle with 'e':
  - "classic" = original phase-offset (arc modes phase/natural/circular/figure8)
  - "spatial" = geometric (path picker; arc knob->size, bias knob->aim)
  New live params: itd_gain, shadow. New keys in TUI: e=engine, p/P=path.
  Default engine = spatial. start() now soft-attacks (volume glides from 0) to
  kill the delay-line priming tick + gentler onsets.

**Verification (ALL PASS):** ITD@90°=656µs (theory match); PM sidebands survive
in geometric render; full 360° orbit IPD continuous (no wrap click); dual-engine
headless test green (classic ±150°@40Hz + spatial finite/clickless/sidebands);
live 'pulse' stream runs spatial + live engine-toggle/path-cycle cleanly.
Figures: `out/spatial_shapes.png`, `out/spatial_proof.png`. New demo WAVs:
`out/spatial_*.wav` (pendulum/orbit/figure8/halo/rose/lissajous).

## 2026-07-08 — Session 4: multi-voice rack

**Goal:** patent's "multiple paths at different frequencies" — N independent SAM
voices at once (e.g. theta-left + gamma-right).

**Refactor:** `engine.py` now has `Voice` (its own params/phases/Spatializer/
mute) + `Engine` holding a rack of up to `MAX_VOICES=4`. Globals (master volume,
pink noise) live at engine level. Callback sums all voices -> +noise -> master
-> clip. Per-voice params: carrier, f_mod, arc, bias, level_depth, itd_gain,
shadow, GAIN. Clickless per-voice gain+mute via smoothed `_mg`. Engine proxies
(cur/target/mode/engine_mode/path_name + setters) keep single-voice callers/tests
working; `set()` routes global keys to globals.

**New:** `resonance/compose.py` — `render_voices(specs, dur, noise, master)`
offline multi-voice mixer (spatial via render_path, classic via generators.sam).

**TUI rewrite (`tui.py`):** voice-rack mixer. Rack bar top; selected voice's
knobs below; GLOBAL section; peak meter. Keys: tab/v/V select · a add · x mute ·
X delete · e engine · p/P path · m/M mode · 1-5 band · -/= master vol · [/] noise.

**Verification (ALL PASS):**
- offline: two carriers (220+6Hz sidebands, 400+40Hz sidebands) coexist; voices
  spatially split (A +1.3 dB left, B -2.7 dB right). Note: low-carrier ILD is
  physically small so RMS split is modest — ITD carries the percept.
- realtime: 3-voice rack finite + clickless + all 3 carriers present.
- single-voice test still green; live 'pulse' stream runs 3 voices with
  select/mute/path/noise/remove under load.
Figure: `out/multivoice_proof.png`. Demos: `out/chord_theta_gamma.wav`,
`out/chord_triad.wav`.

### Open threads / next candidates
- [x] Multi-voice rack (up to 4 independent SAM voices). (S4)
- [x] SAM depth: parametric 3D paths + true-ITD spatializer + hemisphere aim. (S3)
- [ ] Multi-voice: N simultaneous SAM sources (patent's "multiple paths at
      different freqs" — e.g. theta-left + gamma-right). Next obvious SAM step.
- [ ] HRTF rendering (real externalization + elevation front/back) — needs an
      HRTF dataset (CIPIC/MIT-KEMAR/SADIE). Bigger lift, next fidelity tier.
- [ ] EPOC closed-loop via /mnt/fast/emokit — measure ASSR / frequency-following,
      compare classic vs spatial vs itd_gain, adapt stimulus.
- [ ] Session sequencing / presets / recording live sessions to FLAC.
- [ ] Radial (distance) modulation path; non-sinusoidal traversal velocity.
- [ ] Sequoyah to LISTEN to `out/*.wav` and report subjective effect (esp. SAM
      spatial swirl + whether gamma "buzzes"). His ears are the real ground truth.
- [ ] Decide interaction model: offline session-renderer vs real-time live-knob
      engine (sounddevice callback). Changes architecture.
- [ ] Session layer: sequencing + frequency ramps (Focus-10 -> Focus-12 style),
      presets, layering modalities (SAM + pink noise + pad).
- [ ] SAM depth: HRTF-based true spatialization; non-sinusoidal arc paths;
      hemisphere-targeting experiments (bias_deg).
- [ ] Possible EEG closed-loop (Muse/OpenBCI) to actually measure entrainment.
- [ ] Polish: CLI, config files for sessions, FLAC export, long-render streaming.
