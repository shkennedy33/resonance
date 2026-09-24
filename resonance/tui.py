"""Live control surface for the multi-voice SAM engine — a voice-rack mixer.

Run:  .venv/bin/python -m resonance.tui      (or:  .venv/bin/python live.py)

Keys
----
  VOICE RACK
    tab / v        next voice        V        previous voice
    a              add voice         x        mute/unmute       X   delete voice
  SELECTED VOICE
    up/down (j/k)  select knob       left/right (h/l)  turn knob
    e              engine spatial/classic
    p / P          cycle path (spatial)      m / M   cycle arc mode (classic)
    1 2 3 4 5      f_mod -> delta/theta/alpha/beta/gamma
  GLOBAL
    - / =          master volume     [ / ]    pink-noise bed
  PRESETS & SESSIONS
    w              write (save) the rack as a preset
    o              open a preset (glides there over 3 s)
    s              start a session / stop the running one
  space play/pause     q quit

Launch straight into something:
    live.py --preset theta-gamma        live.py --session tour
"""
from __future__ import annotations

import curses
import time

from .engine import (Engine, VOICE_PARAMS, GLOBAL_PARAMS, MODES, ENGINE_MODES,
                     BANDS, MAX_VOICES)
from .paths import PATH_NAMES
from .presets import list_presets, load_preset, save_preset
from .session import list_sessions, load_session, fmt_time

_MODE_HINT = {
    "phase": "L/R phase swing (patent-faithful)",
    "natural": "phase swing + loudness follows position",
    "circular": "source orbits your head",
    "figure8": "source traces a figure-8",
}
_BAND_KEYS = {"1": "delta", "2": "theta", "3": "alpha", "4": "beta", "5": "gamma"}


def _bar(frac, width, fill="█", empty="─"):
    frac = max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return fill * n + empty * (width - n)


def _draw(stdscr, eng, psel, msg=""):
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    snap = eng.snapshot()
    voices, sel = snap["voices"], snap["sel"]
    detail, glob = snap["detail"], snap["globals"]
    tgt = detail["target"]
    barw = max(10, min(34, w - 40))

    def put(y, x, s, attr=0):
        if 0 <= y < h and 0 <= x < w:
            stdscr.addnstr(y, x, s, max(0, w - x - 1), attr)

    status = "▶ playing" if snap["running"] else "⏸ paused"
    put(0, 2, "resonance — multi-voice SAM", curses.A_BOLD)
    put(0, max(30, w - 12), status, curses.A_BOLD |
        (curses.color_pair(2) if snap["running"] else curses.color_pair(3)))

    # ---- voice rack bar ----
    put(1, 2, "voices:", curses.A_BOLD)
    x = 11
    for i, v in enumerate(voices):
        tok = f"{i+1}:{v['f_mod']:.0f}Hz {v['path'][:4]} {v['gain']:.2f}"
        if v["muted"]:
            tok = "~" + tok
        tok = " " + tok + " "
        attr = curses.A_REVERSE if i == sel else curses.color_pair(1)
        if v["muted"]:
            attr = attr | curses.A_DIM
        put(1, x, tok, attr)
        x += len(tok) + 1
    if len(voices) < MAX_VOICES:
        put(1, x, "  +add(a)", curses.A_DIM)

    # ---- session / morph status ----
    sess = snap["session"]
    if sess:
        if sess["done"]:
            put(2, 2, f"session {sess['name']}: complete", curses.color_pair(2))
        else:
            frac = sess["t"] / max(sess["length"], 1e-9)
            line = (f"session {sess['name']}  {fmt_time(sess['t'])}/{fmt_time(sess['length'])} "
                    f"[{_bar(frac, 16, '■', '·')}] {sess['scene']}")
            put(2, 2, line, curses.color_pair(2) | curses.A_BOLD)
    elif snap["morphing"]:
        put(2, 2, "gliding…", curses.color_pair(2))

    # ---- selected voice knobs ----
    put(3, 2, f"voice {sel+1}  [{detail['engine_mode']}]"
              + (f"  path:{detail['path']}" if detail['engine_mode'] == 'spatial'
                 else f"  mode:{detail['mode']}")
              + ("   (MUTED)" if detail["muted"] else ""),
        curses.A_BOLD | curses.color_pair(1))
    for i, p in enumerate(VOICE_PARAMS):
        y = 4 + i
        val = tgt[p["key"]]
        frac = (val - p["min"]) / (p["max"] - p["min"] + 1e-9)
        selrow = (i == psel)
        put(y, 2, ("▸ " if selrow else "  ") + f"{p['label']:<12}",
            curses.A_REVERSE if selrow else 0)
        put(y, 18, "[" + _bar(frac, barw) + "]", curses.color_pair(1) if selrow else 0)
        put(y, 20 + barw, f"{val:6.1f} {p['unit']:<3}",
            curses.A_REVERSE if selrow else 0)

    # ---- hint line for path/mode ----
    hy = 4 + len(VOICE_PARAMS)
    if detail["engine_mode"] == "spatial":
        put(hy, 2, "path: " + " ".join(f"[{n}]" if n == detail["path"] else n
                                        for n in PATH_NAMES), curses.A_DIM)
    else:
        put(hy, 2, "mode: " + _MODE_HINT.get(detail["mode"], ""), curses.A_DIM)

    # ---- globals ----
    gy = hy + 2
    put(gy, 2, "GLOBAL", curses.A_BOLD)
    for j, p in enumerate(GLOBAL_PARAMS):
        val = glob[p["key"]]
        frac = (val - p["min"]) / (p["max"] - p["min"] + 1e-9)
        put(gy + 1 + j, 4, f"{p['label']:<11}")
        put(gy + 1 + j, 18, "[" + _bar(frac, barw) + "]", curses.color_pair(2))
        put(gy + 1 + j, 20 + barw, f"{val:4.2f}")

    # ---- peak meter ----
    py = gy + 1 + len(GLOBAL_PARAMS) + 1
    peak = snap["peak"]; clip = peak > 0.99
    put(py, 2, "output")
    put(py, 18, "[" + _bar(peak, barw) + "]",
        curses.color_pair(3) if clip else curses.color_pair(2))
    put(py, 20 + barw, "CLIP!" if clip else f"{peak:0.2f}",
        curses.color_pair(3) if clip else 0)

    # ---- footer ----
    fy = py + 2
    put(fy, 2, "tab voice · a add · x mute · e engine · p path · m mode · 1-5 band",
        curses.A_DIM)
    put(fy + 1, 2, "↑↓ knob · ←→ adjust · -=vol · []noise · space play · q quit",
        curses.A_DIM)
    put(fy + 2, 2, "w save preset · o open preset · s session start/stop",
        curses.A_DIM)
    put(fy + 3, 2, "⚠ headphones. not while driving. epilepsy = don't.",
        curses.color_pair(3))
    if msg:
        put(fy + 5, 2, msg, curses.A_BOLD)
    stdscr.refresh()


def _pick(stdscr, title, items):
    """Modal list picker. Returns the chosen item or None. Audio keeps running."""
    if not items:
        return None
    i = 0
    while True:
        h, w = stdscr.getmaxyx()
        top = 3
        for r in range(len(items) + 3):
            if top + r < h:
                stdscr.addnstr(top + r, 4, " " * 44, w - 5)
        stdscr.addnstr(top, 4, f" {title} (enter ok · esc cancel) ", w - 5,
                       curses.A_BOLD | curses.A_REVERSE)
        for j, it in enumerate(items):
            if top + 2 + j < h:
                stdscr.addnstr(top + 2 + j, 6, f" {it} ", w - 7,
                               curses.A_REVERSE if j == i else curses.color_pair(1))
        stdscr.refresh()
        c = stdscr.getch()
        if c in (curses.KEY_UP, ord("k")):
            i = (i - 1) % len(items)
        elif c in (curses.KEY_DOWN, ord("j")):
            i = (i + 1) % len(items)
        elif c in (10, 13, curses.KEY_ENTER):
            return items[i]
        elif c in (27, ord("q")):
            return None


def _prompt(stdscr, label):
    h, w = stdscr.getmaxyx()
    y = min(h - 1, 3)
    stdscr.addnstr(y, 4, " " * 50, w - 5)
    stdscr.addnstr(y, 4, label, w - 5, curses.A_BOLD | curses.A_REVERSE)
    curses.echo(); curses.curs_set(1); stdscr.timeout(-1)
    try:
        raw = stdscr.getstr(y, 5 + len(label), 40)
    finally:
        curses.noecho(); curses.curs_set(0); stdscr.timeout(50)
    return raw.decode(errors="ignore").strip()


def _loop(stdscr, eng, preset=None, session=None):
    curses.curs_set(0)
    stdscr.timeout(50)
    if curses.has_colors():
        curses.start_color(); curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_RED, -1)

    psel = 0
    msg, msg_until = "", 0.0

    def say(text):
        nonlocal msg, msg_until
        msg, msg_until = text, time.monotonic() + 4.0

    if preset:
        eng.morph_to(load_preset(preset), 0)
    if session:
        eng.play_session(load_session(session))
    eng.start()
    was_done = False
    try:
        while True:
            if time.monotonic() > msg_until:
                msg = ""
            if eng.session_done and not was_done and eng.running:
                eng.stop()
                say("session complete — space to play on, s for another")
            was_done = eng.session_done
            _draw(stdscr, eng, psel, msg)
            try:
                c = stdscr.getch()
            except curses.error:
                c = -1
            if c == -1:
                continue
            if c in (ord("q"), 27):
                break
            elif c in (curses.KEY_UP, ord("k")):
                psel = (psel - 1) % len(VOICE_PARAMS)
            elif c in (curses.KEY_DOWN, ord("j")):
                psel = (psel + 1) % len(VOICE_PARAMS)
            elif c in (curses.KEY_LEFT, ord("h")):
                eng.nudge(VOICE_PARAMS[psel]["key"], -1)
            elif c in (curses.KEY_RIGHT, ord("l")):
                eng.nudge(VOICE_PARAMS[psel]["key"], +1)
            elif c in (ord("\t"), ord("v")):
                eng.select(+1)
            elif c == ord("V"):
                eng.select(-1)
            elif c == ord("a"):
                eng.add_voice()
            elif c == ord("x"):
                eng.toggle_mute()
            elif c == ord("X"):
                eng.remove_voice()
            elif c == ord("e"):
                eng.toggle_engine()
            elif c == ord("p"):
                eng.cycle_path(+1)
            elif c == ord("P"):
                eng.cycle_path(-1)
            elif c == ord("m"):
                eng.cycle_mode(+1)
            elif c == ord("M"):
                eng.cycle_mode(-1)
            elif c in (ord("-"), ord("_")):
                eng.nudge_global("volume", -1)
            elif c in (ord("="), ord("+")):
                eng.nudge_global("volume", +1)
            elif c == ord("["):
                eng.nudge_global("noise", -1)
            elif c == ord("]"):
                eng.nudge_global("noise", +1)
            elif c == ord(" "):
                eng.stop() if eng.running else eng.start()
            elif c == ord("w"):
                name = _prompt(stdscr, "save preset as: ")
                if name:
                    try:
                        say(f"saved {save_preset(eng, name).name}")
                    except ValueError as e:
                        say(f"not saved: {e}")
            elif c == ord("o"):
                name = _pick(stdscr, "open preset", list_presets())
                if name:
                    try:
                        eng.morph_to(load_preset(name), 3.0)
                        say(f"gliding to {name}")
                    except (ValueError, FileNotFoundError) as e:
                        say(f"can't load {name}: {e}")
            elif c == ord("s"):
                if eng.session is not None:
                    eng.stop_session()
                    say("session stopped — rack stays where it was")
                else:
                    name = _pick(stdscr, "start session", list_sessions())
                    if name:
                        try:
                            eng.play_session(load_session(name))
                            if not eng.running:
                                eng.start()
                            say(f"session {name} started")
                        except (ValueError, FileNotFoundError) as e:
                            say(f"can't load {name}: {e}")
            elif 0 <= c < 256 and chr(c) in _BAND_KEYS:
                eng.set_band(_BAND_KEYS[chr(c)])
    finally:
        eng.stop()


def main(preset=None, session=None):
    # validate before curses takes over the screen, so errors are readable
    if preset:
        load_preset(preset)
    if session:
        load_session(session)
    curses.wrapper(_loop, Engine(), preset, session)


if __name__ == "__main__":
    main()
