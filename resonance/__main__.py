"""resonance command line.

    python -m resonance live [--preset NAME | --session NAME]   live TUI
    python -m resonance render SESSION [-o out.flac]            session -> audio file
    python -m resonance render --preset NAME [-d 60] [-o ...]   preset  -> audio file
    python -m resonance list                                    presets & sessions
"""
from __future__ import annotations

import argparse
import sys
import time


def _list(_):
    from .presets import list_presets, load_preset, PRESET_DIR
    from .session import list_sessions, load_session, fmt_time
    import json
    print("presets  (" + str(PRESET_DIR) + ")")
    for n in list_presets():
        note = json.loads((PRESET_DIR / f"{n}.json").read_text()).get("note", "")
        vs = load_preset(n)["voices"]
        print(f"  {n:<16} {len(vs)} voice{'s' if len(vs) > 1 else ' '}  {note}")
    print("\nsessions")
    for n in list_sessions():
        s = load_session(n)
        print(f"  {n:<16} {fmt_time(s.length):>6}  {len(s.scenes)} scenes  {s.note}")


def _live(a):
    from .tui import main
    main(preset=a.preset, session=a.session)


def _render(a):
    from .core import write
    t0 = time.time()
    if a.preset:
        from .presets import load_preset, render_state
        L, R = render_state(load_preset(a.preset), dur=a.duration)
        out = a.output or f"out/{a.preset}.flac"
        write(out, L, R)
    else:
        if not a.session:
            sys.exit("render needs a SESSION name or --preset NAME")
        from .session import load_session, render_session, fmt_time
        s = load_session(a.session)
        out = a.output or f"out/{s.name}.flac"

        def prog(t, total):
            print(f"\r  {fmt_time(t)} / {fmt_time(total)}", end="", flush=True)
        render_session(s, out, progress=prog)
        print()
    print(f"wrote {out}  ({time.time() - t0:.1f}s)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="resonance", description="open SAM / entrainment suite")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("live", help="live control surface (TUI)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--preset", help="start from this preset")
    g.add_argument("--session", help="start this session")
    p.set_defaults(fn=_live)

    p = sub.add_parser("render", help="render a session or preset to a file")
    p.add_argument("session", nargs="?", help="session name or path")
    p.add_argument("--preset", help="render a preset instead")
    p.add_argument("-d", "--duration", type=float, default=60.0, help="preset length, s")
    p.add_argument("-o", "--output", help="output path (.flac or .wav)")
    p.set_defaults(fn=_render)

    p = sub.add_parser("list", help="list presets and sessions")
    p.set_defaults(fn=_list)

    a = ap.parse_args(argv)
    if not getattr(a, "fn", None):
        a = ap.parse_args(["live"] + (argv or sys.argv[1:]))
    a.fn(a)


if __name__ == "__main__":
    main()
