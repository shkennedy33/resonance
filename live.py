"""Launch the live SAM control surface.

    .venv/bin/python live.py                      # blank rack
    .venv/bin/python live.py --session tour       # guided listening tour
    .venv/bin/python live.py --preset theta-gamma
"""
import sys

from resonance.__main__ import main

if __name__ == "__main__":
    main(["live"] + sys.argv[1:])
