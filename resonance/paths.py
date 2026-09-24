"""Parametric spatial paths — the geometry SAM travels.

A *path* is a function of phase theta (radians) that returns the source position
as (azimuth, elevation) in radians:
    azimuth   0 = straight ahead, +right / -left, wraps at +/-pi (behind)
    elevation 0 = ear level,      +up / -down

The source completes one full traversal of the path per 1/f_mod second, so the
SHAPE sets the character of the motion and f_mod sets its rate. `extent` scales
the angular size of the figure (the patent's "peak phase deviation" generalized
from a scalar into the size of an arbitrary curve).

All paths are plain math on numpy arrays, so you can plot the (az, el) trace to
literally see the shape before you ever hear it.
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2 * np.pi


def pendulum(theta, extent=1.2):
    """Left<->right swing through center — the classic SAM arc. Crosses center
    twice per cycle (tremolo at 2*f_mod)."""
    return extent * np.sin(theta), np.zeros_like(theta)


def arc_front(theta, extent=0.9):
    """A shallow arc that stays in front (never goes behind). Gentle."""
    return extent * np.sin(theta), 0.15 * extent * (np.cos(theta) - 1.0)


def orbit(theta, extent=1.0):
    """Full horizontal revolution around the head — passes behind you. `extent`
    is ignored for angle (it's a full circle); it scales nothing here."""
    az = np.arctan2(np.sin(theta), np.cos(theta))  # = wrapped theta, full 360
    return az, np.zeros_like(theta)


def halo(theta, extent=1.0):
    """Vertical circle in the frontal plane — a ring spinning in front of you."""
    return extent * np.sin(theta), extent * np.cos(theta)


def figure8(theta, extent=1.0):
    """Lissajous 1:2 — the infinity sign. Level moves at twice the azimuth rate."""
    return extent * np.sin(theta), extent * np.sin(2 * theta) * 0.6


def lissajous(theta, a=3, b=2, delta=np.pi / 2, extent=1.0):
    """General Lissajous figure. Different (a, b) integer ratios = different knots."""
    return extent * np.sin(a * theta + delta), extent * 0.6 * np.sin(b * theta)


def rose(theta, k=3, extent=1.1):
    """Rose / rosette: r = cos(k*theta). Odd k -> k petals, even k -> 2k petals."""
    r = extent * np.cos(k * theta)
    return r * np.cos(theta), r * 0.6 * np.sin(theta)


def spinner(theta, extent=1.0):
    """Horizontal figure that darts side to side but always faces front-ish —
    az swings wide, elevation dips at the extremes (a 'smile' path)."""
    return extent * np.sin(theta), -0.4 * extent * np.abs(np.sin(theta))


# name -> (function, default kwargs). The registry the engine/TUI iterate over.
PATHS = {
    "pendulum": (pendulum, {}),
    "arc_front": (arc_front, {}),
    "orbit": (orbit, {}),
    "halo": (halo, {}),
    "figure8": (figure8, {}),
    "lissajous": (lissajous, {}),
    "rose": (rose, {}),
    "spinner": (spinner, {}),
}

PATH_NAMES = list(PATHS.keys())


def get_path(name):
    """Return a zero-arg-friendly callable az,el = f(theta) for a named path,
    with its default kwargs baked in."""
    fn, kw = PATHS[name]
    return lambda theta, **over: fn(theta, **{**kw, **over})
