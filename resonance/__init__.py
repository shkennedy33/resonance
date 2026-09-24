"""
resonance — an open, inspectable suite of brainwave-entrainment audio tools
inspired by the Monroe Institute (Hemi-Sync, SAM), built from first principles.

Modalities:
    - SAM   (Spatial Angle Modulation) — patent US20130010967A1 equations
    - binaural beats
    - monaural beats
    - isochronic tones
    - noise/pad beds (pink, brown)

Everything renders to plain numpy arrays (float64, range ~[-1, 1]) as (L, R)
tuples, so any generator can be layered, sequenced, analyzed, or written to disk.
"""
from .core import SR, timeline, fade, stereo, mix, normalize, write, seconds
from .generators import (
    sam,
    binaural,
    monaural,
    isochronic,
    pink_noise,
    brown_noise,
    tone,
)

__all__ = [
    "SR", "timeline", "fade", "stereo", "mix", "normalize", "write", "seconds",
    "sam", "binaural", "monaural", "isochronic", "pink_noise", "brown_noise", "tone",
]
