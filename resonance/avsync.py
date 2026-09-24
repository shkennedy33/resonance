"""Audio -> visual timing bridge (shared memory, lock-free seqlock).

The audio callback publishes, once per block, everything a light needs to
phase-lock to the sound: WHEN this block's first sample reaches the speakers
(on the system-wide monotonic clock, so another process can read it), the lead
voice's mod phase at that sample, its effective rate, and its strike settings.
The flicker process extrapolates the phase to any instant:

    phase(t) = mphase + 2*pi * f_mod * (t - dac_mono)

Seqlock: the writer makes `seq` odd while writing, even when done; a reader
retries if it sees an odd or changed seq. No locks in the audio thread.
"""
from __future__ import annotations

import time
from multiprocessing import shared_memory

import numpy as np

FIELDS = ["seq", "dac_mono", "mphase", "f_mod", "pulse", "decay", "hits",
          "hit_at", "nest", "running", "writer_alive"]
_IDX = {k: i for i, k in enumerate(FIELDS)}
SIZE = 8 * len(FIELDS)


class AVWriter:
    def __init__(self):
        self.shm = shared_memory.SharedMemory(create=True, size=SIZE)
        self.a = np.ndarray((len(FIELDS),), dtype=np.float64, buffer=self.shm.buf)
        self.a[:] = 0.0
        self.name = self.shm.name

    def publish(self, **kw):
        a = self.a
        a[0] += 1.0                         # odd: writing
        for k, v in kw.items():
            a[_IDX[k]] = v
        a[_IDX["writer_alive"]] = time.monotonic()
        a[0] += 1.0                         # even: done

    def close(self):
        try:
            self.a = None
            self.shm.close()
            self.shm.unlink()
        except (FileNotFoundError, BufferError):
            pass


class AVReader:
    def __init__(self, name):
        try:        # reader must not "own" (and later unlink) the writer's block
            self.shm = shared_memory.SharedMemory(name=name, track=False)
        except TypeError:                     # Python < 3.13
            self.shm = shared_memory.SharedMemory(name=name)
        self.a = np.ndarray((len(FIELDS),), dtype=np.float64, buffer=self.shm.buf)
        self.last = None

    def read(self):
        for _ in range(8):
            s0 = self.a[0]
            snap = self.a.copy()
            if s0 == self.a[0] and int(s0) % 2 == 0:
                self.last = {k: float(snap[i]) for k, i in _IDX.items()}
                break
        return self.last

    def close(self):
        self.a = None
        self.shm.close()


def dac_monotonic(time_info, fallback_latency=0.03):
    """Monotonic time at which the current callback's first sample plays."""
    now = time.monotonic()
    try:
        ahead = time_info.outputBufferDacTime - time_info.currentTime
        if 0.0 <= ahead < 0.5 and time_info.currentTime > 0:
            return now + ahead
    except AttributeError:
        pass
    return now + fallback_latency
