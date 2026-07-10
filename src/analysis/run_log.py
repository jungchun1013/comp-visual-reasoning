"""Tee stdout/stderr into <out_dir>/log.txt.

Every analysis run keeps its log INSIDE its own output folder (user rule,
2026-07-08) — no more launcher-side redirection scattering logs one level up.
New module because src/analysis had no IO/logging utility to extend.

Usage, first thing after the output dir is known:
    from analysis.run_log import tee_stdout
    tee_stdout(out_dir)
"""
from __future__ import annotations

import atexit
import sys
from datetime import datetime
from pathlib import Path


class _Tee:
    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, s):
        self._stream.write(s)
        if not self._fh.closed:  # atexit closes fh before final interpreter flush
            self._fh.write(s)
        return len(s)

    def flush(self):
        self._stream.flush()
        if not self._fh.closed:
            self._fh.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def tee_stdout(out_dir):
    """Append stdout+stderr of this run to <out_dir>/log.txt (creates dir)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / "log.txt", "a")
    fh.write(f"=== {datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(sys.argv)} ===\n")
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    atexit.register(fh.close)
    return fh
