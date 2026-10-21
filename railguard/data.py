"""Loading + quality control of Rail-VIVID run CSVs.

Real schema (differs from the README, see decisions.md D02):
Timestamp (ISO), Channel1..Channel6 (g), Temperature, Humidity, Latitude, Longitude.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

_CH = [f"Channel{i}" for i in C.CHANNELS]


class ValidationAccessError(RuntimeError):
    pass


def run_path(run_id: str) -> Path:
    return C.RAW_DIR / run_id / f"{run_id}.csv"


def load_csv(src, max_gap_fill: int = 10):
    """Load a Rail-VIVID-style CSV (path or file-like). Returns (df, qc).

    df columns: t (s from start), ch1..ch6 (float32, g), lat, lon, valid (bool).
    Handling of bad data:
      * non-finite sample values -> linearly interpolated if the gap is
        <= max_gap_fill samples, otherwise left NaN; either way ``valid`` is
        False for repaired/NaN samples so segments containing them are dropped.
      * timestamp gaps (dt != 0.5 ms +-10 %) -> counted in qc; ``valid`` False
        for the samples after a gap (the segment would otherwise span a hole).
    """
    raw = pd.read_csv(src)
    raw = raw.rename(columns=lambda c: c.replace("_", ""))      # tolerate Channel_1
    missing = [c for c in _CH + ["Latitude", "Longitude", "Timestamp"] if c not in raw]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    ts = pd.to_datetime(raw["Timestamp"], errors="coerce")
    t = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
    df = pd.DataFrame({"t": t})
    for i, c in zip(C.CHANNELS, _CH):
        df[f"ch{i}"] = raw[c].to_numpy(dtype=np.float32)
    df["lat"] = raw["Latitude"].to_numpy()
    df["lon"] = raw["Longitude"].to_numpy()

    chans = [f"ch{i}" for i in C.CHANNELS]
    bad = ~np.isfinite(df[chans].to_numpy()).all(axis=1) | ~np.isfinite(t)
    n_bad = int(bad.sum())
    if n_bad:
        df[chans] = df[chans].interpolate(limit=max_gap_fill, limit_direction="both")
    valid = ~bad
    dt = np.diff(t, prepend=t[0] - 1 / C.FS)
    gap = ~np.isfinite(dt) | (np.abs(dt - 1 / C.FS) > 0.1 / C.FS)
    valid &= ~gap
    df["valid"] = valid
    qc = {"session": str(ts.iloc[0].date()), "start": str(ts.iloc[0]), "n_samples": len(df), "n_nonfinite": n_bad, "n_time_gaps": int(gap.sum()),
          "duration_s": float(t[-1] - t[0]), "pct_valid": float(valid.mean() * 100)}
    return df, qc


def load_run(run_id: str, allow_validation: bool = False):
    """Load a named run. Validation runs are blocked unless explicitly allowed."""
    if run_id in C.VALIDATION and not (allow_validation or os.environ.get("RAILGUARD_ALLOW_VALIDATION") == "1"):
        raise ValidationAccessError(
            f"{run_id} is in the held-out VALIDATION split reserved for the user. "
            "Refusing to load it (set RAILGUARD_ALLOW_VALIDATION=1 only when you, "
            "the owner, run your own validation).")
    return load_csv(run_path(run_id))
