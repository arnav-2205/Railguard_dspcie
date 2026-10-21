"""Segment a run and compute the four spectral features per channel."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import dsp
from .track import speed_ms


CRUISE_FRAC = 0.9   # segment is 'cruise' if speed >= 90 % of the run's median speed


def segment_starts(n: int, seg_len: int = C.SEG_LEN, hop: int = C.HOP):
    return np.arange(0, n - seg_len + 1, hop)


def extract_run_features(df: pd.DataFrame, run_id: str, ref=None, session: str = "unknown",
                         seg_len=C.SEG_LEN, hop=C.HOP, window=C.WINDOW,
                         detrend=C.DETREND, fmax=C.FMAX) -> pd.DataFrame:
    """One row per segment. Segments containing any invalid sample are dropped."""
    starts = segment_starts(len(df), seg_len, hop)
    valid = df["valid"].to_numpy()
    csum = np.concatenate([[0], np.cumsum(~valid)])
    ok = (csum[starts + seg_len] - csum[starts]) == 0
    starts = starts[ok]

    lat, lon, t = df["lat"].to_numpy(), df["lon"].to_numpy(), df["t"].to_numpy()
    v = speed_ms(lat, lon, t)
    mid = starts + seg_len // 2
    rows = {"run": run_id, "group": C.group_of(run_id), "session": session, "seg": np.arange(len(starts)),
            "start": starts, "t_mid": t[mid], "lat": lat[mid], "lon": lon[mid], "speed": v[mid]}
    if ref is not None:
        s, off = ref.project(lat[mid], lon[mid])
        rows["s"], rows["offset"] = s, off
    out = pd.DataFrame(rows)
    # cruise flag: exclude accel/brake phases and standstill (D11)
    cruise_v = float(np.median(v[mid]))
    out["cruise"] = (out["speed"] >= CRUISE_FRAC * cruise_v) & (cruise_v > 2.0)

    for ch in C.CHANNELS:
        x = df[f"ch{ch}"].to_numpy(dtype=np.float64)
        feats = [dsp.segment_features(x[a:a + seg_len], C.FS, window, None, detrend, fmax)
                 for a in starts]
        for name in C.FEATURES:
            out[f"c{ch}_{name}"] = [f[name] for f in feats]
    return out
