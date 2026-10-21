"""Simulated real-time processing: segments are consumed one hop at a time and the
features / score are computed on arrival from the raw samples.

Causality notes (what differs from the batch pipeline, see decisions.md D19):
  * cruise flag uses the RUNNING median speed so far (batch uses the whole-run median);
  * a segment's channel-health uses its neighbours (dropout edge rule), so the
    segment i-1 is scored when segment i arrives -> latency = one hop (0.256 s);
  * GPS speed comes from the precomputed smoothed track speed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import baseline as B
from . import config as C
from . import dsp


def _features_for(df: pd.DataFrame, a: int) -> dict:
    out = {}
    for ch in C.CHANNELS:
        x = df[f"ch{ch}"].to_numpy(dtype=np.float64)[a:a + C.SEG_LEN]
        f = dsp.segment_features(x, C.FS, C.WINDOW, None, C.DETREND, C.FMAX)
        for k, v in f.items():
            out[f"c{ch}_{k}"] = v
    return out


def stream_segments(df: pd.DataFrame, meta: pd.DataFrame, model: dict, session: str,
                    run_name: str = "live"):
    """Yield one dict per scored segment, in time order.

    meta: rows with columns start, t_mid, speed (from features.extract_run_features;
          only these metadata columns are used - features are recomputed here).
    """
    bl, rthr, hthr = model["baseline"], model["review_thr"], model["high_thr"]
    rows, speeds = [], []

    def score_target(j):
        lo = max(0, j - 1)
        sub = pd.DataFrame(rows[lo:j + 2])
        sc, nch = bl.score(sub)
        k = j - lo
        s = float(sc[k]) if not np.isnan(sc[k]) else float("nan")
        cls = B.classify(np.array([s]), rthr, hthr)[0]
        r = rows[j]
        return {"seg": j, "start": r["start"], "t_mid": r["t_mid"], "speed": r["speed"],
                "cruise": r["cruise"], "score": s, "nch": int(nch[k]), "class": cls}

    for i, m in enumerate(meta.itertuples(index=False)):
        a = int(m.start)
        speeds.append(float(m.speed))
        med = float(np.median(speeds))
        row = {"run": run_name, "group": "live", "session": session, "start": a,
               "t_mid": float(m.t_mid), "speed": float(m.speed),
               "cruise": bool(m.speed >= 0.9 * med and med > 2.0)}
        row.update(_features_for(df, a))
        rows.append(row)
        if i >= 1:
            yield score_target(i - 1)
    if rows:
        yield score_target(len(rows) - 1)
