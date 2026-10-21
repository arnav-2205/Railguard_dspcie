"""Evaluation.

Rail-VIVID ships NO machine-readable anomaly locations (decisions.md D03/D14), so
the primary evaluation is SEMI-SYNTHETIC: known defect signatures are injected into
real TEST-run signals at known positions, and the whole pipeline (features ->
baseline score -> threshold) is scored against those known positives.

Event model (a rail-joint / dip impact ring-down):
    e(t) = A * exp(-t / tau) * sin(2*pi*f0*t),  tau = 20 ms, f0 = 200 Hz
added to Ch1-4.  A is in g.

Labelling per event (one event -> one positive segment):
  * positive : segment whose CENTRAL half [start+N/4, start+3N/4) contains the event centre
  * ignored  : other segments that overlap the event (partial, edge-attenuated by the window)
  * negative : segments that don't overlap any event
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

TAU, F0, EV_LEN = 0.020, 200.0, 0.100       # s, Hz, s (5 tau)


def event_waveform(amp: float, fs: float = C.FS, f0=F0, tau=TAU, dur=EV_LEN):
    t = np.arange(int(dur * fs)) / fs
    return amp * np.exp(-t / tau) * np.sin(2 * np.pi * f0 * t)


def inject_events(df: pd.DataFrame, amp: float, n_events: int, seed: int, min_gap_s: float = 3.0):
    """Return (modified copy of df, sorted array of event start sample indices)."""
    rng = np.random.default_rng(seed)
    n = len(df)
    lo, hi = int(15 * C.FS), n - int(15 * C.FS)          # keep away from accel/brake phases
    starts, tries = [], 0
    while len(starts) < n_events and tries < 10000:
        tries += 1
        s = int(rng.integers(lo, hi))
        if all(abs(s - o) > min_gap_s * C.FS for o in starts):
            starts.append(s)
    starts = np.sort(np.array(starts, int))
    out = df.copy()
    w = event_waveform(amp).astype(np.float32)
    for s in starts:
        for ch in C.SCORE_CHANNELS:
            out.iloc[s:s + len(w), out.columns.get_loc(f"ch{ch}")] += w
    return out, starts


def label_segments(seg_start: np.ndarray, ev_starts: np.ndarray, seg_len=C.SEG_LEN, ev_len=int(EV_LEN * C.FS)):
    """-> array of {1: positive, 0: negative, -1: ignored}."""
    lab = np.zeros(len(seg_start), int)
    if len(ev_starts) == 0:
        return lab
    centre = ev_starts + ev_len // 2
    for i, a in enumerate(seg_start):
        b = a + seg_len
        if np.any((centre >= a + seg_len // 4) & (centre < a + 3 * seg_len // 4)):
            lab[i] = 1
        elif np.any((ev_starts < b) & (ev_starts + ev_len > a)):
            lab[i] = -1
    return lab


def prf(pred_pos: np.ndarray, label: np.ndarray):
    """Precision/recall/F1/FPR on labelled (non-ignored) segments."""
    m = label >= 0
    y, p = label[m] == 1, np.asarray(pred_pos)[m].astype(bool)
    tp, fp = int((y & p).sum()), int((~y & p).sum())
    fn, tn = int((y & ~p).sum()), int((~y & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) and not np.isnan(prec + rec) else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1, fpr=fpr)


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson 95 % interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)
