"""Reference baseline + anomaly score.

Per collection-session (date) group, fit on healthy TRAIN cruise segments:
    x = [log RMS, log spectral energy, log spectral centroid, log dominant freq]
        for each scoring channel (Ch1-4)                               (16-D)
    mu, Sigma (Ledoit-Wolf shrinkage; complete-case segments = all 4 channels healthy)
Score = sqrt( Mahalanobis^2 / d ), d = number of features of the *healthy* channels
in that segment (Gaussian marginal: sub-vector of mu, sub-block of Sigma).
Typical segment ~1; dead channels are dropped from the score, not scored as 'zero'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from . import config as C

EPS = 1e-9
DEAD_RMS = 0.005       # g; sensor noise floor is ~3e-4 g, healthy vibration >= ~0.05 g (D12)
NF = len(C.FEATURES)


def channel_health(df: pd.DataFrame, channels=C.SCORE_CHANNELS) -> np.ndarray:
    """(n_seg, n_ch) bool; True = channel healthy. A channel is unhealthy in a segment
    if its RMS < DEAD_RMS or in either neighbouring segment of the same run (the
    50 %-overlapping neighbour shares half the samples, so a dropout edge lands in it)."""
    ok = np.ones((len(df), len(channels)), bool)
    runs = df["run"].to_numpy()
    for j, ch in enumerate(channels):
        dead = (df[f"c{ch}_rms"].to_numpy() < DEAD_RMS)
        same_prev = np.concatenate([[False], runs[1:] == runs[:-1]])   # neighbour in same run
        same_next = np.concatenate([runs[:-1] == runs[1:], [False]])
        prev = np.concatenate([[False], dead[:-1]]) & same_prev
        nxt = np.concatenate([dead[1:], [False]]) & same_next
        ok[:, j] = ~(dead | prev | nxt)
    return ok


def feature_matrix(df: pd.DataFrame, channels=C.SCORE_CHANNELS) -> np.ndarray:
    cols = [np.log(df[f"c{ch}_{f}"].to_numpy(float) + EPS) for ch in channels for f in C.FEATURES]
    return np.column_stack(cols)


class Baseline:
    def __init__(self, channels=C.SCORE_CHANNELS, key: str = "session", min_fit: int = 200):
        self.channels = list(channels)
        self.min_fit = min_fit
        self.key = key          # column that defines a baseline group (D13)
        self.models: dict[str, tuple[np.ndarray, np.ndarray]] = {}   # key -> (mu, Sigma)

    def _fit_one(self, part):
        X = feature_matrix(part, self.channels)
        lw = LedoitWolf().fit(X)
        return lw.location_, lw.covariance_

    def fit(self, train: pd.DataFrame):
        ok = channel_health(train, self.channels).all(axis=1)
        t = train[train["cruise"].to_numpy() & ok]
        self.models["__global__"] = self._fit_one(t)
        for g, part in t.groupby(self.key):
            if len(part) >= self.min_fit:
                self.models[g] = self._fit_one(part)
        self.n_fit = len(t)
        return self

    def score(self, df: pd.DataFrame, group: str | None = None):
        """-> (score array [NaN if unscorable], n_channels_used array)."""
        X = feature_matrix(df, self.channels)
        ok = channel_health(df, self.channels)
        groups = df[self.key].to_numpy() if group is None else np.full(len(df), group)
        cruise = df["cruise"].to_numpy()
        out = np.full(len(df), np.nan)
        used = ok.sum(axis=1)
        pats = [tuple(r) for r in ok]
        for g in np.unique(groups):
            mu, S = self.models.get(g, self.models["__global__"])
            for pat in set(pats):
                m = (groups == g) & np.array([p == pat for p in pats]) & cruise
                if not m.any() or not any(pat):
                    continue
                idx = np.concatenate([np.arange(j * NF, (j + 1) * NF) for j, o in enumerate(pat) if o])
                diff = X[m][:, idx] - mu[idx]
                P = np.linalg.inv(S[np.ix_(idx, idx)])
                out[m] = np.sqrt(np.einsum("ij,jk,ik->i", diff, P, diff) / len(idx))
        return out, used


def loro_scores(train: pd.DataFrame, channels=C.SCORE_CHANNELS, key: str = "session"):
    """Leave-one-run-out scores for TRAIN segments (non-optimistic)."""
    out = np.full(len(train), np.nan); used = np.zeros(len(train), int)
    for run in train["run"].unique():
        m = (train["run"] == run).to_numpy()
        b = Baseline(channels, key).fit(train[~m])
        out[m], used[m] = b.score(train[m])
    return out, used


def classify(score, review_thr, high_thr):
    s = np.asarray(score, float)
    lab = np.where(s >= high_thr, "HIGH DEVIATION",
                   np.where(s >= review_thr, "REVIEW RECOMMENDED", "NORMAL-LIKE"))
    return np.where(np.isnan(s), "N/A (not at cruise speed / no healthy channel)", lab)
