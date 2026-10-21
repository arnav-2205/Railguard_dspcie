"""Common along-track coordinate so segments from different runs (and from both
travel directions) can be compared at the same physical location.

s = distance in metres along a reference polyline (built from the longest TRAIN
run's GPS; GPS geometry only, no labels).  Each GPS fix is projected on the
polyline by nearest neighbour.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

R_EARTH = 6371000.0


def to_xy(lat, lon, lat0=40.2340, lon0=-77.8830):
    """Local equirectangular projection (m); fine over a 2 km corridor."""
    x = np.radians(np.asarray(lon) - lon0) * R_EARTH * np.cos(np.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * R_EARTH
    return np.column_stack([x, y])


class TrackReference:
    def __init__(self, lat, lon, step=1):
        xy = to_xy(lat[::step], lon[::step])
        seg = np.hypot(*np.diff(xy, axis=0).T)
        self.xy = xy
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        # orient so s increases from the end nearest the lower latitude (Point B side)
        if xy[0, 1] > xy[-1, 1]:
            self.s = self.s[-1] - self.s
        self.tree = cKDTree(xy)
        self.length = float(self.s.max())

    def project(self, lat, lon):
        """-> (s [m], offset [m]); offset = distance from the reference line."""
        d, i = self.tree.query(to_xy(lat, lon))
        return self.s[i], d


def speed_ms(lat, lon, t, win=2000):
    """Ground speed (m/s) from GPS, averaged over `win` samples (1 s)."""
    xy = to_xy(lat, lon)
    k = min(win, len(xy) // 2)
    dist = np.hypot(*(xy[k:] - xy[:-k]).T)
    dt = t[k:] - t[:-k]
    v = dist / np.maximum(dt, 1e-9)
    pad = np.full(k, v[0]); return np.concatenate([pad[: k // 2], v, np.full(k - k // 2, v[-1])])[: len(xy)]
