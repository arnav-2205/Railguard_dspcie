"""Extract per-segment features for TRAIN + TEST runs (never validation).
Output: outputs/features.csv.gz and outputs/track_ref.npz"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from railguard import config as C, data, features, track


def main():
    C.OUT_DIR.mkdir(exist_ok=True)
    runs = C.TRAIN + C.TEST
    # reference polyline from the TRAIN run with the longest GPS path (geometry only)
    best, best_len = None, -1
    cache = {}
    for r in C.TRAIN:
        df, qc = data.load_run(r)
        xy = track.to_xy(df.lat.values[::200], df.lon.values[::200])
        L = np.hypot(*np.diff(xy, axis=0).T).sum()
        cache[r] = (df, qc)
        print(f"{r:14s} path {L:7.0f} m  dur {qc['duration_s']:6.1f}s  nonfinite {qc['n_nonfinite']}  gaps {qc['n_time_gaps']}")
        if L > best_len: best, best_len = r, L
    df_ref = cache[best][0]
    ref = track.TrackReference(df_ref.lat.values, df_ref.lon.values, step=200)
    print("reference run:", best, f"length {ref.length:.0f} m")
    np.savez(C.OUT_DIR / "track_ref.npz", xy=ref.xy, s=ref.s, run=best)

    parts = []
    for r in runs:
        df, qc = cache[r] if r in cache else data.load_run(r)
        ft = features.extract_run_features(df, r, ref, session=qc['session'])
        ft["split"] = "train" if r in C.TRAIN else "test"
        parts.append(ft)
        print(f"{r:14s} segs {len(ft):4d} cruise {int(ft.cruise.sum()):4d} cruise-speed {ft.speed[ft.cruise].median():.2f} m/s  session {qc['session']}")
    pd.concat(parts, ignore_index=True).to_csv(C.OUT_DIR / "features.csv.gz", index=False)


if __name__ == "__main__":
    main()
