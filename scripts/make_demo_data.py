"""Build demo CSVs for the Streamlit upload box (from a TEST run, never validation).

demo_data/demo_clean.csv                  unmodified 75 s excerpt (real data)
demo_data/demo_with_injected_events.csv   same excerpt + 4 synthetic impact events (known times)
demo_data/expected_output_*.csv           what the live pipeline outputs for each file
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pickle
import numpy as np, pandas as pd
from railguard import config as C, data, evaluate as E, features, live

RUN = "BtoA_25_3"            # TEST run, 2025-04-11 (session with 12 training runs -> solid baseline)
T0, T1 = 40.0, 115.0         # excerpt seconds (in cruise)
EVENTS = [(15.0, 3.2), (30.0, 3.2), (45.0, 1.6), (60.0, 3.2)]   # (seconds into excerpt, peak g)
OUT = C.ROOT / "demo_data"


def main():
    OUT.mkdir(exist_ok=True)
    raw = pd.read_csv(C.RAW_DIR / RUN / f"{RUN}.csv")
    a, b = int(T0 * C.FS), int(T1 * C.FS)
    clean = raw.iloc[a:b].reset_index(drop=True)
    clean.to_csv(OUT / "demo_clean.csv", index=False)

    inj = clean.copy()
    w = E.event_waveform(1.0)
    for t, amp in EVENTS:
        s = int(t * C.FS)
        for ch in C.SCORE_CHANNELS:
            inj.loc[s:s + len(w) - 1, f"Channel{ch}"] += amp * w
    inj.to_csv(OUT / "demo_with_injected_events.csv", index=False)

    model = pickle.load(open(C.OUT_DIR / "model.pkl", "rb"))
    for name in ("demo_clean", "demo_with_injected_events"):
        df, qc = data.load_csv(OUT / f"{name}.csv")
        ft = features.extract_run_features(df, name, None, session=qc["session"])
        rows = pd.DataFrame(list(live.stream_segments(df, ft[["start", "t_mid", "speed"]], model, qc["session"], name)))
        rows["thr_review"], rows["thr_high"] = model["review_thr"], model["high_thr"]
        rows.round(3).to_csv(OUT / f"expected_output_{name}.csv", index=False)
        flagged = rows[rows["class"].isin(["REVIEW RECOMMENDED", "HIGH DEVIATION"])]
        print(f"{name}: {len(rows)} segments; classes {rows['class'].value_counts().to_dict()}")
        print("   flagged at t(s):", [round(t, 1) for t in flagged.t_mid])
    print("event centres (s into file):", [t + 0.05 for t, _ in EVENTS])


if __name__ == "__main__":
    main()
