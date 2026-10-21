"""Generate documentation figures into outputs/figures/ (train + test runs only)."""
import pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from railguard import config as C, data, dsp, features, plots

FIG = C.OUT_DIR / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def main():
    # --- real-data windowed vs unwindowed (TRAIN run, a mid-run cruise segment) ---
    run = "BtoA_27.5_2"
    df, qc = data.load_run(run)
    ft = features.extract_run_features(df, run, None, session=qc["session"])
    ok = ft[ft.cruise & (ft.c1_rms > 0.2) & (ft.c2_rms > 0.2)]
    a = int(ok.iloc[len(ok) // 2].start)
    out = {}
    for ch in (1, 2):
        x = df[f"ch{ch}"].to_numpy(float)[a:a + C.SEG_LEN]
        f, Ar = dsp.amplitude_spectrum(x, C.FS, "rect")
        _, Ah = dsp.amplitude_spectrum(x, C.FS, "hann")
        hi = (f > 300) & (f <= 480)
        out[ch] = (20 * np.log10(np.median(Ar[hi])), 20 * np.log10(np.median(Ah[hi])))
        plots.spectrum(x, C.FS, ["rect", "hann"], None, "constant", "dB", welch=True).savefig(FIG / f"real_ch{ch}_window_vs_rect_dB.png", dpi=130)
        plots.spectrum(x, C.FS, ["rect", "hann"], None, "constant", "linear", welch=False).savefig(FIG / f"real_ch{ch}_window_vs_rect_linear.png", dpi=130)
    print(f"Real segment ({run}, start sample {a}): median level 300-480 Hz  [rect dB, hann dB]:", {k: tuple(round(v, 1) for v in vv) for k, vv in out.items()})

    plots.spectrogram(df["ch2"].to_numpy(float), C.FS, C.SEG_LEN, 0.5, "hann").savefig(FIG / "spectrogram_ch2_50pct.png", dpi=130)
    plots.spectrogram(df["ch2"].to_numpy(float), C.FS, C.SEG_LEN, 0.75, "hann").savefig(FIG / "spectrogram_ch2_75pct.png", dpi=130)
    # whole-run Welch PSD showing the 500 Hz low-pass
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for ch in (1, 2, 3, 4):
        f, P = dsp.welch_psd(df[f"ch{ch}"].to_numpy(float), C.FS, nperseg=2048)
        ax.semilogy(f[1:], P[1:], label=f"Ch{ch}")
    ax.axvline(500, color="k", ls="--"); ax.set_xlabel("Hz"); ax.set_ylabel("PSD g$^2$/Hz"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "welch_psd_500Hz_lowpass.png", dpi=130)

    # --- test-run score timeline ---
    m = pickle.load(open(C.OUT_DIR / "model.pkl", "rb"))
    r = "BtoA_27.5_3"
    d2, q2 = data.load_run(r)
    f2 = features.extract_run_features(d2, r, None, session=q2["session"])
    f2["score"], _ = m["baseline"].score(f2)
    plots.score_timeline(f2.t_mid, f2.score, m["review_thr"], m["high_thr"]).savefig(FIG / "test_run_score_timeline.png", dpi=130)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
