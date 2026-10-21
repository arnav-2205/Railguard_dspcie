"""RailGuard Streamlit interface.   Run:  streamlit run app.py

Pipeline behind each tab is documented in system_architecture.md section 7.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import time
import matplotlib.pyplot as plt
from railguard import baseline as B, config as C, data, dsp, features, live, plots

st.set_page_config(page_title="RailGuard", layout="wide")
st.title("RailGuard - FFT-based track vibration anomaly screening")

MODEL_PATH = C.OUT_DIR / "model.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner="Loading + extracting features ...")
def process(source_key: str, file_bytes: bytes | None):
    """Load a run (named or uploaded) -> (df, qc, feature table)."""
    if file_bytes is None:
        df, qc = data.load_run(source_key)
        name = source_key
    else:
        import io
        df, qc = data.load_csv(io.BytesIO(file_bytes))
        name = "uploaded"
    ft = features.extract_run_features(df, name, None, session=qc["session"])
    return df, qc, ft


# ----------------------------------------------------------------- sidebar
st.sidebar.header("1. Data")
mode = st.sidebar.radio("Source", ["Select a run", "Upload CSV"])
file_bytes, source_key = None, None
available = [r for r in C.TRAIN + C.TEST if data.run_path(r).exists()]
if mode == "Select a run":
    if not available:
        st.error("No run CSVs found. Run `python scripts/download_data.py` first.")
        st.stop()
    source_key = st.sidebar.selectbox(
        "Run", available,
        format_func=lambda r: f"{r}  [{'train' if r in C.TRAIN else 'test'}]")
    st.sidebar.caption("The 3 held-out validation runs are intentionally not listed.")
else:
    up = st.sidebar.file_uploader("Rail-VIVID style CSV", type="csv")
    if up is None:
        st.info("Upload a CSV with Timestamp, Channel1..6, Latitude, Longitude (2 kHz).")
        st.stop()
    file_bytes, source_key = up.getvalue(), up.name

try:
    df, qc, ft = process(source_key, file_bytes)
except Exception as e:                       # noqa: BLE001 - show any load problem to the user
    st.error(f"Could not load data: {e}")
    st.stop()

st.sidebar.header("2. Spectrum settings")
ch = st.sidebar.selectbox("Channel", C.CHANNELS, index=1, format_func=lambda c: f"Channel {c}")
win_sel = st.sidebar.multiselect("Windows to overlay", ["hann", "hamming", "rect"], default=["hann", "rect"],
                                 help="'rect' = no window. Compare to see spectral leakage.")
detrend = st.sidebar.radio("Pre-processing", ["constant", "linear", "none"], index=0,
                           format_func={"constant": "remove mean (default)", "linear": "remove mean + linear trend",
                                        "none": "none (shows DC)"}.get)
pad = st.sidebar.select_slider("Zero-padding factor", [1, 2, 4, 8], value=1,
                               help="Interpolates the spectrum; does NOT add true resolution.")
db = st.sidebar.toggle("dB scale", value=True)
st.sidebar.header("3. STFT (spectrogram)")
ovl = st.sidebar.select_slider("Overlap", [0.0, 0.5, 0.75], value=0.5, format_func=lambda v: f"{int(v*100)} %")

# ----------------------------------------------------------------- model + scores
model = load_model()
ft = ft.copy()
if model is None:
    st.warning("No trained model at outputs/model.pkl - run scripts/extract_features.py then "
               "scripts/fit_and_evaluate.py. Scores/classification disabled.")
    ft["score"], ft["nch"], ft["class"] = np.nan, 0, "N/A"
else:
    bl, rthr, hthr = model["baseline"], model["review_thr"], model["high_thr"]
    if ft["session"].iloc[0] not in bl.models:
        st.info(f"No baseline for session {ft['session'].iloc[0]} - using the global TRAIN baseline "
                "(less specific; scores are less reliable).")
    ft["score"], ft["nch"] = bl.score(ft)
    ft["class"] = B.classify(ft["score"].to_numpy(), rthr, hthr)

with st.expander("Data quality report", expanded=False):
    st.json(qc)
    deadfrac = {f"Ch{c}": float((ft[f"c{c}_rms"] < B.DEAD_RMS)[ft.cruise].mean()) for c in C.SCORE_CHANNELS}
    st.write("Fraction of cruise-speed segments where the channel is flat-lined (RMS < "
             f"{B.DEAD_RMS} g, a sensor dropout - not a track event):", deadfrac)

tab_live, tab_t, tab_f, tab_feat, tab_score, tab_spec = st.tabs(
    ["Live demo", "Time domain", "FFT spectrum", "Features", "Anomaly score", "Spectrogram"])

seg_i = st.sidebar.slider("Segment index (for FFT / features)", 0, len(ft) - 1, min(len(ft) // 2, len(ft) - 1))
row = ft.iloc[seg_i]
a = int(row["start"]); b = a + C.SEG_LEN
x_seg = df[f"ch{ch}"].to_numpy(dtype=float)[a:b]
t_all = df["t"].to_numpy()

with tab_t:
    st.pyplot(plots.time_domain(t_all, df[f"ch{ch}"].to_numpy(), t_all[a], t_all[b - 1], f"Ch{ch} (g)"))
    st.caption(f"Segment {seg_i}: t = {t_all[a]:.2f}-{t_all[b-1]:.2f} s, {C.SEG_LEN} samples "
               f"({C.SEG_LEN / C.FS:.3f} s), Δf = fs/N = {C.FS / C.SEG_LEN:.3f} Hz. "
               f"Speed {row['speed']:.1f} m/s, cruise={bool(row['cruise'])}.")

with tab_f:
    c1, c2 = st.columns(2)
    nfft = C.SEG_LEN * pad
    with c1:
        st.pyplot(plots.spectrum(x_seg, C.FS, win_sel or ["hann"], nfft, detrend if detrend != "none" else None, "linear"))
    with c2:
        st.pyplot(plots.spectrum(x_seg, C.FS, win_sel or ["hann"], nfft, detrend if detrend != "none" else None, "dB"))
    te, fe = dsp.parseval_check(x_seg, win_sel[0] if win_sel else "hann", detrend if detrend != "none" else None)
    st.caption(f"Left: linear. Right: dB (20·log10, because amplitude spans >5 decades). nfft = {nfft}, "
               f"bin spacing {C.FS / nfft:.3f} Hz, true resolution still ≈ {C.FS / C.SEG_LEN:.2f} Hz. "
               f"Parseval check (time vs freq energy): {te:.6g} vs {fe:.6g}.")

with tab_feat:
    cols = ["rms", "dom_freq", "spec_energy", "spec_centroid"]
    seg_tbl = pd.DataFrame({f: [row[f"c{c}_{f}"] for c in C.CHANNELS] for f in cols},
                           index=[f"Ch{c}" for c in C.CHANNELS])
    seg_tbl.columns = ["RMS (g)", "Dominant freq (Hz, DC excluded)", "Spectral energy (g²·s, 0-500 Hz)", "Spectral centroid (Hz)"]
    st.subheader(f"Segment {seg_i}")
    st.dataframe(seg_tbl.style.format("{:.4g}"))
    st.subheader("All segments")
    show = ["seg", "t_mid", "speed", "cruise"] + [f"c{ch}_{f}" for f in cols]
    st.dataframe(ft[show], height=300)

with tab_score:
    if model is None:
        st.write("Model not available.")
    else:
        st.write(f"Baseline session: **{row['session']}**. Thresholds from TRAIN leave-one-run-out: "
                 f"REVIEW ≥ {rthr:.2f}, HIGH ≥ {hthr:.2f}.")
        st.pyplot(plots.score_timeline(ft["t_mid"], ft["score"], rthr, hthr))
        counts = ft["class"].value_counts()
        st.write(counts.to_frame("segments"))
        st.write(f"**Selected segment {seg_i}:** score = {row['score']:.2f} "
                 f"using {int(ft.loc[ft.index[seg_i], 'nch'])}/4 healthy channels → **{ft['class'].iloc[seg_i]}**")
        flagged = ft[ft["class"].isin(["REVIEW RECOMMENDED", "HIGH DEVIATION"])]
        st.dataframe(flagged[["seg", "t_mid", "speed", "score", "nch", "class"]], height=250)
        st.caption("A high score means 'unlike this session's normal behaviour'. It may be a track feature OR a "
                   "sensor/vehicle effect; it is a triage flag, not a defect diagnosis. See decisions.md D12-D16.")

with tab_spec:
    seg_len, hop = C.SEG_LEN, int(C.SEG_LEN * (1 - ovl))
    st.markdown(f"**STFT parameters** - window: **Hann**, length **{seg_len}** samples ({seg_len / C.FS * 1000:.0f} ms), "
                f"hop **{hop}** samples ({hop / C.FS * 1000:.0f} ms), overlap **{int(ovl*100)} %**, "
                f"Δf = {C.FS / seg_len:.3f} Hz, fs = {C.FS:.0f} Hz.")
    st.pyplot(plots.spectrogram(df[f"ch{ch}"].to_numpy(dtype=float), C.FS, seg_len, ovl, "hann"))
    st.caption("More overlap = smoother time axis, but adjacent columns are increasingly redundant (they share samples); "
               "it does not improve true time or frequency resolution. Energy above 500 Hz is the anti-alias "
               "low-pass stop-band (data was filtered at 500 Hz).")


# ----------------------------------------------------------------- live demo
with tab_live:
    st.markdown("**Simulated live processing.** The file is replayed as a stream; each new 0.512 s segment is "
                "windowed, FFT'd, turned into the 4 features per channel and scored on arrival "
                "(one-hop = 0.256 s latency, see decisions.md D19).")
    if model is None:
        st.warning("Needs outputs/model.pkl (run scripts/fit_and_evaluate.py).")
    else:
        dur = float(t_all[-1])
        c1, c2, c3, c4 = st.columns(4)
        play_from = c1.number_input("Start at (s)", 0.0, max(dur - 10, 0.0), min(20.0, max(dur - 70, 0.0)), 5.0)
        play_len = c2.number_input("Play length (s)", 10.0, dur, min(60.0, dur - play_from), 10.0)
        mult = c3.select_slider("Speed", [1, 2, 5, 10, 25], value=5, format_func=lambda m: f"{m}x real-time")
        live_ch = c4.selectbox("Plot channel", C.CHANNELS, index=1, key="live_ch")
        go = st.button("▶ Start live processing", type="primary")
        badge, frame, stats = st.empty(), st.empty(), st.empty()
        if go:
            meta = ft[(ft["t_mid"] >= play_from) & (ft["t_mid"] <= play_from + play_len)][["start", "t_mid", "speed"]]
            hop_s = C.HOP / C.FS
            res, t_start = [], time.time()
            for out in live.stream_segments(df, meta, model, ft["session"].iloc[0], "live"):
                t_b = time.time()
                res.append(out)
                color = {"NORMAL-LIKE": "#2e7d32", "REVIEW RECOMMENDED": "#ef6c00", "HIGH DEVIATION": "#c62828"}.get(out["class"], "#607d8b")
                score_txt = "—" if np.isnan(out["score"]) else f"{out['score']:.2f}"
                badge.markdown(f"<div style='padding:10px 16px;border-radius:8px;background:{color};color:white;font-size:20px'>"
                               f"t = {out['t_mid']:.1f} s &nbsp;|&nbsp; score {score_txt} &nbsp;|&nbsp; <b>{out['class']}</b>"
                               f" &nbsp;|&nbsp; healthy channels {out['nch']}/4</div>", unsafe_allow_html=True)
                n = len(res)
                if n % max(1, int(mult)) == 0 or n == len(meta):
                    a = int(out["start"]); b = a + C.SEG_LEN
                    lo = max(0, b - int(10 * C.FS))
                    arr = df[f"ch{live_ch}"].to_numpy(dtype=float)
                    sc = pd.DataFrame(res)
                    fig = plots.live_frame(t_all[lo:b:4], arr[lo:b:4], arr[a:b], sc["t_mid"], sc["score"], rthr, hthr, live_ch)
                    frame.pyplot(fig); plt.close(fig)
                    cnt = sc["class"].value_counts()
                    stats.write({k: int(v) for k, v in cnt.items()} | {"elapsed (s)": round(time.time() - t_start, 1)})
                time.sleep(max(0.0, hop_s / mult - (time.time() - t_b)))
            st.success(f"Finished: {len(res)} segments processed.")
            sc = pd.DataFrame(res)
            flagged = sc[sc["class"].isin(["REVIEW RECOMMENDED", "HIGH DEVIATION"])]
            st.dataframe(flagged[["seg", "t_mid", "speed", "score", "nch", "class"]], height=220)
