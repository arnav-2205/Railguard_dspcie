"""Matplotlib figure builders shared by the Streamlit app and scripts/make_figures.py."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sps

from . import config as C
from . import dsp

CLASS_COLORS = {"NORMAL-LIKE": "#2e7d32", "REVIEW RECOMMENDED": "#ef6c00", "HIGH DEVIATION": "#c62828"}


def time_domain(t, x, seg_t0=None, seg_t1=None, ylabel="g", max_pts=20000):
    step = max(1, len(x) // max_pts)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.plot(t[::step], x[::step], lw=0.5)
    if seg_t0 is not None:
        ax.axvspan(seg_t0, seg_t1, color="orange", alpha=0.35, label="selected segment")
        ax.legend(loc="upper right")
    ax.set_xlabel("time (s)"); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def spectrum(x, fs=C.FS, windows=("hann", "rect"), nfft=None, detrend="constant",
             scale="dB", welch=True, fmax_line=C.FMAX):
    """Overlay one-sided amplitude spectra for each window in `windows`."""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for w, col in zip(windows, ("C0", "C3", "C2")):
        f, A = dsp.amplitude_spectrum(x, fs, w, nfft, detrend)
        y = dsp.to_db(A) if scale == "dB" else A
        ax.plot(f, y, lw=0.9, color=col, label=f"{'no window (rect)' if w == 'rect' else w} window")
    if welch and len(x) >= 256:
        f, P = dsp.welch_psd(x, fs, nperseg=256, overlap=0.5)
        Aw = np.sqrt(P * (fs / 256) * 1.5)       # rough amplitude-equivalent for overlay only
        ax.plot(f, dsp.to_db(Aw) if scale == "dB" else Aw, lw=1.6, color="k", alpha=0.7,
                label="Welch (256-pt, 50 %) - shape only")
    ax.axvline(fmax_line, color="grey", ls="--", lw=0.8)
    ax.text(fmax_line + 5, ax.get_ylim()[1], " 500 Hz anti-alias\n low-pass", va="top", fontsize=8, color="grey")
    ax.set_xlim(0, fs / 2)
    ax.set_xlabel("frequency (Hz)  [0 ... Nyquist = fs/2]")
    ax.set_ylabel("amplitude (dB re 1 g)" if scale == "dB" else "amplitude (g)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def spectrogram(x, fs=C.FS, nperseg=C.SEG_LEN, overlap=0.5, window=C.WINDOW, t0=0.0, fmax=C.FS / 2):
    f, t, S = sps.spectrogram(x - np.mean(x), fs=fs, window=window, nperseg=nperseg,
                              noverlap=int(nperseg * overlap), scaling="density", mode="psd")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    im = ax.pcolormesh(t + t0, f, 10 * np.log10(S + 1e-18), shading="auto", cmap="magma")
    ax.set_ylim(0, fmax); ax.set_xlabel("time (s)"); ax.set_ylabel("frequency (Hz)")
    fig.colorbar(im, ax=ax, label="PSD (dB re 1 g^2/Hz)")
    fig.tight_layout()
    return fig


def score_timeline(t, score, review_thr, high_thr, xlabel="time (s)"):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, score, lw=0.8, color="C0")
    ax.axhline(review_thr, color="#ef6c00", ls="--", label=f"REVIEW >= {review_thr:.2f}")
    ax.axhline(high_thr, color="#c62828", ls="--", label=f"HIGH >= {high_thr:.2f}")
    ax.set_xlabel(xlabel); ax.set_ylabel("anomaly score"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_yscale("symlog", linthresh=3)
    fig.tight_layout()
    return fig


def live_frame(t_win, x_win, seg_x, scores_t, scores, review_thr, high_thr, ch, win_name="hann"):
    """One figure for the live demo: rolling waveform | current-segment spectrum | score so far."""
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.2), gridspec_kw={"width_ratios": [1.2, 1.2, 1.2]})
    ax = axs[0]
    ax.plot(t_win, x_win, lw=0.5)
    ax.set_title(f"Ch{ch} waveform (last 10 s)", fontsize=9); ax.set_xlabel("time (s)"); ax.set_ylabel("g"); ax.grid(alpha=0.3)
    ax = axs[1]
    f, A = dsp.amplitude_spectrum(seg_x, C.FS, win_name)
    ax.plot(f, dsp.to_db(A), lw=0.8, color="C3")
    ax.axvline(C.FMAX, color="grey", ls="--", lw=0.8)
    ax.set_xlim(0, C.FS / 2); ax.set_ylim(-140, 0)
    ax.set_title("Current segment FFT (Hann, dB re 1 g)", fontsize=9); ax.set_xlabel("Hz"); ax.grid(alpha=0.3)
    ax = axs[2]
    ax.plot(scores_t, scores, lw=0.8, color="C0")
    ax.axhline(review_thr, color="#ef6c00", ls="--", lw=1)
    ax.axhline(high_thr, color="#c62828", ls="--", lw=1)
    ax.set_yscale("symlog", linthresh=3); ax.set_title("Anomaly score so far", fontsize=9)
    ax.set_xlabel("time (s)"); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
