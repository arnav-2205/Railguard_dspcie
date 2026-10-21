"""Core DSP routines for RailGuard.

Every function here is validated against synthetic signals in
tests/test_dsp_synthetic.py before it is ever applied to real data.

Conventions
-----------
* x       : 1-D real signal segment (units of g in Rail-VIVID)
* fs      : sampling rate in Hz (2000 for Rail-VIVID)
* N       : number of samples in the segment (NOT the zero-padded length)
* nfft    : FFT length actually used (>= N; equals N when no zero padding)
* All spectra are ONE-SIDED (0 ... fs/2) and come from np.fft.rfft.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps


# --------------------------------------------------------------------------
# Pre-processing
# --------------------------------------------------------------------------
def preprocess(x: np.ndarray, detrend: str = "constant") -> np.ndarray:
    """Remove DC ('constant') or DC + linear trend ('linear') from a segment."""
    x = np.asarray(x, dtype=float)
    if detrend in (None, "none"):
        return x
    return sps.detrend(x, type=detrend)


def make_window(N: int, kind: str = "hann") -> np.ndarray:
    """Periodic (DFT-even) window, the correct form for spectral analysis.

    'rect' returns ones (i.e. no windowing).
    """
    if kind in ("rect", "none", None):
        return np.ones(N)
    return sps.get_window(kind, N, fftbins=True)


# --------------------------------------------------------------------------
# Frequency axis + one-sided amplitude spectrum
# --------------------------------------------------------------------------
def freq_axis(nfft: int, fs: float) -> np.ndarray:
    """f_k = k * fs / nfft for k = 0 .. nfft//2  (non-negative half only)."""
    return np.fft.rfftfreq(nfft, d=1.0 / fs)


def amplitude_spectrum(x, fs, window="hann", nfft=None, detrend="constant"):
    """One-sided amplitude spectrum, calibrated so a sine of amplitude A
    shows a peak of height A (to within window scalloping loss).

    A_k = 2 * |X_k| / sum(w)      for 0 < k < nfft/2
    A_0 = |X_0| / sum(w)          (DC bin  - not doubled)
    A_Ny = |X_Ny| / sum(w)        (Nyquist - not doubled, even nfft)

    sum(w) is the window's "coherent gain" * N. For a rectangular window
    sum(w) = N, which reduces to the textbook "divide by N". Dividing by
    sum(w) instead of N makes the amplitude correct for tapered windows too.
    Zero padding (nfft > N) does NOT change the normalisation: sum(w) is
    still taken over the N real samples.

    Returns (freqs, amplitude).
    """
    x = preprocess(x, detrend)
    N = len(x)
    nfft = N if nfft is None else int(nfft)
    w = make_window(N, window)
    X = np.fft.rfft(x * w, n=nfft)          # zero-pads automatically if nfft > N
    A = np.abs(X) / np.sum(w)
    A[1:] *= 2.0
    if nfft % 2 == 0:                        # Nyquist bin is its own mirror
        A[-1] /= 2.0
    return freq_axis(nfft, fs), A


def to_db(amp, ref=1.0, floor=1e-12):
    """20*log10(amp/ref). Amplitude (not power) -> 20, power -> 10."""
    return 20.0 * np.log10(np.maximum(np.asarray(amp), floor) / ref)


# --------------------------------------------------------------------------
# Power spectral density
# --------------------------------------------------------------------------
def periodogram_psd(x, fs, window="hann", detrend="constant"):
    """One-sided windowed periodogram, units g^2/Hz.  PSD_k = |X_k|^2/(fs*sum(w^2)),
    doubled for 0<k<N/2.  Single-segment estimate: high variance (~100 % of the
    mean per bin, regardless of N)."""
    x = preprocess(x, detrend)
    N = len(x)
    w = make_window(N, window)
    X = np.fft.rfft(x * w)
    P = np.abs(X) ** 2 / (fs * np.sum(w ** 2))
    P[1:] *= 2.0
    if N % 2 == 0:
        P[-1] /= 2.0
    return freq_axis(N, fs), P


def welch_psd(x, fs, nperseg=256, overlap=0.5, window="hann", detrend="constant"):
    """Welch PSD: average periodograms of overlapping sub-segments.
    Variance falls roughly as 1/K (K = number of averaged sub-segments, a bit
    worse with overlap) at the price of coarser resolution fs/nperseg."""
    nperseg = min(nperseg, len(x))
    f, P = sps.welch(x, fs=fs, window=window, nperseg=nperseg,
                     noverlap=int(nperseg * overlap), detrend=detrend,
                     scaling="density")
    return f, P


# --------------------------------------------------------------------------
# Parseval check
# --------------------------------------------------------------------------
def parseval_check(x, window="hann", detrend="constant"):
    """Return (time_energy, freq_energy) for the *windowed, preprocessed*
    segment.  Parseval:  sum_n |y[n]|^2 = (1/N) sum_k |Y[k]|^2  (full spectrum).
    We evaluate the right side from the one-sided rfft by re-weighting the
    mirrored bins, so the two numbers must agree to ~1e-12 relative."""
    x = preprocess(x, detrend)
    N = len(x)
    y = x * make_window(N, window)
    Y = np.fft.rfft(y)
    p = np.abs(Y) ** 2
    weights = np.full_like(p, 2.0)
    weights[0] = 1.0
    if N % 2 == 0:
        weights[-1] = 1.0
    return float(np.sum(y ** 2)), float(np.sum(weights * p) / N)


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
def segment_features(x, fs, window="hann", nfft=None, detrend="constant",
                     fmax=None):
    """The four RailGuard features for one segment.

    rms           : time-domain RMS after mean removal (g)
    dom_freq      : frequency of the largest amplitude bin, k=0 EXCLUDED (Hz)
    spec_energy   : energy in (0, fmax] from the spectrum via Parseval, in g^2*s.
                    With fmax=None it equals sum(x^2)/fs of the mean-removed
                    segment minus its DC bin (checked in tests)
    spec_centroid : sum(f*P)/sum(P) over k>=1 (Hz)
    """
    x = preprocess(x, detrend)
    N = len(x)
    nfft = N if nfft is None else int(nfft)
    w = make_window(N, window)
    X = np.fft.rfft(x * w, n=nfft)
    f = freq_axis(nfft, fs)
    A = np.abs(X) / np.sum(w)
    A[1:] *= 2.0
    if nfft % 2 == 0:
        A[-1] /= 2.0

    P = np.abs(X) ** 2
    weights = np.full_like(P, 2.0)
    weights[0] = 1.0
    if nfft % 2 == 0:
        weights[-1] = 1.0
    # window-power-corrected so that energy ~ sum(x^2) of the un-windowed segment
    Pn = weights * P / (nfft * np.mean(w ** 2))

    sel = slice(1, None) if fmax is None else slice(1, int(np.searchsorted(f, fmax, side="right")))
    f_s, A_s, P_s = f[sel], A[sel], Pn[sel]

    return {
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "dom_freq": float(f_s[np.argmax(A_s)]),
        "spec_energy": float(np.sum(P_s) / fs),
        "spec_centroid": float(np.sum(f_s * P_s) / np.sum(P_s)),
    }
