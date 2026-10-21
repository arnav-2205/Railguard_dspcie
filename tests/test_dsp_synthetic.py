"""Unit tests: validate the FFT chain on a synthetic signal with KNOWN content
before it is trusted on real data.  Run:  pytest -s tests/"""
import numpy as np
import pytest

from railguard import dsp

FS = 2000.0
N = 1024                      # candidate segment length (see decisions.md)
DF = FS / N
# (frequency Hz, amplitude g).  50 Hz is deliberately NOT on a bin centre
# (50/1.953125 = 25.6 bins) so leakage/scalloping is exercised.
TONES = [(50.0, 1.00), (137.0, 0.50), (310.0, 0.25)]
DC, SLOPE, NOISE_SD = 2.0, 0.3, 0.05


def synth(seed=0, n=N, noise_sd=NOISE_SD):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    x = sum(a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f, a in TONES)
    return x + DC + SLOPE * t + rng.normal(0, noise_sd, n)


def top_peaks(f, A, k=3):
    from scipy.signal import find_peaks
    idx, _ = find_peaks(A)
    idx = idx[np.argsort(A[idx])[::-1][:k]]
    return sorted(zip(f[idx], A[idx]))


def test_freq_axis_and_nyquist():
    f = dsp.freq_axis(N, FS)
    assert f[0] == 0 and f[-1] == FS / 2 and len(f) == N // 2 + 1
    assert np.isclose(f[1] - f[0], DF)


def test_peaks_match_true_frequencies_and_amplitudes():
    f, A = dsp.amplitude_spectrum(synth(), FS, "hann")
    peaks = top_peaks(f, A)
    print("\nDetected (Hz, g):", [(round(a, 2), round(b, 3)) for a, b in peaks])
    for (fd, ad), (ft, at) in zip(peaks, sorted(TONES)):
        assert abs(fd - ft) <= DF, f"freq {fd} vs {ft}"          # within one bin
        # Hann worst-case scalloping loss is 1.42 dB (=15 %); allow 17 %
        assert abs(ad - at) / at < 0.17, f"amp {ad} vs {at}"


def test_hann_reduces_leakage_vs_rect():
    # Noiseless on purpose: with noise sigma=0.05 the noise floor (~-50 dB) sits
    # ABOVE the leakage skirts and hides the effect (found when first run).
    x = synth(noise_sd=0.0)
    f, Ar = dsp.amplitude_spectrum(x, FS, "rect")
    _, Ah = dsp.amplitude_spectrum(x, FS, "hann")
    far = (f > 400)                 # no tones above 310 Hz: pure leakage
    rect_peak = 20 * np.log10(Ar[far].max())
    hann_peak = 20 * np.log10(Ah[far].max())
    print(f"\nWorst leakage >400 Hz (noiseless): rect {rect_peak:.1f} dB, hann {hann_peak:.1f} dB")
    assert hann_peak < rect_peak - 20


def test_exact_bin_sine_amplitude_is_exact_rect():
    # tone exactly on bin 100 -> no leakage, rectangular window -> amplitude exact
    t = np.arange(N) / FS
    x = 0.8 * np.sin(2 * np.pi * (100 * DF) * t)
    f, A = dsp.amplitude_spectrum(x, FS, "rect")
    assert np.isclose(A[100], 0.8, rtol=1e-9) and np.isclose(f[100], 100 * DF)


def test_mean_removal_kills_dc_and_detrend_kills_ramp():
    x = synth()
    _, A_none = dsp.amplitude_spectrum(x, FS, "hann", detrend=None)
    _, A_dc = dsp.amplitude_spectrum(x, FS, "hann", detrend="constant")
    _, A_lin = dsp.amplitude_spectrum(x, FS, "hann", detrend="linear")
    assert A_none[0] > 1.5 and A_dc[0] < 1e-2                  # DC bin ~2 g removed
    # ramp leaks into low bins; linear detrend should lower low-frequency content
    assert A_lin[1:6].sum() < A_dc[1:6].sum()


def test_zero_padding_interpolates_but_does_not_add_resolution():
    x = synth()
    f1, A1 = dsp.amplitude_spectrum(x, FS, "hann")
    f4, A4 = dsp.amplitude_spectrum(x, FS, "hann", nfft=4 * N)
    assert np.isclose(f4[1], f1[1] / 4)                        # finer grid ...
    assert A4.max() >= A1.max() * 0.999                         # ... better peak pick
    # main-lobe width is set by window length, unchanged: peak height can only
    # recover scalloping (<= 1.42 dB), never exceed the true amplitude by much
    assert A4.max() < 1.0 * 1.05


def test_parseval_holds_to_machine_precision():
    for w in ("rect", "hann", "hamming"):
        te, fe = dsp.parseval_check(synth(), window=w)
        assert np.isclose(te, fe, rtol=1e-10), (w, te, fe)


def test_features_spectral_energy_matches_time_energy():
    x = dsp.preprocess(synth(), "constant")
    feat = dsp.segment_features(x, FS, "hann")
    target = np.sum(x ** 2) / FS
    print("\nFeatures:", {k: round(v, 4) for k, v in feat.items()}, "time-energy", round(target, 5))
    # windowing makes this approximate (window-power correction), not exact
    assert abs(feat["spec_energy"] - target) / target < 0.10
    assert np.isclose(feat["rms"], np.sqrt(np.mean(x ** 2)))
    assert abs(feat["dom_freq"] - 50.0) <= DF                  # k=0 excluded
    # centroid of 3-tone power: power ~ A^2/2 -> (50*1 + 137*.25 + 310*.0625)/1.3125
    expected = (50 * 1 + 137 * 0.25 + 310 * 0.0625) / (1 + 0.25 + 0.0625)
    assert abs(feat["spec_centroid"] - expected) / expected < 0.10


def test_dominant_frequency_ignores_dc():
    x = synth() + 50.0                                          # huge DC, no mean removal
    feat = dsp.segment_features(x, FS, "hann", detrend=None)
    assert feat["dom_freq"] > 0


def test_welch_is_lower_variance_than_periodogram():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 16 * N)                                # white noise: true PSD flat
    _, Pp = dsp.periodogram_psd(x, FS)
    _, Pw = dsp.welch_psd(x, FS, nperseg=N)
    cv_p = Pp[1:-1].std() / Pp[1:-1].mean()
    cv_w = Pw[1:-1].std() / Pw[1:-1].mean()
    print(f"\nCoeff. of variation, periodogram {cv_p:.2f} vs Welch {cv_w:.2f}")
    assert cv_w < 0.5 * cv_p
    # white noise sigma^2 = 1 -> one-sided PSD = 2*sigma^2/fs
    assert np.isclose(Pw[1:-1].mean(), 2.0 / FS, rtol=0.1)
