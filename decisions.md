# RailGuard – Decision Log

Chronological. Each entry: **Decision / Alternatives / Why / Evidence**.
Status tags: ✅ made · 🟡 proposed, awaiting your approval · ⚠️ problem you need to know about.

---

## D01 ✅ Which files of Rail-VIVID to download
- **Decision:** download only the 20 run CSVs (~1.9 GB), the README and `Acquisition.py`. Skip images and FARO point clouds.
- **Alternatives:** full dataset (114 GB).
- **Why:** RailGuard is vibration-only; images/point clouds are ~98 % of the bytes and unused.
- **Evidence:** HF file API listing: 20 folders `AtoB_{20,22.5}_{1-4}`, `BtoA_{22.5,25,27.5}_{1-4}`, one CSV each, 73–121 MB.
- First download: `BtoA_27.5_2.csv` (73.6 MB) for schema inspection. It is in the **train** split (see D06), so no test/validation data has been looked at.
- Note: the DOI 10.57967/hf/8411 resolves to `huggingface.co/datasets/saluslab/Rail-VIVID`.

## D02 ⚠️ Actual schema differs from the documentation
| Documentation says | Actual file |
|---|---|
| Columns `Channel_1 … Channel_6` | `Channel1 … Channel6` (no underscore) |
| Timestamp `MM/DD/YYYY HH:MM:SS AM/PM` | ISO `2025-04-11 15:49:34.403500` |
| 12 variables | **11** columns (Timestamp, 6 channels, Temperature, Humidity, Latitude, Longitude) |
| Units "m/s²" in variable list, "g" in methods | Values look like **g**: Ch3 reads ≈ +0.93 at rest start (≈ 1 g gravity). Std 0.03–0.64. Treated as g. |
| "no missing data" | Confirmed on the one run inspected: 0 NaN in 388 020 rows, every Δt exactly 0.5 ms (=2 kHz), no gaps. Loader will still check every run. |
| 9 documented anomalies | Repo's `Anomaly/Anomaly3DScanning/` contains only **Anomaly1, 2, 4, 5** (4 folders). |
- Run `BtoA_27.5_2`: 388 020 samples = 194.0 s, path length 1493 m by GPS haversine, mean speed 7.7 m/s = 27.7 km/h (nominal 27.5 → GPS and timestamps are consistent).
- Temperature/Humidity are constants (41 °F, 0.7) within a run → not used as features.
- GPS has ~one unique value per sample (interpolated), so it is smooth and usable as a position axis.

## D03 ⚠️ Ground-truth anomaly locations are NOT in the data files (biggest risk to the project)
- The CSVs have **no anomaly label column**. The README says anomalies are documented only as FARO 3D scans (`.fls`, proprietary binary) + photos, with no table of GPS coordinates or distance-along-track.
- Only 4 of 9 anomaly folders are in the repo. The Scientific Data paper (10.1038/s41597-026-07555-y) may contain a location table, but Nature returned an auth redirect and a web search found nothing, so I could not read it.
- **Consequence:** requirement 6 (precision/recall/F1/FPR vs ground truth) cannot be done yet. Options are listed in the hand-off message; need your decision.
- **Small-sample warning (independent of the above):** even with all 9 locations, the test split has 3 runs → at most 27 anomaly passes. Precision/recall will have very wide confidence intervals (e.g. 20/27 recall → 95 % Wilson CI ≈ 55–87 %). Numbers will be reported with CIs and labelled "indicative, not conclusive".

## D04 ⚠️ The data is already low-pass filtered at 500 Hz (affects the "500 Hz band" requirement)
- Rail-VIVID applies a 500 Hz Butterworth low-pass *after* 2 kHz sampling. The Nyquist limit is 1 kHz, but 500–1000 Hz holds no information.
- **Evidence (Welch PSD, ch3, run BtoA_27.5_2, 2048-pt Hann):** 450–500 Hz −57.3 dB, 500–550 Hz −63.6, 600–700 −85.0, 800–1000 **−146.1 dB** re g²/Hz. Share of total power above 500 Hz: ≤0.017 % on all channels.
- **Decision:** all features are computed on 0–500 Hz (`fmax = 500`). We still FFT/plot to 1 kHz (Nyquist) for the plots, as the DSP spec requires, and we use the empty upper band to show that the filter is present.
- Why: including 500–1000 Hz would add only filter-stopband noise to spectral centroid, diluting it.

## D05 ✅ Channel selection (provisional, from run BtoA_27.5_2 only)
Welch PSD power split per channel:
| Ch | std (g) | 0–50 Hz | 50–200 Hz | 200–500 Hz | peak (>1 Hz) |
|---|---|---|---|---|---|
| 1 | 0.40 | 92 % | 6.6 % | 1.4 % | 16.6 Hz |
| 2 | 0.35 | 0.6 % | 67.9 % | 31.5 % | 58.6 Hz |
| 3 | 0.40 | 91 % | 7.3 % | 1.6 % | 16.6 Hz |
| 4 | 0.64 | 29.6 % | 42.1 % | 28.3 % | 50.8 Hz |
| 5 | 0.036 | 100 % | 0 | 0 | 2.0 Hz |
| 6 | 0.031 | 100 % | 0 | 0 | 2.0 Hz |
- Ch5/6 carry ~10× less energy, entirely < 50 Hz (likely a different mounting/sensor); poor candidates for broadband track-defect features. Ch1/3 are low-frequency; Ch2/4 are broadband.
- **I do not know the physical mounting (axle box vs bogie vs car body) – the docs don't say.** Provisional plan: compute features for all six channels and let you pick; the baseline is built per channel. Needs your input / paper.

## D06 ✅ Train/test/validation split (by RUN, seed 42) — adopted (you delegated the decision)
- 20 runs × 70/15/15 → **14 train / 3 test / 3 validation**.
- **Why by run:** consecutive segments within a run overlap 50 % and share the same wheel/rail state, speed and weather; random segment splits would leak near-duplicates into test.
- **Why stratified:** speed changes the spectrum (frequency content scales with speed/track-wavelength), so I grouped runs by (direction, speed) – 5 groups of 4 – and required every group to keep ≥ 2 runs in train, with test/validation each drawn from different groups. Selection used only run *names* (no signal data), via `numpy.random.default_rng(42)`.
- Proposed:
  - **train (14):** AtoB_20_1, AtoB_20_2, AtoB_22.5_1, AtoB_22.5_2, AtoB_22.5_3, AtoB_22.5_4, BtoA_22.5_1, BtoA_22.5_2, BtoA_22.5_3, BtoA_25_1, BtoA_25_2, BtoA_27.5_1, BtoA_27.5_2, BtoA_27.5_4
  - **test (3):** BtoA_27.5_3, AtoB_20_4, BtoA_25_3
  - **validation (3, yours – I will not load these):** AtoB_20_3, BtoA_25_4, BtoA_22.5_4
- **Flaw found later (see D13):** I stratified on nominal speed, but the dominant nuisance factor turned out to be the *collection day*. Only 2 train runs (AtoB_22.5_4, BtoA_27.5_4) and 2 of 3 test runs (BtoA_27.5_3, AtoB_20_4) are from 2025-04-18. I did NOT re-split afterwards: by then I had seen test-score distributions (re-splitting would be test-informed) and I will not read timestamps of your validation files. The day composition of the validation split is unknown to me; check it yourself (first CSV row).
- Limitation: unseen (direction, speed) combos cannot be tested with 3 runs; AtoB/BtoA cover different travel directions over the same rail so some direction shift is expected.

## D07 ✅ DSP core design (`railguard/dsp.py`)
| Topic | Choice | Alternatives | Why |
|---|---|---|---|
| DC removal | always subtract mean | none | A 0.06 g mean (gravity projection, offset) puts a huge k=0 line whose Hann skirt contaminates bins 1–2. Test: ±2 g DC → DC bin < 0.01 g after removal. |
| Detrend | **mean only by default**; `linear` available | linear detrend | Linear detrend removes slow drift/tilt change inside a segment. At 0.5 s segments, drift is negligible relative to vibration; the windowed spectrum already suppresses it. Test shows linear detrend reduces bins 1–5 more (so it's a switch, not a default); will re-evaluate on real data. |
| Window | **Hann, periodic** (`fftbins=True`) | rect, Hamming | Hann sidelobes fall 18 dB/oct and the first at −31 dB; Hamming has lower first sidelobe (−43 dB) but slow decay. We have strong low-frequency content (92 % of power < 50 Hz on ch1/3) next to weak 200–500 Hz content, so far-sidelobe decay matters more → Hann. Periodic form is the correct one for DFT analysis. |
| One-sided | `rfft`/`rfftfreq` | full complex `fft` | Real input → negative half is the conjugate mirror; zero information lost. |
| Frequency axis | `f_k = k·fs/nfft`, k=0…nfft/2, last = Nyquist | — | tested |
| Amplitude norm | `A_k = 2|X_k| / Σw`, DC and Nyquist not doubled | divide by N | Σw=N for rectangular, so identical to "divide by N"; for Hann Σw = N/2 and dividing by N would under-read amplitude by 6 dB. ×2 for the folded negative half. |
| dB | `20·log10(A)` | 10·log10 | Amplitude not power → 20. dB is used because vibration spectra span >5 decades (this data: −43 dB to −146 dB/Hz) and perception/ratios are multiplicative. |
| Zero padding | **none** in the feature pipeline; optional `nfft=4N` for plots only | pad to 2^k or ×4 | Padding interpolates the same spectrum (sinc-interpolation) – finer grid, same main-lobe width, no new resolution. Our N is already a power of two. Measured: ×4 pad moves the 50 Hz peak from 50.78 Hz / 0.896 g to 49.80 Hz / 0.989 g (scalloping recovered, peak located to 0.2 Hz) but the main-lobe width is unchanged. |
| PSD | periodogram (single segment) **and** Welch | — | Periodogram's per-bin coefficient of variation is ≈100 % no matter how long the segment (test: 1.00). Welch averaging cut it to 0.17 in the test (16 segments, 50 % overlap) at cost of resolution. Use periodogram/amplitude for peak finding in a segment; Welch for a stable baseline spectrum and for the whole-run view. |
| Parseval | `Σy² = (1/N)Σ|Y|²` (weights ×2 for mirrored bins) | — | Passes to rtol 1e-10 for rect/hann/hamming; used as FFT self-check. |

## D08 ✅ Synthetic-signal validation (required before real data)
`tests/test_dsp_synthetic.py` – 10 tests, **all pass** (`pytest -s tests`).
- Signal: 50 Hz (1.0 g) + 137 Hz (0.5 g) + 310 Hz (0.25 g), random phases, +2 g DC, + linear ramp (0.3 g/s), + white noise σ=0.05; fs=2000, N=1024 (Δf=1.953 Hz).
- Detected peaks (Hann): **50.78 Hz / 0.896 g, 136.72 Hz / 0.490 g, 310.55 Hz / 0.239 g** vs truth 50/1.0, 137/0.5, 310/0.25. Frequencies are within 1 bin. The 50 Hz amplitude is 10 % low – *expected*: 50 Hz lies at bin 25.6, i.e. 0.4 bin from the nearest bin centre, and the Hann scalloping loss is up to 1.42 dB (15 %).
- Tone placed exactly on bin 100 with rect window → amplitude 0.8 reads 0.8 to 1e-9.
- Leakage (noiseless, bins >400 Hz where there are no tones): rect **−52.4 dB**, Hann **−124.7 dB**.
  - *Test-design bug found and fixed:* my first version measured the median far floor with noise σ=0.05 present; the noise floor (≈−50 dB) is above the leakage skirt, so Hann looked no better than rect (−50.3 vs −50.6). This is a real lesson for the data: **leakage matters only where signal dynamic range exceeds the noise floor**, as it does here (ch1/3 low-frequency power is ~40 dB above 200–500 Hz).
- Parseval (time vs frequency energy) equal to 1e-10.
- Features on the segment: RMS 0.8126, dominant 50.78 Hz (DC excluded), centroid 80.7 Hz (theory from tone powers 78.95 Hz, 2.2 % off – noise and leakage), spectral energy 0.3345 vs time-domain 0.3381 g²·s (1 % difference; windowing + noise).
- Periodogram CV 1.00 vs Welch CV 0.17; Welch level on white noise matches 2σ²/fs within 10 %.

## D09 ✅ Segmentation: N=1024, hop=512 (50 %), Hann — adopted
- Candidates (fs = 2000 Hz; speed 5.6–7.7 m/s ≈ 20–27.7 km/h):

| N | duration | Δf = fs/N | track covered | 
|---|---|---|---|
| 512 | 0.256 s | 3.91 Hz | 1.4–2.0 m |
| **1024** | **0.512 s** | **1.95 Hz** | **2.9–3.9 m** |
| 2048 | 1.024 s | 0.98 Hz | 5.7–7.9 m |
- **Proposal: N = 1024, 50 % overlap (hop 512 = 0.256 s ≈ 1.4–2.0 m), Hann.**
- Trade-off: Δf (frequency resolution) = 1/T, so halving the segment doubles Δf but halves time (position) localisation. A track defect is a short event (< ~1 m long at 7 m/s ≈ 0.15 s), so a 0.5 s window already averages it with ~3 m of good track; 2048 would dilute it further. 1024 gives Δf = 1.95 Hz; a Hann main lobe is 4Δf wide null-to-null (7.8 Hz at N=1024), so two lines ≥ ~8 Hz apart are just resolved. The per-channel peaks I saw (Ch4 50.8 Hz, Ch2 58.6 Hz – different channels, so not a proof of need) are ~8 Hz apart; at N=512 the lobe is 15.6 Hz wide and such lines would merge. 2048 would be better in frequency but averages a short event with 6–8 m of normal track.
- Hann at 50 % overlap satisfies COLA (constant overlap-add), so no data is down-weighted overall.
- STFT for the spectrogram: the same N=1024 / hop 512 / Hann for consistency, 75 % overlap offered as a UI option (smoother, but adjacent columns ≈ 75 % redundant – not independent measurements).

---

## D10 ✅ Channel use (replaces provisional D05)
- Features are computed for all six channels (table + app), but the **anomaly score uses Ch1–4**. Ch5/6 are ~10× quieter, entirely < 50 Hz, and I couldn't find their mounting in the docs, so they carry no broadband defect information I can justify.
- Ch2 is the only channel that never drops out (D12), so it is the default plot channel in the app.

## D11 ✅ Cruise-speed filter
- **Problem:** run `BtoA_27.5_2` – GPS speed (1 s smoothing) is ≈4 m/s in the first 15 s and ≈4.9 m/s in the last window; after stopping, RMS is ≈0 on several channels. These phases (accel/brake/standstill) would pollute the baseline.
- **Decision:** a segment is *cruise* if GPS speed ≥ 90 % of the run's median speed. Only cruise segments are used to fit the baseline and are scored; others are shown as "N/A".
- **Alternatives:** fixed speed threshold (fails because actual speeds differ from nominal); trimming a fixed number of seconds (runs differ in length 173–297 s).
- **Evidence:** actual cruise speeds differ from nominal: "27.5 km/h" runs measured 8.2–9.9 m/s (29–36 km/h); "20 km/h" runs 5.4–6.2 m/s (19–22 km/h). So nominal-speed groups are *loose labels*.
- Does speed matter for the spectrum? Segment-level R² of log-feature vs log-speed ≤ 0.06 in almost every direction/channel/feature (exception: BtoA Ch2 RMS, R² = 0.38); slope signs are inconsistent across channels. So I did **not** build a speed-conditional model.

## D12 ⚠️ Sensor dropouts (Ch1, Ch3, Ch4) – a data-quality finding, handled in the baseline
- **Evidence:** during cruise, segment RMS on Ch1/3/4 falls to ≈3×10⁻⁴ g (the accelerometer noise floor) simultaneously, while Ch2 keeps vibrating at 0.35–0.6 g. The pooled RMS histogram is clearly bimodal (dead floor ≈10^-3.5 g vs healthy ≥ 0.1 g). Fraction of cruise segments dead on Ch1/3/4 per run: 3 %–69 % (BtoA_27.5_4 ≈67–69 %, AtoB_22.5_4 ≈47 %, BtoA_25_3 ≈47 %, BtoA_25_2 41–58 %). Ch1/3/4 die together (98–99.6 % agreement in the run I checked); Ch2 essentially never. Cause unknown (a DAQ-module or cabling issue is my guess – not verifiable from the docs).
- **First symptom:** before handling this, my "extreme" scores (13–27) in test run BtoA_27.5_3 were all dropout segments, not track events – they would have been reported as the worst anomalies.
- **Decision:** a channel is *unhealthy* in a segment if its RMS < 0.005 g **or** the same-run neighbouring segment's is (50 % overlap means the neighbour shares half the samples, so a dropout edge lands in it). The baseline is fitted on segments where all four scoring channels are healthy (8 753 train segments). For a segment with missing channels, the score uses the **Gaussian marginal** (sub-vector of μ, sub-block of Σ) of the healthy channels, normalised by their feature count d. With no healthy channel the score is N/A.
- **Alternatives:** drop the whole segment (loses most of some runs and the still-valid Ch2 information); impute (invents data); treat dead as "zero vibration" (wrongly flagged as hugely anomalous).
- Threshold 0.005 g: ~15× above the dead floor, ~70× below the healthy median; the valley between the two populations is sparsely populated, so results are insensitive to the exact value.

## D13 ⚠️ Baseline grouping: by collection SESSION (date), not by nominal speed
- **Evidence (Ch2, per run, cruise segments):** 2025-04-11 runs: RMS 0.341–0.377 g; 2025-04-18 runs: RMS 0.589–0.636 g; dominant frequency 58.6/117.2 Hz (Apr 11) vs 175.8 Hz (Apr 18). A ~1.7× step, bigger than any speed or direction effect. Cause unknown (vehicle/sensor state between days); the README is silent.
- With the speed-group baseline, whole Apr-18 runs had median scores 2.7–8.5 (everything flagged). With a per-session baseline their medians fell to 0.6–1.1.
- **Decision:** Gaussian baseline per `session` (date of first timestamp; also available for uploads), Ledoit-Wolf-shrunk covariance of 16 log-features (4 channels × 4 features). Unknown session → global baseline plus a UI warning.
- **Why log features:** RMS, energy and centroid are positive and right-skewed; log makes them closer to Gaussian and turns multiplicative gain differences into additive shifts.
- **Why Mahalanobis (not independent z-scores):** features are strongly correlated (RMS and spectral energy are nearly the same quantity; channels share vibration). Independent z's would double count. Ledoit-Wolf shrinkage keeps Σ invertible despite the near-collinear pair. Score = √(D²/d) ≈ 1 for typical segments regardless of d.
- **Limitation:** the Apr-18 baseline rests on only 2 train runs (one with ~69 % dropout). Day-specific baselines need reference runs from that day – an operational requirement of this method.

## D14 ⚠️ No ground-truth anomaly positions → what I tried, what I report instead
- Tried **pseudo-labels by cross-run GPS consensus**: for each train run, flag its top 5 % of segments by within-run score (each run scored against its own typical behaviour), bin at 10 m along track, and test with a binomial (base rate 0.133) whether any bin is flagged in more runs than chance, Bonferroni-corrected over 142 bins.
- **Result: nothing significant** (best: s≈140–160 m, 6/14 runs, raw p = 0.006, Bonferroni q = 0.90; s≈1390 m 5/13; others weaker). With the cross-session baseline score, no bin exceeded 50 % of runs (chance level also 0).
- **Consequence:** I do **not** report precision/recall against pseudo-labels – they would be meaningless. Either the 4 anomalies in the repo aren't detectable at 10 m / 0.5 s resolution with these four features, or the GPS projection (accuracy ~metres) smears them; I can't tell which without true locations.
- Instead: the semi-synthetic evaluation (D15). If you obtain the real locations (authors' table), they can be dropped into the evaluation – see system_architecture.md §9 (not wired yet; deliberately left small).
- The repo has only Anomaly1/2/4/5 folders; all 9 would need the paper's supplementary table.

## D15 ✅ Semi-synthetic evaluation protocol (test split only)
- Inject a damped 200 Hz ring-down (τ = 20 ms, 100 ms long) – a crude rail-joint/dip impact – into Ch1-4 of each TEST run at 40 random positions (≥3 s apart, ≥15 s from run ends) at peak amplitudes A ∈ {0.4, 0.8, 1.6, 3.2} g. Ambient RMS is 0.35–0.7 g and the largest value seen ≈ 1.8 g, so 3.2 g is a severe event. Positive = the one segment whose central half contains the event centre; segments only partially overlapping are ignored; all others are negatives. 115 positives per amplitude (3 runs). Seeds 1000 + test-run index.
- Caveat: these are events of my choosing, not real defects. They show the **sensitivity limit of the pipeline**, not real-world accuracy.

## D16 ✅ Thresholds and results
- **Rule:** REVIEW = median over TRAIN runs of each run's own leave-one-run-out (LORO) 97.5th percentile of score; HIGH = same with 99.5th → **REVIEW ≥ 2.69, HIGH ≥ 5.00**.
- **Why this rule:** pooling all LORO train scores gives REVIEW 5.34, inflated by a few outlier *whole runs* (e.g. AtoB_22.5_3 median score 3.2). With 5.34, test recall was 0 even at 1.6 g (first evaluation run) although the scores themselves separated well (AUC 0.91 at 1.6 g, 0.60 at 0.4 g, 0.999 at 6.4 g on BtoA_25_3) – a threshold problem, not a scoring bug. The median-of-quantiles rule is robust to those runs and uses no test information.
- **LORO** (not in-sample) train scores are used so the train distribution isn't optimistic.
- **Results on TEST (3 runs, 115 positives per amplitude, REVIEW threshold):**

| peak A | precision | recall (95 % Wilson CI) | F1 | FPR |
|---|---|---|---|---|
| 0.4 g | 0.30 | 0.09 (0.05–0.15) | 0.14 | 1.2 % |
| 0.8 g | 0.38 | 0.12 (0.07–0.19) | 0.18 | 1.2 % |
| 1.6 g | 0.52 | 0.22 (0.15–0.30) | 0.31 | 1.2 % |
| 3.2 g | 0.71 | 0.49 (0.40–0.58) | 0.58 | 1.2 % |

  HIGH threshold (5.0): recall 0 at all amplitudes, FPR 0.4 %.
- Clean (un-injected) test runs, share of cruise segments flagged: REVIEW 4.6 % (BtoA_27.5_3), 0.0 % (AtoB_20_4), 0.4 % (BtoA_25_3); HIGH 1.5 %, 0, 0. These are *flag rates* on unlabelled data, not false-positive rates.
- **Tuning on test (as you specified):** sweeping REVIEW on test injections, the best mean-F1 value is 1.5 (at 3.2 g: recall 0.83, precision 0.57, F1 0.68, but FPR 3.7 %). I did **not** adopt it by default because it is tuned on my own invented event shape (optimistic); I kept the train-derived thresholds and saved the sweep (`outputs/threshold_sweep.csv`). Your validation split is the right place for an unbiased check.
- **Why sensitivity is modest:** features are averaged over 0.512 s; a 0.1 s transient adds only ~10 % to segment RMS even at 1.6 g. Transient detectors usually add crest factor / kurtosis / peak-to-RMS – outside the four required features, and the most valuable follow-up.
- **Small-sample caveat:** only 3 test runs (2 from the same day). The *trend* with amplitude is reliable (same data); absolute numbers for other days/runs are not.

## D17 ✅ Misc. engineering decisions
- **Validation guard:** `data.load_run` raises `ValidationAccessError` for the 3 validation runs (unless you set `RAILGUARD_ALLOW_VALIDATION=1`); the app doesn't list them; a unit test checks this. Their CSVs *are* on disk (downloaded with the rest, size-verified) but were never opened.
- **Missing/corrupt data:** non-finite samples → linear interpolation if the gap ≤ 10 samples, always marked invalid; timestamp gaps (dt off by > 10 %) marked invalid; any segment containing an invalid sample is dropped. On the 17 train/test runs: 0 non-finite samples and 0 timestamp gaps (so this path is covered only by a unit test with synthetic NaNs). A first-sample off-by-one in my gap detector (flagged sample 0 as a gap) was found and fixed.
- **Column names:** loader strips `_`, so both `Channel1` (actual) and `Channel_1` (documented) work.
- **Along-track position:** reference polyline from the longest train run's GPS (BtoA_27.5_4, 1509 m); nearest-vertex projection; segments > 15 m off-line excluded from the consensus analysis.
- **Streamlit** (as required), tested headlessly with `streamlit.testing.AppTest` (run switch, channel, dB/linear, padding, detrend, segment index – no exceptions).
- **Windowed vs unwindowed on real data** (`outputs/figures/real_ch2_window_vs_rect_dB.png`): Ch2, BtoA_27.5_2 mid-run segment, median level 300–480 Hz: rect −45.4 dB vs Hann −53.9 dB (8.5 dB lower); in the 500 Hz stop-band the rect floor sits near −100 dB while Hann reaches ≈ −180 dB. Ch1 shows no difference (−51.6 vs −51.1 dB) because it is noise-floor-limited there – same lesson as the synthetic test: leakage matters only where signal dynamic range exceeds the noise floor.
- **Welch overlay in the app plot** is rescaled approximately for visual shape comparison only (labelled "shape only"), not calibrated like the single-segment spectrum.
- **Secondary dataset (Ponneri Bridge):** not used – optional, no labels, and the main dataset already needed substantial QC work.
- **Environment:** Python 3.14.3; numpy 2.4.3, scipy 1.17.1, pandas 2.3.3, scikit-learn 1.8.0, matplotlib 3.10.8, streamlit 1.66.0; 16 tests pass.

## D18 ✅ Spectral-energy definition (Parseval)
- `spec_energy = Σ_k c_k |X_k|² / (nfft · mean(w²)) / fs` over bins 1…500 Hz (c_k = 2 for mirrored bins, 1 for DC/Nyquist), in g²·s. Dividing by `mean(w²)` undoes the Hann window's power loss. It equals `Σx²/fs` of the mean-removed segment to ~1 % (synthetic: 0.3345 vs 0.3381). Exact Parseval equality (rtol 1e-10) holds for the *windowed* signal and is a unit test; the window-corrected feature is approximate by design.

## D19 ✅ Live (simulated real-time) demo mode
- **Request:** upload the held-out 15 % as CSV and see the model compute "in real time" for a demo.
- **Decision:** a "Live demo" tab replays an uploaded (or selected) run as a stream. `railguard/live.py` consumes one 512-sample hop at a time and computes the windowed FFT → 4 features × 6 channels → score from the raw samples *on arrival*; the UI shows rolling 10 s waveform, current-segment dB spectrum, score-so-far with thresholds, and a coloured class badge. Controls: start offset, play length, 1–25× speed, channel.
- **Why simulated, not true streaming:** the data is a recorded CSV; a replay with the same per-segment code path is what a live deployment would run. Per-segment compute is ~7 ms vs the 256 ms hop budget (≈36× faster than real time), so real-time feasibility is demonstrated, not assumed.
- **Differences from the batch pipeline (causality):** (1) cruise flag uses the *running* median speed (11 of 741 segments differ in `BtoA_27.5_3`, all early in the run); (2) a segment is scored when its next neighbour arrives, because channel health uses neighbours (dropout-edge rule) → 0.256 s latency; (3) GPS speed is the precomputed smoothed track speed.
- **Verification:** on test run BtoA_27.5_3 the live scores equal the batch scores exactly (max |Δ| = 0.0 over the 582 segments scored by both). Headless AppTest of the live loop passes.
- **Upload path:** tested on a 74 MB train CSV (parse 0.8 s, features 0.6 s). Upload limit raised to 500 MB in `.streamlit/config.toml` (largest run is 121 MB). A session date that has no baseline falls back to the global one with a warning.
- **I have not run any of the 3 validation files through the app** (they are yours); a validation run uploaded by you is processed with the same code as any other file.
