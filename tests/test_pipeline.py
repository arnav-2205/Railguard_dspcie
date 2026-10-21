"""Pipeline tests that need no real data (a synthetic Rail-VIVID-style CSV is built on the fly)."""
import io

import numpy as np
import pandas as pd
import pytest

from railguard import baseline as B, config as C, data, evaluate as E, features


def make_csv(n=60 * 2000, dead_from=None, seed=0, extra_nan_at=None):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / C.FS
    ts = pd.Timestamp("2025-04-11 15:00:00") + pd.to_timedelta(t, unit="s")
    d = {"Timestamp": ts}
    for i in range(1, 7):
        d[f"Channel{i}"] = 0.3 * np.sin(2 * np.pi * (40 + 20 * i) * t) + 0.1 * rng.normal(size=n) + 0.05
    if dead_from is not None:
        d["Channel3"][dead_from:] = 3e-4 * rng.normal(size=n - dead_from)
    if extra_nan_at is not None:
        d["Channel2"][extra_nan_at:extra_nan_at + 5] = np.nan
    d["Temperature"], d["Humidity"] = 41.0, 0.7
    d["Latitude"] = 40.23 + 1e-5 * np.arange(n) * 0.5
    d["Longitude"] = -77.88 - 1e-5 * np.arange(n) * 0.5
    buf = io.StringIO(); pd.DataFrame(d).to_csv(buf, index=False); buf.seek(0)
    return buf


def test_validation_runs_are_blocked():
    for r in C.VALIDATION:
        with pytest.raises(data.ValidationAccessError):
            data.load_run(r)


def test_split_is_disjoint_and_sized():
    assert len(C.TRAIN) == 14 and len(C.TEST) == 3 and len(C.VALIDATION) == 3
    assert not set(C.TRAIN) & set(C.TEST) and not set(C.TRAIN + C.TEST) & set(C.VALIDATION)


def test_loader_handles_nan_and_flags_segments():
    df, qc = data.load_csv(make_csv(extra_nan_at=40000))
    assert qc["n_samples"] == 120000 and qc["n_nonfinite"] == 5
    ft = features.extract_run_features(df, "x_y_1", None)
    # repaired samples are marked invalid -> every segment containing them is dropped
    all_starts = features.segment_starts(len(df))
    assert len(ft) < len(all_starts)
    assert not ((ft.start <= 40000) & (ft.start + C.SEG_LEN > 40000)).any()


def test_features_and_dropout_detection():
    df, _ = data.load_csv(make_csv(dead_from=60000))
    ft = features.extract_run_features(df, "x_y_1", None)
    ft["cruise"] = True
    ok = B.channel_health(ft)
    j = C.SCORE_CHANNELS.index(3)
    late = (ft.start > 62000).to_numpy()
    assert not ok[late, j].any() and ok[~(ft.start > 59000).to_numpy(), j].all()


def test_injection_creates_scoreable_positive_segments():
    df, _ = data.load_csv(make_csv())
    df2, evs = E.inject_events(df, 1.0, 5, seed=1)
    assert len(evs) == 5
    ft = features.extract_run_features(df2, "x_y_1", None)
    lab = E.label_segments(ft.start.to_numpy(), evs)
    assert (lab == 1).sum() == 5            # exactly one positive segment per event
    pos = ft[lab == 1]; neg = ft[lab == 0]
    assert pos.c2_rms.median() > neg.c2_rms.median()


def test_baseline_scores_normal_data_near_one_and_event_higher():
    train = pd.concat([features.extract_run_features(data.load_csv(make_csv(seed=s))[0], f"AtoB_20_{s}", None, session="s1")
                       for s in (1, 2, 3)], ignore_index=True)
    train["cruise"] = True
    bl = B.Baseline(min_fit=100).fit(train)
    test_df = data.load_csv(make_csv(seed=9))[0]
    ft = features.extract_run_features(test_df, "AtoB_20_9", None, session="s1"); ft["cruise"] = True
    s0, _ = bl.score(ft)
    assert 0.5 < np.nanmedian(s0) < 2.0
    df2, evs = E.inject_events(test_df, 2.0, 5, seed=2)
    ft2 = features.extract_run_features(df2, "AtoB_20_9", None, session="s1"); ft2["cruise"] = True
    s1, _ = bl.score(ft2)
    lab = E.label_segments(ft2.start.to_numpy(), evs)
    assert np.nanmedian(s1[lab == 1]) > np.nanmedian(s0)
