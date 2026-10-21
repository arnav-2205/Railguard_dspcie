"""Fit baseline on TRAIN, choose thresholds, evaluate on TEST (semi-synthetic injection).
Never loads validation runs. Outputs -> outputs/."""
import json, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from railguard import config as C, data, features, baseline as B, evaluate as E

AMPS = [0.4, 0.8, 1.6, 3.2]             # g, injected peak amplitude (ambient RMS is 0.35-0.7 g, max seen ~1.8 g)
N_EVENTS = 40                            # per test run per amplitude
REVIEW_Q, HIGH_Q = 0.975, 0.995
SWEEP = np.round(np.arange(1.0, 12.01, 0.5), 2)


def main():
    feats = pd.read_csv(C.OUT_DIR / "features.csv.gz")
    tr = feats[feats.split == "train"].reset_index(drop=True)
    tr["score"], tr["nch"] = B.loro_scores(tr)
    clean = tr[tr.cruise & tr.score.notna()]
    # robust to a few outlier runs: median over runs of each run's own quantile (D15)
    review_thr = float(clean.groupby("run").score.quantile(REVIEW_Q).median())
    high_thr = float(clean.groupby("run").score.quantile(HIGH_Q).median())
    pooled = float(clean.score.quantile(REVIEW_Q))
    print(f"thresholds (TRAIN leave-one-run-out, median of per-run quantiles): REVIEW>={review_thr:.2f} (q{REVIEW_Q}), HIGH>={high_thr:.2f} (q{HIGH_Q}); pooled-q{REVIEW_Q} would have been {pooled:.2f}")

    bl = B.Baseline().fit(tr)
    with open(C.OUT_DIR / "model.pkl", "wb") as f:
        pickle.dump({"baseline": bl, "review_thr": review_thr, "high_thr": high_thr}, f)

    results = {"thresholds": {"review": review_thr, "high": high_thr, "review_q": REVIEW_Q, "high_q": HIGH_Q},
               "split": {"seed": C.SPLIT_SEED, "train": C.TRAIN, "test": C.TEST},
               "n_events_per_run": N_EVENTS, "amps": {}, "clean_flag_rate": {}}
    raw = {r: data.load_run(r) for r in C.TEST}
    store = {}                                           # (amp, run) -> (score, label)
    for amp in [0.0] + AMPS:
        for r in C.TEST:
            df, qc = raw[r]
            if amp > 0:
                df2, evs = E.inject_events(df, amp, N_EVENTS, seed=1000 + C.TEST.index(r))
            else:
                df2, evs = df, np.array([], int)
            ft = features.extract_run_features(df2, r, None, session=qc["session"])
            ft["score"], ft["nch"] = bl.score(ft)
            lab = E.label_segments(ft.start.to_numpy(), evs)
            lab[~ft.cruise.to_numpy() | ft.score.isna().to_numpy()] = -1     # unscorable -> ignored
            store[(amp, r)] = (ft.score.to_numpy(), lab)
            if amp == 0.0:
                sc = ft.score.dropna()
                results["clean_flag_rate"][r] = {"n": int(len(sc)),
                    "review_pct": float((sc >= review_thr).mean() * 100),
                    "high_pct": float((sc >= high_thr).mean() * 100)}
        if amp == 0.0:
            continue
        sc = np.concatenate([store[(amp, r)][0] for r in C.TEST])
        lab = np.concatenate([store[(amp, r)][1] for r in C.TEST])
        res = {}
        for name, thr in (("review", review_thr), ("high", high_thr)):
            m = E.prf(np.nan_to_num(sc, nan=-1) >= thr, lab)
            m["recall_ci95"] = E.wilson(m["tp"], m["tp"] + m["fn"])
            m["fpr_ci95"] = E.wilson(m["fp"], m["fp"] + m["tn"])
            res[name] = m
        results["amps"][str(amp)] = res
        r_, h_ = res["review"], res["high"]
        print(f"A={amp:4.1f} g  REVIEW: P={r_['precision']:.2f} R={r_['recall']:.2f} [{r_['recall_ci95'][0]:.2f},{r_['recall_ci95'][1]:.2f}] "
              f"F1={r_['f1']:.2f} FPR={r_['fpr']:.3f} | HIGH: R={h_['recall']:.2f} FPR={h_['fpr']:.3f} (positives {r_['tp']+r_['fn']})")

    sw = []
    for amp in AMPS:
        sc = np.concatenate([store[(amp, r)][0] for r in C.TEST])
        lab = np.concatenate([store[(amp, r)][1] for r in C.TEST])
        for thr in SWEEP:
            m = E.prf(np.nan_to_num(sc, nan=-1) >= thr, lab); m.update(amp=amp, thr=float(thr)); sw.append(m)
    sw = pd.DataFrame(sw)
    sw.to_csv(C.OUT_DIR / "threshold_sweep.csv", index=False)
    best = float(sw.groupby("thr").f1.mean().idxmax())
    results["test_tuned_review_thr_mean_f1"] = best
    print("clean-test flag rates:", {k: (round(v['review_pct'], 1), round(v['high_pct'], 1)) for k, v in results["clean_flag_rate"].items()})
    print("REVIEW threshold maximizing mean F1 over amplitudes on TEST (optimistic - tuned on test):", best)
    json.dump(results, open(C.OUT_DIR / "results.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
