"""Equivalence (TOST) test for the matched-mean transfer result (Section 5.3).

Question: for a Kaczmarz-trained model, is the *shape* of the gate responsible
for the transfer cost, or only its mean? Equivalently: is a gate with the
Kaczmarz shape but the fixed rule's mean (gate_m) equivalent, within a stated
tolerance, to the Kaczmarz rule itself?

The comparison is *paired by construction*: the same trained model is evaluated
under four rules, so the difference gate_m - kacz is a within-model quantity.
The paired analysis is therefore the correct one, and it is more precise than
the difference of two independent means.

Data (frozen weights, one evaluation per rule):
  cross_eval_matched_out.txt  d=16,  3 seeds per training rule
  cross_eval_d24_out.txt      d=24, 10 seeds per training rule
  cross_eval_d32_out.txt      d=32,  3 seeds per training rule

Margin provenance: +-0.01 is chosen FOR THIS ANALYSIS (it is not a
pre-registered quantity in this paper) and is justified substantively: it is
1 percentage point of absolute accuracy, about 14% of the baseline-budget gate
advantage (+0.0729) that Section 5 discusses, and an order of magnitude larger
than the differences the equivalence claim is about (|delta| <= 0.0011). It is
a tolerance, not a target.
"""

import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

C = Path(r"C:\ai_exp")
FILES = {16: "cross_eval_matched_out.txt", 24: "cross_eval_d24_out.txt",
         32: "cross_eval_d32_out.txt"}
MARGIN = 0.01
ALPHA = 0.05


def load(d):
    raw = (C / FILES[d]).read_bytes()
    enc = "utf-16" if raw[:2] == b"\xff\xfe" else "utf-8"
    text = raw.decode(enc, errors="ignore")
    rows = []
    for line in text.split("\n"):
        parts = line.split()
        if len(parts) == 7 and parts[0] in ("fixed", "kaczmarz"):
            rows.append(dict(train=parts[0], seed=int(parts[1]),
                             Einv_all=float(parts[2]), fixed=float(parts[3]),
                             kacz=float(parts[4]), gate_m=float(parts[5]),
                             const_m=float(parts[6])))
    return rows


def paired_stats(deltas):
    d = np.asarray(deltas, dtype=float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    df = n - 1
    tcrit = stats.t.ppf(0.975, df)
    t_lower = (mean + MARGIN) / se
    t_upper = (mean - MARGIN) / se
    p_lower = 1 - stats.t.cdf(t_lower, df)      # H0: delta <= -margin
    p_upper = stats.t.cdf(t_upper, df)          # H0: delta >= +margin
    p_tost = max(p_lower, p_upper)
    return dict(n=n, mean=mean, sd=sd, se=se, df=df,
                ci=(mean - tcrit * se, mean + tcrit * se),
                p_lower=float(p_lower), p_upper=float(p_upper), p_tost=float(p_tost),
                equivalent=bool(p_tost < ALPHA))


def unpaired_stats(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    df = len(a) + len(b) - 2
    mean = a.mean() - b.mean()
    t_lower = (mean + MARGIN) / se
    t_upper = (mean - MARGIN) / se
    p = max(1 - stats.t.cdf(t_lower, df), stats.t.cdf(t_upper, df))
    return dict(mean=float(mean), se=float(se), df=df, p_tost=float(p),
                equivalent=bool(p < ALPHA))


def main():
    print(f"TOST, margin +-{MARGIN} (chosen for this analysis, not pre-registered), alpha={ALPHA}")
    print()
    out = {}
    for d in (16, 24, 32):
        rows = load(d)
        kz = [r for r in rows if r["train"] == "kaczmarz"]
        fx = [r for r in rows if r["train"] == "fixed"]
        print(f"=== d_mem={d}  (n={len(kz)} kaczmarz-trained, {len(fx)} fixed-trained)")

        # the equivalence question: shape-matched gate vs the gate itself
        dm = [r["gate_m"] - r["kacz"] for r in kz]
        st = paired_stats(dm)
        up = unpaired_stats([r["gate_m"] for r in kz], [r["kacz"] for r in kz])
        print(f"  [shape vs mean] gate_m - kacz  (kaczmarz-trained)")
        print(f"    per-seed: {', '.join(f'{v:+.4f}' for v in dm)}")
        print(f"    paired: mean {st['mean']:+.5f}  sd {st['sd']:.5f}  SE {st['se']:.5f}  "
              f"df {st['df']}  95% CI [{st['ci'][0]:+.5f}, {st['ci'][1]:+.5f}]")
        print(f"    TOST: p_lower {st['p_lower']:.3e}  p_upper {st['p_upper']:.3e}  "
              f"p_TOST {st['p_tost']:.3e}  -> equivalent: {st['equivalent']}")
        print(f"    (unpaired version for reference: mean {up['mean']:+.5f}  SE {up['se']:.5f}  "
              f"p_TOST {up['p_tost']:.3e}  -> equivalent: {up['equivalent']})")

        # the large effect: mean-matched constant vs the gate itself
        dc = [r["const_m"] - r["kacz"] for r in kz]
        sc = paired_stats(dc)
        print(f"  [mean vs shape] const_m - kacz (kaczmarz-trained): mean {sc['mean']:+.4f}  "
              f"95% CI [{sc['ci'][0]:+.4f}, {sc['ci'][1]:+.4f}]  (the shape cost)")

        # mirror image on fixed-trained models
        dmf = [r["gate_m"] - r["fixed"] for r in fx]
        sf = paired_stats(dmf)
        dcf = [r["const_m"] - r["fixed"] for r in fx]
        scf = paired_stats(dcf)
        print(f"  [fixed-trained] gate_m - fixed: mean {sf['mean']:+.4f}  "
              f"95% CI [{sf['ci'][0]:+.4f}, {sf['ci'][1]:+.4f}]  "
              f"TOST p {sf['p_tost']:.3e} -> equivalent: {sf['equivalent']}")
        print(f"  [fixed-trained] const_m - fixed: mean {scf['mean']:+.4f}  "
              f"95% CI [{scf['ci'][0]:+.4f}, {scf['ci'][1]:+.4f}]")
        print()
        out[d] = dict(shape_vs_mean=st, mean_vs_shape=sc,
                      fixed_gate_m=sf, fixed_const_m=scf,
                      unpaired_shape_vs_mean=up)
    return out


if __name__ == "__main__":
    main()
