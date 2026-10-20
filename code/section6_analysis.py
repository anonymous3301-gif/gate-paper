"""Section 6 analysis: curriculum x budget, paired over seeds.

Sources (all d_mem=16, d=64, lr 3e-4, batch 32, V=64, MQAR 8 pairs / 32
distractors at evaluation):

  curriculum_2x2_out.txt  : {one, kaczmarz} x {curriculum off, on}, 1500 steps
  norm_curriculum_out.txt : {raw, normalized} x {curriculum off, on}, 1500 steps
  curriculum_budget_out.txt: {one} x {curriculum off, on}, 3000 steps
  budget_match_complete_summary.md: the no-curriculum ladder rows (identical
                                    protocol; the shared cells reproduce
                                    exactly, see checks below)

Curriculum = n_dist 8 for the first half of training, then 32; evaluation is
always at n_dist=32.

The 95% CI is a seed-paired t interval with n=3 (t_{0.975,2} = 4.302653).
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

T_CRIT = stats.t.ppf(0.975, df=2)

runs = {
    ("one", False, 1500): [0.1849, 0.1935, 0.1972],
    ("one", True, 1500): [0.6604, 0.2052, 0.4284],
    ("kaczmarz", False, 1500): [0.9364, 0.9408, 0.9345],
    ("kaczmarz", True, 1500): [0.9324, 0.9430, 0.9383],
    ("raw", False, 1500): [0.0853, 0.0157, 0.1722],
    ("raw", True, 1500): [0.2340, 0.1896, 0.2097],
    ("one", False, 3000): [0.6662, 0.7041, 0.5740],
    ("one", True, 3000): [0.9230, 0.9275, 0.9192],
}


def describe(a):
    a = np.asarray(a, dtype=float)
    return {
        "seeds": [round(float(x), 4) for x in a],
        "mean": float(a.mean()),
        "std_sample": float(a.std(ddof=1)),
        "std_pop": float(a.std()),
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def paired(a, b):
    """a - b, seed-paired."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = len(d)
    mean = float(d.mean())
    s = float(d.std(ddof=1))
    half = T_CRIT * s / np.sqrt(n)
    return {
        "per_seed": [round(float(x), 4) for x in d],
        "mean": mean,
        "ci_half": float(half),
        "ci": [float(mean - half), float(mean + half)],
        "includes_zero": bool((mean - half) <= 0 <= (mean + half)),
        "all_same_sign": bool((d > 0).all() or (d < 0).all()),
    }


out = {"config": {
    "d_mem": 16, "d_model": 64, "lr": 3e-4, "batch": 32, "V": 64,
    "n_pairs": 8, "n_dist_eval": 32,
    "curriculum": "n_dist 8 for the first half of training, then 32",
    "ci": "seed-paired t interval, n=3, t=4.302653",
}, "runs": {}, "paired": {}}

for key, vals in runs.items():
    out["runs"][f"{key[0]}|curriculum={key[1]}|steps={key[2]}"] = describe(vals)

comparisons = [
    ("curriculum effect, one (normalized, beta=1), 1500",
     ("one", True, 1500), ("one", False, 1500)),
    ("curriculum effect, raw key, beta=1, 1500",
     ("raw", True, 1500), ("raw", False, 1500)),
    ("curriculum effect, kaczmarz, 1500",
     ("kaczmarz", True, 1500), ("kaczmarz", False, 1500)),
    ("budget effect (3000-1500), one, no curriculum",
     ("one", False, 3000), ("one", False, 1500)),
    ("budget effect (3000-1500), one, curriculum",
     ("one", True, 3000), ("one", True, 1500)),
    ("curriculum effect, one (normalized, beta=1), 3000 vs kaczmarz 1500 (unpaired, means only)",
     None, None),
]
for name, a, b in comparisons:
    if a is None:
        continue
    out["paired"][name] = paired(runs[a], runs[b])

out["means_only"] = {
    "one+curriculum, 3000 (mean of 3 seeds)": describe(runs[("one", True, 3000)])["mean"],
    "kaczmarz, 1500 (mean of 3 seeds)": describe(runs[("kaczmarz", False, 1500)])["mean"],
    "kaczmarz, 3000 (budget-matched ladder stage 3)": 0.9461,
    "fixed beta=0.125, 1500 (ladder stage 2)": 0.8643,
    "fixed beta=0.125, 3000 (ladder stage 2)": 0.9431,
}

Path("section6_analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

for k, v in out["runs"].items():
    print(f"{k:42s} mean={v['mean']:.4f} sd={v['std_sample']:.4f} "
          f"median={v['median']:.4f} range=[{v['min']:.4f},{v['max']:.4f}]")
print()
for k, v in out["paired"].items():
    print(f"{k}")
    print(f"    per-seed {v['per_seed']}  mean {v['mean']:+.4f} +/- {v['ci_half']:.4f}"
          f"  CI={v['ci'][0]:+.4f}..{v['ci'][1]:+.4f}  includes0={v['includes_zero']}"
          f"  same_sign={v['all_same_sign']}")
print()
print("means only:", json.dumps(out["means_only"], indent=2))
print("wrote section6_analysis.json")
