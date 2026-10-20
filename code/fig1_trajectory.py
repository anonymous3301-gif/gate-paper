"""Figure 1: the raw-key trajectory, the clip-protocol control, and the ladder's
per-seed spread at both budgets.

Panel (a)  training loss (log scale) for the raw-key model under three
          clipping protocols and for the normalized-key model (seed 0).
          Source: clip_ablation_log.csv (raw clip 1.0 / 10.0 / none, and
          normalized clip 1.0 from rawk_log.csv).
Panel (b)  E||k||^2 over the same runs, on a log scale, which is where the
          norm-collapse escape is visible.
Panel (c)  per-seed accuracy for stages 0 (raw) and 1 (normalized) at 1500
          and 3000 steps, from budget_match_complete_summary.md.

Run: python fig1_trajectory.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C = Path(r"C:\ai_exp")

COL = {
    "raw_clip1.0": ("#C0392B", "-", r"raw $k$, clip $=1.0$"),
    "raw_clip10.0": ("#E67E22", "--", r"raw $k$, clip $=10.0$"),
    "raw_noclip": ("#7F8C8D", ":", r"raw $k$, no clip"),
    "norm_clip1.0": ("#2980B9", "-", r"normalized $k$, clip $=1.0$"),
}

# per-seed accuracy: stage -> budget -> [seed0, seed1, seed2]
LADDER = {
    ("raw", 1500): [0.0853, 0.0157, 0.1722],
    ("raw", 3000): [0.2714, 0.2339, 0.2638],
    ("norm", 1500): [0.1849, 0.1935, 0.1972],
    ("norm", 3000): [0.6662, 0.7041, 0.5740],
}


def load_clip_log():
    """clip_ablation_log.csv has no header: name, seed, step, loss, E_k2, max_k2, grad."""
    out = defaultdict(list)
    with open(C / "clip_ablation_log.csv", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 7:
                continue
            out[(row[0], int(row[1]))].append(
                (int(row[2]), float(row[3]), float(row[4]), float(row[5])))
    for k in out:
        out[k].sort()
    return out


def load_rawk_log():
    out = defaultdict(list)
    with open(C / "rawk_log.csv", newline="") as f:
        for row in csv.DictReader(f):
            out[row["model"]].append(
                (int(row["step"]), float(row["loss"]), float(row["E_k2"]),
                 float(row["max_k2"])))
    for k in out:
        out[k].sort()
    return out


def main():
    clip = load_clip_log()
    rawk = load_rawk_log()

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1))

    # ---------------- panel (a): loss ----------------
    ax = axes[0]
    for key, sel in [("raw_clip1.0", ("raw_clip1.0", 0)),
                     ("raw_clip10.0", ("raw_clip10.0", 0)),
                     ("raw_noclip", ("raw_noclip", 0))]:
        rows = clip[sel]
        c, ls, lab = COL[key]
        lw = 3.0 if key == "raw_clip1.0" else 1.1
        ax.plot([r[0] for r in rows], [r[1] for r in rows], color=c, ls=ls,
                lw=lw, label=lab, alpha=0.95 if key == "raw_clip1.0" else 1.0)
    rows = rawk["norm"]
    c, ls, lab = COL["norm_clip1.0"]
    ax.plot([r[0] for r in rows], [r[1] for r in rows], color=c, ls=ls, lw=1.4,
            label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("training loss (log)")
    ax.set_title("(a) Loss: the freeze and its escape", loc="left", fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.95)
    ax.grid(alpha=0.25, lw=0.6, which="both")
    ax.set_xlim(0, 1500)

    # ---------------- panel (b): E||k||^2 ----------------
    ax = axes[1]
    for key, sel in [("raw_clip1.0", ("raw_clip1.0", 0)),
                     ("raw_clip10.0", ("raw_clip10.0", 0)),
                     ("raw_noclip", ("raw_noclip", 0))]:
        rows = clip[sel]
        c, ls, lab = COL[key]
        lw = 3.0 if key == "raw_clip1.0" else 1.1
        ax.plot([r[0] for r in rows], [r[2] for r in rows], color=c, ls=ls,
                lw=lw, label=lab, alpha=0.95 if key == "raw_clip1.0" else 1.0)
    rows = rawk["norm"]
    c, ls, lab = COL["norm_clip1.0"]
    ax.plot([r[0] for r in rows], [r[2] for r in rows], color=c, ls=ls, lw=1.4,
            label=lab)
    ax.axhline(2.0, color="#27AE60", lw=1.0)
    ax.text(20, 2.25, r"$\beta\| k\|^2=2$ with $\beta=1$", color="#1E8449",
            fontsize=7.5)
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(r"$E\| k\|^2$ (log)")
    ax.set_title("(b) Key norms: the collapse is the escape", loc="left",
                 fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.95)
    ax.grid(alpha=0.25, lw=0.6, which="both")
    ax.set_xlim(0, 1500)

    # ---------------- panel (c): per-seed strip ----------------
    ax = axes[2]
    groups = [("raw", 1500), ("norm", 1500), ("raw", 3000), ("norm", 3000)]
    labels = ["stage 0\nraw\n1500", "stage 1\nnorm\n1500",
              "stage 0\nraw\n3000", "stage 1\nnorm\n3000"]
    colors = ["#C0392B", "#C0392B", "#E67E22", "#E67E22"]
    for i, (g, b) in enumerate(groups):
        vals = LADDER[(g, b)]
        jitter = np.linspace(-0.09, 0.09, len(vals))
        ax.plot(i + jitter, vals, "o", ms=6, color=colors[i], mec="white",
                mew=0.8, zorder=3)
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1))
        ax.plot([i - 0.22, i + 0.22], [m, m], color="#2C3E50", lw=1.6, zorder=4)
        ax.plot([i - 0.22, i - 0.22], [m - s, m + s], color="#2C3E50", lw=1.2,
                zorder=4)
        ax.plot([i + 0.22, i + 0.22], [m - s, m + s], color="#2C3E50", lw=1.2,
                zorder=4)
        ax.text(i + 0.28, m, f"{m:.4f}\n$\\pm${s:.4f}", fontsize=7.5,
                va="center", color="#2C3E50")
    ax.axhline(1.0 / 64, color="#95A5A6", lw=1.0, ls=":")
    ax.text(-0.45, 1.0 / 64 + 0.02, "chance 1/64", color="#7F8C8D", fontsize=7.5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("MQAR accuracy")
    ax.set_title("(c) Stages 0 and 1, both budgets (3 seeds)", loc="left",
                 fontweight="bold")
    ax.set_xlim(-0.6, 3.75)
    ax.set_ylim(-0.03, 0.85)
    ax.grid(alpha=0.25, lw=0.6, axis="y")

    fig.tight_layout()
    fig.savefig(C / "fig1_trajectory.png", dpi=300, bbox_inches="tight")
    fig.savefig(C / "fig1_trajectory.pdf", bbox_inches="tight")
    print("wrote fig1_trajectory.png / .pdf")

    # echo the numbers used in panel (c)
    for (g, b), vals in LADDER.items():
        print(f"  panel (c) stage {g:4s} {b}: {vals} mean {np.mean(vals):.4f} "
              f"sd {np.std(vals, ddof=1):.4f}")


if __name__ == "__main__":
    main()
