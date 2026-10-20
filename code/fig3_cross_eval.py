"""Figure 3: the matched-mean post-hoc transfer matrix at d_mem = 16 and 32.

Two panels, one per capacity. Rows are the training rule, columns are the
evaluation rule applied to the frozen model; each cell is the mean accuracy
over the seeds, annotated with the mean and (in brackets) the per-seed values
are in the text. The point of the figure is the asymmetry: a Kaczmarz-trained
model loses almost everything when the gate is replaced by a constant of the
same mean, while a fixed-trained model is barely affected by either swap.

Data:
  cross_eval_matched_out.txt  d_mem=16, 3 seeds per training rule
  cross_eval_d32_out.txt      d_mem=32, 3 seeds per training rule
Columns in the file: train, seed, Einv_all, fixed, kacz, gate_m, const_m

Run: python fig3_cross_eval.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C = Path(r"C:\ai_exp")
FILES = {16: "cross_eval_matched_out.txt", 32: "cross_eval_d32_out.txt"}
COLS = ["fixed", "kacz", "gate_m", "const_m"]
COL_LABELS = ["fixed\n($\\beta=0.125$)", "Kaczmarz\n($1/\\|k\\|^2$)",
              "gate@0.125\n(Kacz shape,\nmean 0.125)", "const@E\n(constant at\nthe gate mean)"]


def load(d):
    raw = (C / FILES[d]).read_bytes()
    enc = "utf-16" if raw[:2] == b"\xff\xfe" else "utf-8"
    text = raw.decode(enc, errors="ignore")
    rows = []
    for line in text.split("\n"):
        p = line.split()
        if len(p) == 7 and p[0] in ("fixed", "kaczmarz"):
            rows.append(dict(train=p[0], seed=int(p[1]),
                             fixed=float(p[3]), kacz=float(p[4]),
                             gate_m=float(p[5]), const_m=float(p[6])))
    return rows


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for ax, d in zip(axes, (16, 32)):
        rows = load(d)
        trains = ["fixed", "kaczmarz"]
        M = np.zeros((2, 4))
        for i, tr in enumerate(trains):
            sub = [r for r in rows if r["train"] == tr]
            for j, c in enumerate(COLS):
                M[i, j] = np.mean([r[c] for r in sub])
        im = ax.imshow(M, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        for i in range(2):
            for j in range(4):
                trained_here = (trains[i] == "fixed" and COLS[j] == "fixed") or \
                               (trains[i] == "kaczmarz" and COLS[j] == "kacz")
                ax.text(j, i - 0.16, f"{M[i, j]:.4f}", ha="center", va="center",
                        fontsize=10, fontweight="bold" if trained_here else "normal",
                        color="black")
                if trained_here:
                    ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                               fill=False, ec="#2C3E50", lw=2.0))
        # annotate the two decisive drops
        # the two decisive drops, computed per capacity
        shape_cost = M[0, 2] - M[0, 0]      # fixed-trained, gate@0.125 vs fixed
        mean_cost = M[1, 3] - M[1, 1]        # kacz-trained, const@E vs Kaczmarz
        ax.annotate("", xy=(2, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#7B241C", lw=1.8))
        ax.text(1.0, 0.36, f"shape cost\n{shape_cost:+.3f}", color="#7B241C",
                fontsize=8.5, ha="center")
        ax.annotate("", xy=(3, 1), xytext=(1, 1),
                    arrowprops=dict(arrowstyle="->", color="#7B241C", lw=1.8))
        ax.text(2.0, 1.36, f"mean cost\n{mean_cost:+.3f}", color="#7B241C",
                fontsize=8.5, ha="center")
        ax.set_xticks(range(4))
        ax.set_xticklabels(COL_LABELS, fontsize=8)
        ax.set_yticks(range(2))
        ax.set_yticklabels(["trained with\n`fixed`", "trained with\n`Kaczmarz`"],
                           fontsize=8.5)
        ax.set_title(f"(d_mem = {d}, 3 seeds)", loc="left", fontweight="bold")
    fig.suptitle("Post-hoc transfer: the trained rule fixes which swap is harmful",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(C / "fig3_cross_eval.png", dpi=300, bbox_inches="tight")
    fig.savefig(C / "fig3_cross_eval.pdf", bbox_inches="tight")
    print("wrote fig3_cross_eval.png / .pdf")
    for d in (16, 32):
        rows = load(d)
        for tr in ("fixed", "kaczmarz"):
            sub = [r for r in rows if r["train"] == tr]
            print(f"  d={d} trained {tr:9s} " +
                  "  ".join(f"{c}={np.mean([r[c] for r in sub]):.4f}" for c in COLS))


if __name__ == "__main__":
    main()
