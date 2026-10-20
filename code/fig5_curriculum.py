"""Figure 5: what a curriculum does and does not rescue (Section 6).

Panel (a)  1500 steps: for each write rule, the three seeds without a
          curriculum and the same three seeds with one, connected pairwise.
          Shows that the curriculum removes the raw stage's chance-level seed
          without lifting the level, does nothing under Kaczmarz, and inflates
          the seed spread of the normalized stage.
Panel (b)  budget: the normalized beta=1 stage with and without a curriculum
          at 1500 and 3000 steps, with the Kaczmarz levels of the two budgets
          as reference lines.

Data:
  norm_curriculum_out.txt   raw / normalized x curriculum, 1500 steps, 3 seeds
  curriculum_2x2_out.txt    one / kaczmarz x curriculum, 1500 steps, 3 seeds
  curriculum_budget_out.txt  one x curriculum, 3000 steps, 3 seeds
  budget_match_complete_summary.md  the no-curriculum ladder rows

Run: python fig5_curriculum.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C = Path(r"C:\ai_exp")


def read(p):
    raw = (C / p).read_bytes()
    enc = "utf-16" if raw[:2] == b"\xff\xfe" else "utf-8"
    return raw.decode(enc, errors="ignore")


def parse_norm_curriculum():
    """key_norm curriculum seed0 seed1 seed2 mean std"""
    out = {}
    for line in read("norm_curriculum_out.txt").split("\n"):
        p = line.split()
        if len(p) == 7 and p[0] in ("True", "False"):
            out[(p[0] == "True", p[1] == "True")] = [float(p[2]), float(p[3]), float(p[4])]
    return out


def parse_2x2():
    """beta curriculum seed0 seed1 seed2 mean"""
    out = {}
    for line in read("curriculum_2x2_out.txt").split("\n"):
        p = line.split()
        if len(p) == 6 and p[0] in ("one", "kaczmarz"):
            out[(p[0], p[1] == "True")] = [float(p[2]), float(p[3]), float(p[4])]
    return out


def parse_budget():
    """beta curriculum steps seed0 seed1 seed2 mean"""
    out = {}
    for line in read("curriculum_budget_out.txt").split("\n"):
        p = line.split()
        if len(p) == 7 and p[0] in ("one", "kaczmarz"):
            out[(p[0], p[1] == "True", int(p[2]))] = [float(p[3]), float(p[4]), float(p[5])]
    return out


def main():
    nc = parse_norm_curriculum()
    two = parse_2x2()
    bud = parse_budget()
    # ladder rows (no curriculum) for the budget reference
    ladder = {
        ("one", 1500): [0.1849, 0.1935, 0.1972],
        ("one", 3000): [0.6662, 0.7041, 0.5740],
        ("kaczmarz", 1500): [0.9364, 0.9408, 0.9345],
        ("kaczmarz", 3000): [0.9458, 0.9482, 0.9444],
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    # ---------------- panel (a) ----------------
    ax = axes[0]
    groups = [
        ("raw $k$, $\\beta=1$", nc[(False, False)], nc[(False, True)], "#C0392B"),
        ("normalized $k$, $\\beta=1$", nc[(True, False)], nc[(True, True)], "#2980B9"),
        ("Kaczmarz", two[("kaczmarz", False)], two[("kaczmarz", True)], "#2C3E50"),
    ]
    for i, (name, off, on, col) in enumerate(groups):
        for s in range(3):
            ax.plot([i - 0.16, i + 0.16], [off[s], on[s]], color=col, lw=0.8,
                    alpha=0.45, zorder=2)
        ax.plot([i - 0.16] * 3, off, "o", ms=5.5, color=col, mec="white", mew=0.7,
                zorder=3)
        ax.plot([i + 0.16] * 3, on, "o", ms=5.5, color=col, mec="white", mew=0.7,
                alpha=0.55, zorder=3)
        ax.plot([i - 0.16], [np.mean(off)], "_", ms=13, mew=2.0, color=col, zorder=4)
        ax.plot([i + 0.16], [np.mean(on)], "_", ms=13, mew=2.0, color=col, alpha=0.55,
                zorder=4)
    ax.axhline(1.0 / 64, color="#95A5A6", lw=1.0, ls=":")
    ax.text(2.45, 1.0 / 64 + 0.015, "chance 1/64", color="#7F8C8D", fontsize=7.5, ha="right")
    ax.set_xticks(range(3))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.5)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.03, 1.16)
    ax.set_ylabel("MQAR accuracy")
    ax.set_title("(a) 1500 steps: no curriculum (left) vs curriculum (right)",
                 loc="left", fontweight="bold")
    ax.grid(alpha=0.25, lw=0.6, axis="y")
    ax.text(0.0, 1.05, "raw: the chance-level seed is\nremoved, the level is not",
            fontsize=7.5, ha="center", color="#7B241C")
    ax.text(1.0, 1.05, "normalized: the mean rises,\nthe spread explodes",
            fontsize=7.5, ha="center", color="#1A5276")
    ax.text(2.0, 1.05, "Kaczmarz: nothing\nhappens", fontsize=7.5, ha="center",
            color="#2C3E50")

    # ---------------- panel (b) ----------------
    ax = axes[1]
    xpos = [0, 1, 2, 3]
    series = [
        ("no curriculum, 1500", ladder[("one", 1500)], 0, "#C0392B"),
        ("curriculum, 1500", nc[(True, True)], 1, "#2980B9"),
        ("no curriculum, 3000", ladder[("one", 3000)], 2, "#C0392B"),
        ("curriculum, 3000", bud[("one", True, 3000)], 3, "#2980B9"),
    ]
    for lab, vals, x, col in series:
        ax.plot([x] * len(vals), vals, "o", ms=6, color=col, mec="white", mew=0.8,
                zorder=3)
        ax.plot([x - 0.18, x + 0.18], [np.mean(vals)] * 2, color="#2C3E50", lw=1.8,
                zorder=4)
        ax.text(x + 0.24, np.mean(vals), f"{np.mean(vals):.4f}", fontsize=7.5,
                va="center", color="#2C3E50")
    for b, lab, xt in ((1500, "Kaczmarz, 1500", -0.42), (3000, "Kaczmarz, 3000", 1.55)):
        v = np.mean(ladder[("kaczmarz", b)])
        ax.axhline(v, color="#7F8C8D", lw=0.9, ls="--")
        ax.text(xt, v + 0.012, f"{lab}  ({v:.4f})", fontsize=7,
                va="bottom", color="#7F8C8D")
    ax.axhline(1.0 / 64, color="#95A5A6", lw=1.0, ls=":")
    ax.set_xticks(xpos)
    ax.set_xticklabels([s[0].replace(", ", ",\n") for s in series], fontsize=7.5)
    ax.set_xlim(-0.5, 3.9)
    ax.set_ylim(-0.03, 1.02)
    ax.set_ylabel("MQAR accuracy")
    ax.set_title("(b) Budget: the curriculum's advantage is a short-budget effect",
                 loc="left", fontweight="bold")
    ax.grid(alpha=0.25, lw=0.6, axis="y")

    fig.tight_layout()
    fig.savefig(C / "fig5_curriculum.png", dpi=300, bbox_inches="tight")
    fig.savefig(C / "fig5_curriculum.pdf", bbox_inches="tight")
    print("wrote fig5_curriculum.png / .pdf")
    for name, off, on, _ in groups:
        print(f"  {name:26s} no-curr {np.mean(off):.4f} (sd {np.std(off, ddof=1):.4f})  "
              f"curr {np.mean(on):.4f} (sd {np.std(on, ddof=1):.4f})  "
              f"delta {np.mean(on)-np.mean(off):+.4f}")


if __name__ == "__main__":
    main()
