# Budget-matched ladder: 1500-step interim summary

**Status.** The 1500-step ladder is complete (12/12 models). The 3000-step ladder is now complete as well; see `budget_match_complete_summary.md` for the final 24-model analysis. This file is retained as the 1500-step snapshot. The original 3000 run completed all models but crashed on a final Unicode print before writing JSON; the JSON was recovered from `budget_match_out.txt`.

## Per-seed results (1500 steps)

| Stage | Rule | Seed 0 | Seed 1 | Seed 2 | Mean | Std | Median |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | raw $k$, $\beta=1$ | 0.0853 | 0.0157 | 0.1722 | 0.0911 | 0.0784 | 0.0853 |
| 1 | normalized $k$, $\beta=1$ | 0.1849 | 0.1935 | 0.1972 | 0.1919 | 0.0063 | 0.1935 |
| 2 | normalized $k$, $\beta=0.125$ | 0.8686 | 0.8754 | 0.8489 | 0.8643 | 0.0138 | 0.8686 |
| 3 | normalized $k$, Kaczmarz | 0.9364 | 0.9408 | 0.9345 | 0.9372 | 0.0032 | 0.9364 |

**Final losses (1500 steps):**

| Stage | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| 0 raw | 4.0503 | 4.1678 | 2.9038 |
| 1 normalized | 3.0819 | 2.9058 | 2.8917 |
| 2 fixed | 0.4753 | 0.4678 | 0.6396 |
| 3 Kaczmarz | 0.2787 | 0.1926 | 0.4067 |

The 1500-step means reproduce the baseline ladder: $0.0911 \to 0.1919 \to 0.8643 \to 0.9372$ (baseline table: $0.091 \to 0.192 \to 0.864 \to 0.937$).

## Stage-to-stage paired deltas (1500 steps; seed-paired)

| Transition | Operation | Paired $\Delta$ | 95% CI | Per-seed $\Delta$ | Relative |
|---|---|---:|---:|---|---:|
| 0 $\to$ 1 | stability | $+0.1008$ | $\pm 0.1898$ | $[0.0996, 0.1778, 0.0250]$ | n/a (CI includes 0) |
| 1 $\to$ 2 | erase strength | $+0.6724$ | $\pm 0.0447$ | $[0.6837, 0.6819, 0.6517]$ | $+350.5\%$ |
| 2 $\to$ 3 | gating | $+0.0729$ | $\pm 0.0274$ | $[0.0678, 0.0654, 0.0856]$ | $+8.4\%$ |

**Reading.** The stability transition has a mean difference whose 95% CI includes zero; at 1500 steps the robust effect is variance reduction (std $0.0784 \to 0.0063$). The erase-strength transition is large and its CI excludes zero. The residual gating transition is smaller but its CI also excludes zero at 1500 steps. The gate's 1500-step advantage does not persist at 3000 steps ($+0.0030 \pm 0.0032$, 95% CI includes zero); see `budget_match_complete_summary.md`.

## 3000-step update (superseded)

The 3000-step ladder is complete (12/12 models). The final per-seed values, paired deltas, and hypothesis judgments are in `budget_match_complete_summary.md`. The partial values below are retained only as history; do not use them for a budget effect.

| Stage | Seed | Accuracy | Final loss |
|---|---:|---:|---:|
| 0 raw | 0 | 0.2714 | 2.2975 |
| 0 raw | 1 | 0.2339 | 2.3783 |

## Interpretation constraints

- **Do not compare `stage1_norm` ($\beta=1$) with `beta_power` $\mu=1.0$ ($\beta\approx0.093$).** The correct budget test for stage 1 is `stage1_norm` at 1500 vs 3000.
- **Kaczmarz mean.** $E[1/\lVert k\rVert^2]\approx0.157$ over all writes; $0.093$ is the `ema` scalar $1/E[\lVert k\rVert^2]$ at $\mu=1.0$. The matched-mean `const@E` uses the model-specific $E[1/\lVert k\rVert^2]$.
- **Raw-key stage 0 is bimodal.** Report per-seed values and the median; the mean is an average over an escape/no-escape mixture.
- **Do not use the shape-sensitivity ratio.** Report absolute and relative drops; the 1500-step relative changes above are stage-to-stage accuracy changes, not the matched-mean shape ratio.

## Provenance

- Run: `budget_match.py` (24 models: 4 stages $\times$ 2 budgets $\times$ 3 seeds; `Wv` bias=False).
- Output: `budget_match_out.txt` (UTF-16LE with BOM).
- Analysis: `budget_match_analysis.py`.
- Final JSON (when complete): `budget_match_results.json`.
- 1500-step and 3000-step results are complete; see `budget_match_complete_summary.md`.
