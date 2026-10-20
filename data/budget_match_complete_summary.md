# Budget-matched ladder: complete 1500/3000 summary

**Status.** All 24 models are complete (4 stages × 2 budgets × 3 seeds). The original run completed every model but crashed with a Windows `cp932` `UnicodeEncodeError` on the final em-dash print, before writing `budget_match_results.json`. The JSON was recovered from `budget_match_out.txt` by `recover_budget_match_json.py`; all 24 result rows are present. The recovered JSON contains final losses, not the full per-step loss histories: those were in memory at the crash and are not recoverable from the output file. The crash was a print-only failure, not an experiment failure. No run was stopped or rerun. `budget_match.py` now writes JSON before the comparison prints and uses ASCII in those prints.

## Per-seed accuracy

| Stage | Rule | Budget | Seed 0 | Seed 1 | Seed 2 | Mean | Std | Median |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 0 | raw $k$, $\beta=1$ | 1500 | 0.0853 | 0.0157 | 0.1722 | 0.0911 | 0.0784 | 0.0853 |
| 0 | raw $k$, $\beta=1$ | 3000 | 0.2714 | 0.2339 | 0.2638 | 0.2564 | 0.0198 | 0.2638 |
| 1 | normalized $k$, $\beta=1$ | 1500 | 0.1849 | 0.1935 | 0.1972 | 0.1919 | 0.0063 | 0.1935 |
| 1 | normalized $k$, $\beta=1$ | 3000 | 0.6662 | 0.7041 | 0.5740 | 0.6481 | 0.0669 | 0.6662 |
| 2 | normalized $k$, $\beta=0.125$ | 1500 | 0.8686 | 0.8754 | 0.8489 | 0.8643 | 0.0138 | 0.8686 |
| 2 | normalized $k$, $\beta=0.125$ | 3000 | 0.9434 | 0.9460 | 0.9399 | 0.9431 | 0.0031 | 0.9434 |
| 3 | normalized $k$, Kaczmarz | 1500 | 0.9364 | 0.9408 | 0.9345 | 0.9372 | 0.0032 | 0.9364 |
| 3 | normalized $k$, Kaczmarz | 3000 | 0.9458 | 0.9482 | 0.9444 | 0.9461 | 0.0019 | 0.9458 |

## Final training loss

| Stage | Budget | Seed 0 | Seed 1 | Seed 2 |
|---|---|---:|---:|---:|
| 0 raw | 1500 | 4.0503 | 4.1678 | 2.9038 |
| 0 raw | 3000 | 2.2975 | 2.3783 | 2.3096 |
| 1 normalized | 1500 | 3.0819 | 2.9058 | 2.8917 |
| 1 normalized | 3000 | 1.2712 | 1.1061 | 1.4212 |
| 2 fixed | 1500 | 0.4753 | 0.4678 | 0.6396 |
| 2 fixed | 3000 | 0.2187 | 0.1999 | 0.1533 |
| 3 Kaczmarz | 1500 | 0.2787 | 0.1926 | 0.4067 |
| 3 Kaczmarz | 3000 | 0.1881 | 0.1829 | 0.1569 |

## Stage-to-stage paired deltas (seed-paired, 95% CI)

| Budget | Transition | Operation | Paired $\Delta$ | 95% CI | Per-seed $\Delta$ | Relative |
|---:|---|---|---:|---:|---|---:|
| 1500 | 0 → 1 | stability | $+0.1008$ | $\pm 0.1898$ | $[0.0996, 0.1778, 0.0250]$ | n/a (CI includes 0) |
| 1500 | 1 → 2 | erase strength | $+0.6724$ | $\pm 0.0447$ | $[0.6837, 0.6819, 0.6517]$ | $+350.5\%$ |
| 1500 | 2 → 3 | gating | $+0.0729$ | $\pm 0.0274$ | $[0.0678, 0.0654, 0.0856]$ | $+8.4\%$ |
| 3000 | 0 → 1 | stability | $+0.3917$ | $\pm 0.1989$ | $[0.3948, 0.4702, 0.3102]$ | $+152.8\%$ |
| 3000 | 1 → 2 | erase strength | $+0.2950$ | $\pm 0.1587$ | $[0.2772, 0.2419, 0.3659]$ | $+45.5\%$ |
| 3000 | 2 → 3 | gating | $+0.0030$ | $\pm 0.0032$ | $[0.0024, 0.0022, 0.0045]$ | $+0.3\%$ (CI includes 0) |

## Within-rule budget deltas (3000 − 1500, seed-paired)

| Stage | Paired $\Delta$ | 95% CI | Per-seed $\Delta$ | Relative |
|---|---:|---:|---|---:|
| 0 raw | $+0.1653$ | $\pm 0.1635$ | $[0.1861, 0.2182, 0.0916]$ | $+181.5\%$ |
| 1 normalized | $+0.4562$ | $\pm 0.1747$ | $[0.4813, 0.5106, 0.3768]$ | $+237.8\%$ |
| 2 fixed | $+0.0788$ | $\pm 0.0268$ | $[0.0748, 0.0706, 0.0910]$ | $+9.1\%$ |
| 3 Kaczmarz | $+0.0089$ | $\pm 0.0033$ | $[0.0094, 0.0074, 0.0099]$ | $+0.9\%$ |

## Hypothesis judgments

### Stability (stage 0 → 1)

At 1500 steps the paired mean difference is $+0.1008$ with a 95% CI that includes zero; the raw stage spans chance to 0.1722, and the clearest descriptive difference is variance reduction ($0.0784 \to 0.0063$).

At 3000 steps the mean difference is $+0.3917$ ($\pm 0.1989$), excluding zero. The raw stage no longer collapses (0.2339–0.2714), but normalization still raises the mean substantially. The variance pattern reverses ($0.0198 \to 0.0669$): the normalized stage has a seed at 0.574, while the raw stage is uniformly low. **The variance-reduction characterization is specific to 1500 steps; at 3000 the effect is a mean shift.**

### Erase strength (stage 1 → 2)

The effect is large at both budgets: $+0.6724 \pm 0.0447$ at 1500 and $+0.2950 \pm 0.1587$ at 3000. The CI excludes zero at both budgets. It is the largest stage-to-stage gain at 1500. At 3000 the stability point estimate is larger ($+0.3917$ vs $+0.2950$) and the intervals overlap, so we do not rank erase strength above stability at 3000. The effect shrinks with budget because the normalized $\beta=1$ stage improves from 0.192 to 0.648, but the fixed $\beta=0.125$ stage remains far above it (0.943). **Erase strength is the main baseline effect, with a budget-dependent magnitude; its persistence at 3000 is established, its rank at 3000 is not.**

### Gating (stage 2 → 3)

At 1500 steps the gate gain is $+0.0729 \pm 0.0274$; the CI excludes zero. At 3000 steps the same transition is $+0.0030 \pm 0.0032$; the CI includes zero. All three per-seed 3000 deltas are positive but tiny ($+0.0022$ to $+0.0045$). The fixed rule gains $+0.0788$ from the extra budget, while Kaczmarz gains only $+0.0089$; the fixed rule numerically catches up. With three seeds, this is not evidence that the gap is exactly zero.

**Pre-specified question: does the gating gap persist at 3000 steps? Answer: the 3000 CI includes zero, so the data do not provide evidence of a persistent gap. With three seeds we do not claim exact equivalence.** The 1500-step gate advantage is best described as a sample-efficiency effect, not an established persistent training-time bias at convergence. This does not invalidate the matched-mean post-hoc result: a Kaczmarz-trained model still co-adapts to the gate shape (Section 5.3). It bounds the claim: training with the gate is not necessary to reach the same accuracy at 3000 steps in this ladder, within the resolution of three seeds.

## Interpretation constraints

- Three seeds per cell; report per-seed values. The 3000 gating CI includes zero; do not claim a positive gating effect at 3000.
- Do not compare `stage1_norm` ($\beta=1$) with `beta_power` $\mu=1.0$ ($\beta\approx0.093$). The correct budget test is `stage1_norm` at 1500 vs 3000.
- The raw stage is bimodal at 1500 and uniformly low at 3000. Do not summarize it with a single mean only.
- The gate's **training-time accuracy advantage** is not established at 3000; the gate's **shape co-adaptation** is a separate post-hoc result.
- `stage3_kaczmarz` at 3000 is 0.9461; `stage2_fixed` is 0.9431. The 0.003 gap is not resolved with three seeds.

## Provenance

- Run: `budget_match.py` (24 models: 4 stages × 2 budgets × 3 seeds; `Wv` bias=False).
- Raw output: `budget_match_out.txt` (UTF-16LE with BOM).
- Analysis: `budget_match_analysis.py`; complete analysis output in `budget_match_analysis_out.txt`.
- Recovered JSON: `budget_match_results.json` (via `recover_budget_match_json.py`).
- The original run's final print crashed after all models finished; no model result is missing.
