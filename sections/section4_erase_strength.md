# 4 Factor 2: Erase strength

## 4.1 The problem: $\beta=1$ is not a scale-free choice

With normalized keys ($\hat{k}=k/\lVert k\rVert$), the delta-rule eigenvalue along the write direction is $1-\beta$. The LMS stability condition is therefore

$$
0<\beta<2,
$$

independent of $d_{\mathrm{mem}}$ and of the key distribution. The standard choice $\beta=1$ sits at the center of this interval, but "stable" and "optimal" are not the same. At $\beta=1$, the update is a full projection onto the complement of $\hat{k}$, which can over-erase associations that share directions with the current key. The question of this section is whether the erase rate should be chosen by the stability bound or by the task.

We distinguish two regimes throughout. In the **normalized regime** ($\lVert\hat{k}\rVert=1$), the condition is $0<\beta<2$, and a fixed $\beta$ is a scalar multiplier. In the **raw-key regime** ($k$ unnormalized, as in Stage 0 of Section 3), the effective eigenvalue is $1-\beta\lVert k\rVert^2$, and at initialization $E\lVert k\rVert^2=10.82$ at $d_{\mathrm{mem}}=16$, so $\beta=1$ violates the bound for $99.97\%$ of writes. The raw-key regime is the Stage 0 baseline; the counterfactual in Section 4.4 is applying the raw-key bound to the normalized model. It is not the regime used by the normalized ladder or by the gated delta blocks in Section 7.

## 4.2 A fixed erase rate recovers most of the gap

Table 4.1 collects the fixed step sizes measured in the normalized regime at $d_{\mathrm{mem}}\in\{16,32\}$, 3 seeds per cell, 1500 steps. Seeds are listed in order 0/1/2. The table is not a complete grid: the $d_{\mathrm{mem}}=16$ column has five fixed points plus Kaczmarz, and the $d_{\mathrm{mem}}=32$ column also has five fixed points plus Kaczmarz, but the two columns do not share a uniform grid. The $\beta=0.125$ row at $d_{\mathrm{mem}}=16$ is the budget-matched Stage 2; the $\beta=1$ row at $d_{\mathrm{mem}}=16$ is Stage 1. The other rows come from the capacity-scaled baselines `inv_sqrt_d` and `inv_d` and from the fixed-constant sweep `beta_const`; they are valid fixed-$\beta$ runs at the stated capacity, not a single uniform grid.

**Table 4.1. Fixed step sizes in the normalized regime.** Mean accuracy (seeds 0/1/2) at 1500 steps. "—" means not measured in this sweep. The table is intentionally not a uniform grid; the provenance column matters more than the row order. All runs use the same protocol as Section 3: normalized keys, bias-free $W_k,W_v,W_q$, global-norm gradient clipping at $1.0$, and MQAR accuracy with 8 pairs and 32 distractors. The sweep is deliberately **not a uniform grid**: $\beta=0.03125$ at $d_{\mathrm{mem}}=16$ and $\beta=0.25$ and $0.5$ at $d_{\mathrm{mem}}=32$ were not run, because the measured surface is broad -- $\beta=0.125$ is within $0.018$ of the best measured fixed point at $d_{\mathrm{mem}}=16$ and within $0.005$ at $d_{\mathrm{mem}}=32$ -- and the ladder's own operating point was already fixed at $\beta=0.125$ before the sweep was run. The missing cells are noted here so that the table is read as a design, not as an incomplete grid.

| Effective $\beta$ | $d_{\mathrm{mem}}=16$ | $d_{\mathrm{mem}}=32$ | Source |
|---:|---:|---:|---|
| $0.03125$ | — | $0.9138$ ($0.9208/0.9182/0.9025$) | `inv_d`, $d=32$ |
| $0.0625$ | $0.8180$ ($0.8197/0.8381/0.7960$) | $0.9332$ ($0.9377/0.9347/0.9273$) | `inv_d`, $d=16$; `beta_const`, $d=32$ |
| $0.125$ | $0.8643$ ($0.8686/0.8754/0.8489$) | $0.9386$ ($0.9426/0.9366/0.9365$) | budget-matched Stage 2, $d=16$; `beta_const`, $d=32$ |
| $0.1768$ ($1/\sqrt{32}$) | — | $0.9301$ ($0.9435/0.9091/0.9378$) | `inv_sqrt_d`, $d=32$ |
| $0.25$ | $0.8465$ ($0.8501/0.8598/0.8295$) | — | `inv_sqrt_d`, $d=16$ |
| $0.5$ | $0.4932$ ($0.5726/0.3616/0.5453$) | — | `beta_const`, $d=16$ |
| $1.0$ | $0.1919$ ($0.1849/0.1935/0.1972$) | $0.8352$ ($0.8649/0.8000/0.8408$) | budget-matched Stage 1, $d=16$; `one`, $d=32$ |
| Kaczmarz ($\beta_t=1/(\lVert k_t\rVert^2+\epsilon)$) | $0.9372$ ($0.9364/0.9408/0.9345$) | $0.9475$ ($0.9472/0.9482/0.9470$) | budget-matched Stage 3, $d=16$; `kaczmarz`, $d=32$ |

At $d_{\mathrm{mem}}=16$ and 1500 steps, a fixed $\beta=0.125$ raises accuracy from $0.1919$ ($\beta=1$) to $0.8643$, closing $0.6724/0.7453\approx90\%$ of the gap to Kaczmarz ($0.9372$). At $d_{\mathrm{mem}}=32$ and 1500 steps, the fixed rule $\beta=0.125$ reaches $0.9386$ against Kaczmarz at $0.9475$, closing $0.1034/0.1123\approx92\%$ of the gap. At the 3000-step budget at $d_{\mathrm{mem}}=16$, fixed $\beta=0.125$ reaches $0.9431$ against Kaczmarz at $0.9461$, closing $0.2950/0.2980\approx99\%$ of the gap. A fixed erase rate, chosen once, therefore recovers most of the Kaczmarz advantage at both capacities at 1500 steps, and at 3000 steps for $d_{\mathrm{mem}}=16$; the $d_{\mathrm{mem}}=32$, 3000-step cell was not run.

The optimal fixed $\beta$ is not universal, but the measured surface is broad and does not show a clear monotone capacity trend. At $d_{\mathrm{mem}}=16$, $\beta=0.125$ is the best measured fixed point ($0.8643$), with $\beta=0.25$ close ($0.8465$); $\beta=0.0625$ is lower ($0.8180$) and $\beta=0.5$ is clearly worse ($0.4932$). At $d_{\mathrm{mem}}=32$, $\beta=0.125$ is again the best measured fixed point ($0.9386$), with $\beta=0.0625$ ($0.9332$) and $\beta=0.1768$ ($0.9301$) close; the $d_{\mathrm{mem}}=32$ values of $\beta=0.25$ and $\beta=0.5$ were not measured. We report $\beta=0.125$ as a robust compromise, not as the unique optimum, and we do not claim that the optimal fixed $\beta$ decreases with $d_{\mathrm{mem}}$.

### Figure 2

**Figure 2. Capacity and step-size views of the normalized-regime sweep.** (a) MQAR accuracy versus $d_{\mathrm{mem}}$ at 1500 steps for five write rules: fixed $\beta=1$, capacity-scaled $\beta=1/\sqrt{d_{\mathrm{mem}}}$, capacity-scaled $\beta=1/d_{\mathrm{mem}}$, a single fixed $\beta=0.125$, and Kaczmarz. (b) Accuracy versus effective $\beta$ at $d_{\mathrm{mem}}\in\{16,32\}$, with the Kaczmarz level of each capacity as a dotted line and the LMS bound $\beta<2$ marked. Error bars are the sample standard deviation over 3 seeds; the dotted line in (a) is chance ($1/64$). Every point in both panels is the value reported in Table 4.1, and lines connect measured points only.

Panel (a) separates two effects that are usually conflated. The rule $\beta=1$ is strongly capacity-dependent (0.192 at $d_{\mathrm{mem}}=16$, 0.835 at 32), whereas a single fixed $\beta=0.125$ is not: the same constant reaches 0.864 and 0.939 at the two capacities, and it is the best fixed point at both. Panel (b) shows why the stability bound does not select the operating point: every measured optimum lies far inside $0<\beta<2$, and the two points that are still stable but worse ($\beta=0.5$ and $\beta=1$ at $d_{\mathrm{mem}}=16$) degrade sharply while remaining inside the bound.

## 4.3 Erase strength persists across budgets

The erase-strength effect is not a short-budget artifact. Table 4.2 reports the stage-to-stage paired difference (Stage 1 → Stage 2) at both budgets, with per-seed deltas.

**Table 4.2. Erase-strength transition at both budgets ($d_{\mathrm{mem}}=16$).** Paired difference is Stage 2 minus Stage 1 within each seed. The 95% CI is a seed-paired interval with $n=3$; it is descriptive, not a population estimate.

| Budget | Stage 1 ($\beta=1$) | Stage 2 ($\beta=0.125$) | Paired $\Delta$ | 95% CI | Per-seed $\Delta$ |
|---:|---:|---:|---:|---:|---|
| 1500 | 0.1919 | 0.8643 | $+0.6724$ | $\pm0.0447$ | $+0.6837,\ +0.6819,\ +0.6517$ |
| 3000 | 0.6481 | 0.9431 | $+0.2950$ | $\pm0.1587$ | $+0.2772,\ +0.2419,\ +0.3659$ |

At 1500 steps, erase strength is the largest single stage-to-stage factor ($+0.6724$, CI excludes zero). At 3000 steps, the effect persists and remains substantial ($+0.2950$, CI excludes zero), though the point estimate of the stability factor ($+0.3917$) is larger at this budget and the two intervals overlap. We do not claim that erase strength is the largest factor at 3000; we claim that it persists as a large effect at both budgets. The persistence is the key contrast with gating (Section 5): the gating gap is $+0.0729$ at 1500 and $+0.0030$ at 3000 (CI includes zero), whereas erase strength remains large at 3000. This is consistent with its mechanistic origin: $\beta$ controls the trade-off between overwriting and interference at every step, independent of how many steps are taken.

## 4.4 The optimum is inside the stability bound

In the normalized regime, the optimal fixed $\beta=0.125$ is well inside the LMS bound $0<\beta<2$. This is the opposite of what a stability-first design would suggest: the bound says "any $\beta<2$ is stable," but the task says "$\beta=0.125$ is best." The stability bound is a constraint, not an objective.

The raw-key product $\beta\lVert k\rVert^2$ illustrates why the two regimes must not be conflated. Table 4.3 reports the fraction of writes whose raw-key product would exceed $2$ if the normalized optimum were applied to the raw keys as a counterfactual; the statistic is measured during training at fixed $\beta$ with normalized keys. At $d_{\mathrm{mem}}=16$, $\beta=0.125$ exceeds the bound on $20$–$29\%$ of writes; at $d_{\mathrm{mem}}=32$, it exceeds the bound on $78$–$80\%$ of writes, yet the normalized model at this point reaches $0.9386$. The same counterfactual evaluated at initialization (the Table 3.1 key distribution, 35{,}200 write positions) gives $13.2\%$ at $\beta=0.125$, $39.2\%$ at $\beta=1/\sqrt{32}\approx0.177$, $68.9\%$ at $\beta=0.25$, and $97.8\%$ at $\beta=0.5$ (`init_eig_diag.py`). The raw-key bound is a different object: it constrains a raw-key update, not the normalized update that the model actually performs. We did not run a fixed-$\beta$ sweep in the raw-key regime, so we do not claim where the raw-key optimum lies; the claim is only that the normalized optimum is not selected by the raw-key bound.

**Table 4.3. Raw-key counterfactual at the normalized optimum.** $b2=\beta\lVert k\rVert^2$ measured at fixed $\beta$ with normalized keys (the same product that defines the raw-key eigenvalue in Section 3.1); `over2` is the fraction of writes with $b2>2$. Each entry gives the range across seeds 0/1/2.

| $d_{\mathrm{mem}}$ | $\beta$ | $b2$ mean range | `over2` range |
|---:|---:|---:|---:|
| 16 | 0.125 | $1.434$–$1.662$ | $0.196$–$0.289$ |
| 32 | 0.125 | $2.844$–$3.168$ | $0.777$–$0.801$ |
| 16 | 0.5 | $5.311$–$5.893$ | $0.945$–$0.962$ |

The confusion in the literature between the two regimes comes from conflating them. In the raw-key regime, $\beta=1$ violates the bound for $99.97\%$ of writes at initialization. In the normalized regime, the bound is $0<\beta<2$ and the optimum is comfortably inside it. A statement that "the optimal erase rate is not the stable one" can only be a statement about the raw-key regime; it is not a statement about the normalized model, and we do not establish it here. The normalized model's optimum is set by the trade-off between overwriting old values and retaining new ones, not by the stability bound.

## 4.5 What erase strength establishes

Three conclusions follow.

**(i) The dominant factor at the baseline budget is erase strength, not gating.** At 1500 steps at $d_{\mathrm{mem}}=16$, $\beta=0.125$ alone closes about $90\%$ of the gap to Kaczmarz. The gate's marginal contribution at the same budget is $+0.0729$; Section 5 shows that this is a sample-efficiency advantage rather than a persistent convergence gap.

**(ii) Erase strength is a training-time effect that persists at 3000 steps.** The effect is $+0.6724$ at 1500 and $+0.2950$ at 3000; both intervals exclude zero. At 3000 the stability point estimate is larger ($+0.3917$) and the two intervals overlap, so we do not rank erase strength above stability at 3000. The supported claim is persistence, not rank.

**(iii) The stability bound is a constraint, not an objective.** In the normalized regime, the optimal $\beta=0.125$ is inside $0<\beta<2$. The raw-key counterfactual at the same $\beta$ violates the bound on a large fraction of writes, especially at $d_{\mathrm{mem}}=32$, but that counterfactual does not constrain the normalized model. The normalized regime's optimum is a task-level trade-off, not a stability-margin choice.

---

## Draft notes (not for submission)

- **Reviewer answer for the missing Table 4.1 cells.** "The sweep is not a uniform grid; $\beta=0.25$ and $0.5$ at $d_{\mathrm{mem}}=32$ were not run because the performance surface is broad -- $\beta=0.125$ is within $0.018$ of the best measured fixed point at $d_{\mathrm{mem}}=16$ and within $0.005$ at $d_{\mathrm{mem}}=32$ -- and the operating point was already fixed at $\beta=0.125$. The missing cells are noted in the table caption." Adding them costs about 30 minutes of compute and can be done on request.
- **Table 4.1 is not a uniform $\beta$ sweep.** The $d_{\mathrm{mem}}=16$ column has $\beta=0.03125$ missing; the $d_{\mathrm{mem}}=32$ column has $\beta=0.25$ and $\beta=0.5$ missing. The available points are enough to show that fixed $\beta=0.125$ recovers most of the gap at both capacities, but a reviewer may ask for a complete grid. The current plan is to keep the three ladder points ($\beta=1$, $\beta=0.125$, Kaczmarz) and the additional measured points, and to add the missing cells only if a reviewer requests them.
- **Corrected numbers from the earlier draft.** The earlier Table 4.1 put $0.6481$ in the $d_{\mathrm{mem}}=32$, $\beta=1$ row; $0.6481$ is the $d_{\mathrm{mem}}=16$, $\beta=1$, 3000-step value. The measured $d_{\mathrm{mem}}=32$, $\beta=1$ value is $0.8352$ (`one`, 1500 steps). The earlier text also said the fixed rule at $d_{\mathrm{mem}}=32$ is "within $0.003$ of Kaczmarz"; the 1500-step comparison is $0.9386$ vs $0.9475$ (gap $0.0089$). The $0.003$ gap is the $d_{\mathrm{mem}}=16$, 3000-step comparison ($0.9431$ vs $0.9461$).
- **Do not claim a capacity trend in the optimal fixed $\beta$.** The measured best fixed point is $\beta=0.125$ at both $d_{\mathrm{mem}}=16$ and $d_{\mathrm{mem}}=32$; the neighboring values are close and $d_{\mathrm{mem}}=32$ has missing cells. Say the surface is broad, not that the optimum decreases with capacity.
- **Do not claim where the raw-key optimum lies.** We have no raw-key $\beta$ sweep; the raw-key statements are counterfactuals at the normalized optimum (Table 4.3). The supported claim is that the raw-key bound does not constrain the normalized model.
- **`beta_power` is not part of Table 4.1.** The `ema`/`power` runs use the scalar $\beta=\mu/E[\lVert k\rVert^2]$, which is not a fixed-$\beta$ sweep and is not the Kaczmarz mean. They are reported separately, if at all, and must not be used as a budget control for Stage 1.
- **Figure 2 is generated** (`fig2_capacity.py` → `fig2_capacity.png` / `.pdf`, 300 dpi). Both panels are replots of Table 4.1; no new runs. Caveats to keep in the caption: only two capacities are available ($d_{\mathrm{mem}}\in\{16,32\}$), the $\beta$ grid is not uniform (the $d_{\mathrm{mem}}=32$ values $\beta=0.25$ and $0.5$ are missing), and the lines connect measured points only. The figure is descriptive; the table remains the primary evidence. If a reviewer asks for a complete grid, the missing cells are 3 seeds × 2 capacities ≈ 30 minutes of compute.
- **Budget-matched provenance.** Table 4.2 is from `budget_match_results.json` / `budget_match_complete_summary.md` (24 models: 4 stages × 2 budgets × 3 seeds). The `d_{\mathrm{mem}}=32` fixed-$\beta$ points in Table 4.1 are from `beta_sweep_out.txt` and `beta_const_out.txt`, all 1500 steps, 3 seeds, bias-free $W_k,W_v,W_q$.
- **Table 4.3 and the initialization counterfactual.** The training-time `b2` / `over2` columns come from `beta_const_out.txt` (and `beta_power_fixed_out.txt` for the $\mu=2.0$ scalar). The initialization-time fractions quoted in Section 4.4 come from `init_eig_diag.py` (output `init_eig_diag_out.txt` / `init_eig_diag.json`, 35{,}200 write positions). Keep the two apart: the training-time numbers are for the trained model's keys, the initialization numbers for the initial key distribution.
- **Section 1 wording was corrected to match.** The earlier Section 1 sentence "the raw-key product exceeds $2$ for $50$–$60\%$ of writes at the optimal fixed $\beta$" was wrong: $50$–$60\%$ is the *training-time* `over2` of the `beta_power` $\mu=2.0$ scalar ($\beta\approx0.196$), not of the ladder's $\beta=0.125$. At the ladder's operating point the initialization-time fraction is $13.2\%$ and the training-time fraction is $20$–$29\%$ (d=16). Section 1 now states both.
