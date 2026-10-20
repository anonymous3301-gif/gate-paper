# 3 Factor 1: Stability (an initialization-time eigenvalue problem)

The first rung of the ladder changes one thing: the key is normalized before the write, while the step size stays at $\beta=1$. Stage 0 uses the raw key $k_t$; stage 1 uses $\tilde k_t=k_t/\lVert k_t\rVert$. The question is whether the raw-key failure is a long-horizon optimization problem—key norms growing during training—or a failure that is already present at initialization. The diagnostics say the latter. Raw keys with $\beta=1$ violate the one-step stability condition for almost every write at initialization, and normalization removes that violation without changing the key distribution. The stage therefore identifies an initialization-time eigenvalue problem, not a key-norm-growth problem.

This distinction matters for the rest of the ladder. If the failure were caused by norms growing, a longer budget or a smaller learning rate might fix it. If it is an initialization-time eigenvalue violation, the fix has to change the write rule itself. The budget-matched rerun shows the practical consequence: at 3000 steps the raw stage is no longer at chance, but it remains far below the normalized stage. Stability is necessary, but it is not sufficient for recall.

## 3.1 The eigenvalue at initialization

For a raw key, the erase eigenvalue along $k_t$ is

$$
\lambda_t = 1-\beta_t\lVert k_t\rVert^2 .
$$

The raw-key LMS stability condition is $0<\beta_t\lVert k_t\rVert^2<2$. With $\beta_t=1$, this becomes $0<\lVert k_t\rVert^2<2$. At initialization the key distribution is far outside this interval. Table 3.1 summarizes the diagnostic, pooled over 35{,}200 write positions (20 batches of 1{,}760). The mean squared key norm is $10.82$, the median is $9.95$, and the maximum is $35.87$; $99.97\%$ of write positions have $\lVert k_t\rVert^2>2$ (11 of 35{,}200 are at or below $2$, with a minimum of $1.74$). The most negative eigenvalue is $-34.87$, attained at the largest key.

**Table 3.1. Initialization diagnostics for the raw-key write.** The same initial key distribution is used for the raw and normalized models; normalization changes the write, not the key distribution. Values are over all write positions.

| Quantity | Value |
|---|---:|
| Mean squared key norm $E\lVert k\rVert^2$ | $10.82$ |
| Median squared key norm | $9.95$ |
| Maximum squared key norm | $35.87$ |
| Fraction of writes with $\lVert k\rVert^2>2$ | $99.97\%$ |
| Most negative erase eigenvalue $1-\lVert k\rVert^2$ | $-34.87$ |
| First forward loss, raw key | $2.36\times10^{18}$ |
| First forward loss, normalized key | $5.35$ |

The first-forward comparison isolates the write rule. Both models see the same keys; the raw-key model's first loss is eighteen orders of magnitude larger. The normalized-key model's first loss ($5.35$) is still above the uniform-prediction loss ($\ln 64\approx4.16$), because the readout is random, but it is finite and trainable. The raw-key loss is not a training result; it is the immediate consequence of a negative erase eigenvalue at initialization.

For a normalized key, the eigenvalue is

$$
\lambda_t = 1-\beta_t,
$$

independent of $\lVert k_t\rVert^2$. The normalized-key stability condition is $0<\beta_t<2$. At $\beta_t=1$ the write erases the old value in one step (eigenvalue $0$). This is stable, but it is also a strong erase; the next rung varies $\beta$ at fixed normalized keys. The stage 0→1 contrast therefore removes the norm-dependent eigenvalue, but it also changes the effective write magnitude. The ladder is a controlled decomposition, not a claim that the two operations are independent.

## 3.2 The clipped trajectory: freeze, then escape by norm collapse

Under the standard global-norm clipping protocol (clip $=1.0$), the raw-key model does not learn during the early part of training. Table 3.2 shows representative points from the seed-0 trajectory. The full trajectory is in `rawk_log.csv`.

**Table 3.2. Raw-key trajectory under global-norm clipping at 1.0 (seed 0).** The first phase is an unstable trap; the model escapes by shrinking the key norms. The normalized model is shown for comparison at the final step.

| Step | Loss | $E\lVert k\rVert^2$ | $\max\lVert k\rVert^2$ | Gradient norm | Phase |
|---:|---:|---:|---:|---:|---|
| 0 | $2.36\times10^{18}$ | $10.95$ | $35.87$ | $\infty$ | unstable |
| 700 | $2.62\times10^{19}$ | $10.85$ | $35.57$ | $\infty$ | unstable |
| 800 | $7.17\times10^{13}$ | $8.26$ | $28.76$ | $1.22\times10^{16}$ | transition |
| 900 | $4.446$ | $1.64$ | $4.59$ | $5.23$ | escape |
| 1499 | $4.0503$ | $0.990$ | $4.34$ | $2.00$ | low-accuracy plateau |
| 1499 (normalized) | $3.0819$ | $11.719$ | $35.326$ | $2.92$ | trainable |

The mechanism is visible in the table. For roughly the first 800 steps the loss oscillates between $10^{18}$ and $10^{20}$, the total gradient norm is infinite, and the key norms remain at their initialization scale. Clipping with an infinite total norm scales the update by approximately zero, so the parameters do not move. When the gradient becomes finite, the model escapes—not by learning to use the large keys, but by shrinking them. The mean squared key norm falls from $10.95$ at initialization to $0.99$ at the final step, and the maximum falls from $35.87$ to $4.34$. The final loss is $4.0503$, close to the uniform-prediction loss $4.159$, and the final accuracy is $0.0853$. The normalized model keeps the key norms near their initialization scale ($11.7$) and reaches loss $3.0819$ and accuracy $0.1849$ at the same budget.

The escape is therefore a stabilization of the update, not a solution to the memory task. The raw model ends near uniform prediction; it has removed the eigenvalue violation by making the keys small, but it has not learned the associations. In this setup the failure is better described as a property of the raw-key write than as insufficient optimization: the raw-key write is unstable at the scale at which the keys are initialized, and the clipped model's escape route destroys the signal.

**Clip-protocol control.** The trajectory in Table 3.2 is observed under global-norm clipping at $1.0$, so we reran the raw-key model under two further clipping protocols and the normalized model without clipping, with everything else identical (3 seeds each, 1500 steps). Table 3.6 reports the results.

**Table 3.6. Clip-protocol control.** Raw-key and normalized models at $d_{\mathrm{mem}}=16$, 1500 steps, 3 seeds. "Freeze end" is the first logged step at which the loss falls below $10^6$ (logged every 10 steps); "—" means the loss never leaves the pathological regime. The clip $=1.0$ row reproduces Table 3.2 and the budget-matched ladder row exactly ($0.0853/0.0157/0.1722$), which is the protocol check for this table.

| Condition | Final accuracy (seed 0/1/2) | Mean | Freeze end (step) | Final loss | $E\lVert k\rVert^2$ at end | $\max\lVert k\rVert^2$ at end |
|---|---|---:|---|---|---:|---:|
| raw $k$, clip $=1.0$ | 0.0853/0.0157/0.1722 | 0.0910 | 850/970/230 | 4.050/4.168/2.904 | 0.990/0.903/0.886 | 4.340/2.700/4.281 |
| raw $k$, clip $=10.0$ | 0.0159/0.0144/0.0160 | 0.0154 | 850/970/230 | 4.167/4.172/4.175 | 1.132/1.170/1.278 | 2.832/3.283/4.403 |
| raw $k$, no clip | 0.0149/0.0162/0.0163 | 0.0158 | — | $1.59\!\times\!10^{17}$/$2.24\!\times\!10^{16}$/$3.43\!\times\!10^{12}$ | 10.573/10.111/8.167 | 34.924/28.443/24.716 |
| normalized $k$, no clip | 0.0331/0.0784/0.1149 | 0.0754 | — (loss finite from step 0) | 4.133/3.941/3.793 | 11.019/10.828/10.045 | 35.630/32.381/26.526 |

Three things follow, and they change how the frozen phase should be described.

First, **the freeze is not created by the clip value.** While the total gradient norm is infinite, any finite clip scales the update to zero, so the frozen phase is *identical* at clip $=1.0$ and clip $=10.0$: the same escape step (850, 970, and 230 for the three seeds), the same number of logged points above $10^6$ (85, 97, and 23), and the same collapse of the key norms. The freeze is a consequence of the infinite gradient norm that the unstable raw-key write produces, not of the protocol. Note also that the freeze length is strongly seed-dependent (230 to 970 steps), and that the earliest escape (seed 2, step 230) is the seed that ends highest ($0.1722$).

Second, **the escape is not a clipping artifact either, but its payoff is.** Without clipping the model never leaves the pathological regime: every one of the 151 logged points has loss above $10^6$, the final losses are $10^{12}$–$10^{17}$, and the key norms remain at their initialization scale ($E\lVert k\rVert^2=8.2$–$10.6$). No NaN occurs, because AdamW's normalized update stays finite; the run does not crash, it simply never learns. What the clip value changes is whether the model learns *after* the escape: at clip $=1.0$ two of the three seeds end above chance, at clip $=10.0$ none does and all three end at the uniform-prediction loss.

Third, **no clipping setting we tested makes the raw-key write learn.** The raw stage's outcome under clip $=1.0$ ($0.0910$) and under the other two protocols ($0.0154$, $0.0158$) is chance-level or near it, and every one of those seeds is below $0.45$. The clipping protocol therefore determines *how badly* the raw model fails -- a chance-level collapse, or a chance-level norm-collapse escape -- and not *whether* it fails. The initialization-time eigenvalue violation of Section 3.1 and the frozen phase of Section 3.2 are thus not artifacts of the clipping protocol, and the claim we had marked as protocol-specific can now be stated without that qualifier.

One caveat survives, and it is about the other model. The normalized stage is not clip-independent either: without clipping it reaches $0.0754$ (std $0.0410$) instead of $0.1919$ at the same budget, although its loss is finite from the first step and it never enters the frozen regime. The *ordering* -- raw fails, normalized learns -- holds under every clipping setting we tested, but the magnitudes reported in Sections 3.3 and 3.4 are stated under the clip $=1.0$ protocol used for the whole ladder.

The 3000-step run provides final losses and accuracies only; the per-step history was not retained. We therefore do not claim that the same freeze-and-escape trajectory occurs at 3000, only that the final raw stage remains far below the normalized stage (Section 3.3).

### Figure 1

![Figure 1: trajectory, clip control, per-seed spread](fig1_trajectory.png)

**Figure 1. The raw-key trajectory, the clip-protocol control, and the per-seed spread of the stability transition.** (a) Training loss (log scale, seed 0) for the raw-key model under three clipping protocols and for the normalized-key model. The two finite-clip runs coincide *exactly* during the frozen phase -- while the total gradient norm is infinite, any finite clip scales the update to zero -- and both escape by norm collapse at about step 800; without clipping the model never escapes and the loss stays above $10^{12}$. (b) $E\lVert k\rVert^2$ over the same runs: the collapse *is* the escape, and the green line marks the raw-key stability threshold $\beta\lVert k\rVert^2=2$ at $\beta=1$, which the key norms stay above until they collapse. (c) Per-seed accuracy for stages 0 and 1 at 1500 and 3000 steps (three seeds each), with the mean, the sample standard deviation, and the chance level; the raw stage is a mixture at 1500 steps (one seed at chance) and a uniformly low band at 3000, while the normalized stage is tight at 1500 and shifts up at 3000 with one lower seed. Sources: `clip_ablation_log.csv` and `rawk_log.csv` for (a) and (b); `budget_match_complete_summary.md` for (c).

## 3.3 Per-seed results at 1500 and 3000 steps

Table 3.3 reports the per-seed accuracies for the stability transition at both budgets. Table 3.4 reports the final training losses.

**Table 3.3. Per-seed accuracy at the stability transition.** Stage 0 is raw $k$, $\beta=1$; stage 1 is $k/\lVert k\rVert$, $\beta=1$. Chance is $1/64=0.0156$.

| Stage | Budget | Seed 0 | Seed 1 | Seed 2 | Mean | Std | Median |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 raw | 1500 | 0.0853 | 0.0157 | 0.1722 | 0.0911 | 0.0784 | 0.0853 |
| 0 raw | 3000 | 0.2714 | 0.2339 | 0.2638 | 0.2564 | 0.0198 | 0.2638 |
| 1 normalized | 1500 | 0.1849 | 0.1935 | 0.1972 | 0.1919 | 0.0063 | 0.1935 |
| 1 normalized | 3000 | 0.6662 | 0.7041 | 0.5740 | 0.6481 | 0.0669 | 0.6662 |

**Table 3.4. Per-seed final training loss at the stability transition.**

| Stage | Budget | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|---:|
| 0 raw | 1500 | 4.0503 | 4.1678 | 2.9038 |
| 0 raw | 3000 | 2.2975 | 2.3783 | 2.3096 |
| 1 normalized | 1500 | 3.0819 | 2.9058 | 2.8917 |
| 1 normalized | 3000 | 1.2712 | 1.1061 | 1.4212 |

At 1500 steps the raw stage is not a single failure mode. Seed 1 stays at chance ($0.0157$, loss $4.1678$, slightly above $\ln 64$), seed 0 is at $0.0853$ (loss $4.0503$), and seed 2 escapes to $0.1722$ (loss $2.9038$). The mean ($0.0911$) is an average over these outcomes; the median ($0.0853$) is a better summary of the typical seed. The normalized stage is tight ($0.1849$–$0.1972$; std $0.0063$), but its mean is only $0.1919$. Stability raises the floor; it does not solve recall.

At 3000 steps the three raw seeds lie in a narrow band ($0.2339$–$0.2714$; mean $0.2564$, std $0.0198$) rather than showing the 1500 two-regime pattern. The raw stage is no longer at chance, but it remains far below the normalized stage ($0.5740$–$0.7041$; mean $0.6481$). The raw final losses are $2.30$–$2.38$; the normalized final losses are $1.11$–$1.42$. The 3000 raw stage is a low-accuracy regime, not a complete collapse.

Table 3.5 summarizes the paired stability transition. The paired difference is normalized minus raw within each seed, so positive values favor normalization.

**Table 3.5. Paired stability transition.** The 95% CI is a seed-paired interval with $n=3$; it is descriptive, not a population estimate. Per-seed deltas are listed in seed order.

| Budget | Raw mean (std) | Normalized mean (std) | Paired $\Delta$ (95% CI) | Per-seed $\Delta$ |
|---:|---:|---:|---:|---|
| 1500 | 0.0911 (0.0784) | 0.1919 (0.0063) | $+0.1008 \pm 0.1898$ | $+0.0996,\ +0.1778,\ +0.0250$ |
| 3000 | 0.2564 (0.0198) | 0.6481 (0.0669) | $+0.3917 \pm 0.1989$ | $+0.3948,\ +0.4702,\ +0.3102$ |

At 1500 the paired mean is $+0.1008$ with a 95% CI that includes zero. All three per-seed deltas are positive, but the smallest is only $+0.0250$, and the raw seed-to-seed spread is large. The clearest descriptive difference at this budget is variance reduction: the standard deviation falls from $0.0784$ to $0.0063$. With three seeds this is descriptive, not a variance test; we do not use it to claim a population-level variance reduction.

At 3000 the paired mean is $+0.3917$ ($\pm0.1989$), excluding zero. The variance pattern reverses: the raw std is $0.0198$, the normalized std is $0.0669$, and one normalized seed reaches only $0.5740$. In this sample, normalization is descriptively a mean shift at 3000, not a variance reduction. The variance-reduction reading is specific to the 1500-step budget and must not be generalized.

The within-rule budget deltas are consistent with the larger 3000 gap. The raw stage gains $+0.1653$ ($\pm0.1635$) from 1500 to 3000; this interval is close to the zero boundary. The normalized stage gains $+0.4562$ ($\pm0.1747$). Both point estimates improve with budget, but the normalized stage improves more, so the normalization gap grows from $+0.1008$ to $+0.3917$. This is a mean shift, not a claim that the raw stage is frozen at 3000; the 3000 raw stage is low, not at chance.

Three seeds per cell are a real constraint. The per-seed values in Tables 3.3–3.5 are the primary evidence; the means and CIs are summaries, not replacements for the seeds. We do not claim that the 3000 raw stage is at a fixed plateau independent of seed, and we do not claim that the 1500 variance reduction is a stable property of normalization.

## 3.4 What the stability stage establishes

The stability stage supports four claims.

First, the raw-key failure is an initialization-time eigenvalue problem. At initialization, $99.97\%$ of writes violate $0<\beta\lVert k\rVert^2<2$; the most negative eigenvalue is $-34.87$; the first forward loss is $2.36\times10^{18}$ for raw keys and $5.35$ for normalized keys with the same key distribution. This is not a key-norm-growth story: in the 1500 trajectory, the raw model's key norms do not grow; under clipping they stay at the initialization scale and then shrink.

Second, the clipped trajectory has a specific failure mode. Under clip $=1.0$, the raw model freezes for 850, 970, and 230 steps in the three seeds (the gradient norm is infinite, so any finite clip scales the update to zero), then escapes by shrinking the keys; the final loss is $4.0503$ and the final accuracy $0.0853$ for seed 0, i.e. the model is near uniform prediction. The clip-protocol control (Table 3.6) shows that this failure is not a clipping artifact: the freeze and the norm collapse are identical at clip $=10.0$, and with no clipping at all the model never escapes and never learns. The remaining protocol dependence is in the magnitudes: the normalized stage also loses accuracy without clipping ($0.0754$ instead of $0.1919$), so all ladder numbers are stated under clip $=1.0$ while the raw-versus-normalized ordering holds under every setting tested.

Third, normalization is necessary but not sufficient. The normalized stage at $\beta=1$ reaches $0.1919$ at 1500 and $0.6481$ at 3000, far below the next rung (fixed $\beta=0.125$: $0.8643$ and $0.9431$). The next gap in the ladder is erase strength, not a further stability change. At 1500 the erase-strength transition is $+0.6724$ ($\pm0.0447$), much larger than the stability transition; at 3000 it is $+0.2950$ ($\pm0.1587$). Erase strength is the dominant effect at the baseline budget and remains large at 3000. At 3000, however, the stability point estimate is larger ($+0.3917$ vs $+0.2950$) and the intervals overlap, so we do not rank erase strength above stability at 3000. The defensible claim is persistence, not rank.

Fourth, the descriptive effect of normalization is budget-dependent. At 1500 the mean shift is not resolved and the clearest difference is variance reduction; at 3000 the effect is a mean shift and the normalized variance is larger. This is why Section 1 reports both budgets and why the paper does not describe normalization as a universal variance reducer. The result is conditional on the 1500/3000 budgets, the MQAR task, and $d_{\mathrm{mem}}=16$; capacity is varied in Sections 4 and 5.

## 3.5 Summary

The stability rung identifies an initialization-time eigenvalue violation in the raw-key delta rule. Normalization removes the violation and prevents the clipped freeze-and-escape trajectory, but it does not by itself produce recall: at 3000 steps the normalized $\beta=1$ stage is $0.6481$, while the fixed $\beta=0.125$ stage is $0.9431$. The stability effect is a variance reduction at 1500 and a mean shift at 3000; it is not a budget-independent variance property. Section 4 takes the normalized stage as the starting point and isolates erase strength.

---

## Draft notes (not for submission)

- **Clip ablation is complete.** The no-clip / finite-clip control is in `clip_ablation.py` → `clip_ablation_out.txt`, `clip_ablation.json`, `clip_ablation_log.csv` (raw key at clip 1.0 / 10.0 / none, and normalized key at no clip; 3 seeds each, 1500 steps). The clip $=1.0$ row reproduces the Section 3.2 trajectory and the budget-matched ladder row exactly ($0.0853/0.0157/0.1722$), which is the protocol check. Table 3.6 in Section 3.2 is the write-up. The old caveat ("the frozen phase is a clipping-protocol effect; the control is still required") is retracted and replaced by the three conclusions in Section 3.2.
- **Trajectory source.** Table 3.2 is seed 0 from `rawk_log.csv` (`rawk_log.py`, 1500 steps, clip $=1.0$). The freeze length is seed-dependent (230, 850, 970 steps for seeds 2, 0, 1); "roughly 800 steps" is a seed-0 statement and must not be reported as the general freeze duration.
- **Table 3.1 provenance (recomputed).** The initialization diagnostics are now reproduced by `init_eig_diag.py` (raw-key model, seed 0, $d_{\mathrm{mem}}=16$; 20 batches × 1{,}760 write positions = 35{,}200 positions), with output in `init_eig_diag_out.txt` / `init_eig_diag.json` and the pooled key norms in `init_eig_diag_k2.npy`. The original values were computed in an earlier session and had no saved artifact; the rerun reproduces the mean ($10.82$), median ($9.95$), max ($35.87$), most negative eigenvalue ($-34.87$), and the two first-forward losses ($2.36\times10^{18}$, $5.35$) to the reported precision. **One value changed:** the fraction of writes violating the bound is $99.97\%$ (11 of 35{,}200 positions at or below $\lVert k\rVert^2=2$), not the $99.98\%$ used in the earlier drafts. The corrected value is used in Sections 1, 3, and 4. The same script also stores the initialization-time raw-key counterfactual (Table 4.3 / Section 4.4).
- **Data source.** Tables 3.3–3.5 are from `budget_match_results.json` / `budget_match_complete_summary.md` (24 models: 4 stages × 2 budgets × 3 seeds). The 3000 run has final losses only; per-step histories were lost in the print crash. Do not claim the same trajectory at 3000.
- **Trajectory source.** Table 3.2 is seed 0 from `rawk_log.csv` (`rawk_log.py`, 1500 steps, clip $=1.0$).
- **No larger-seed raw-$k$ follow-up.** The raw-$k$ comparison is 3 seeds per budget. Section 1 previously said “larger-seed follow-up”; that was wrong and has been changed to “budget-matched follow-up.”
- **Do not say erase strength is the largest effect at 3000.** The point estimates are stability $+0.3917$ and erase strength $+0.2950$; the intervals overlap. Say erase strength is the main effect at 1500 and remains large at 3000, but do not rank the two at 3000.
- **Do not compare `stage1_norm` ($\beta=1$) with `beta_power` $\mu=1.0$ ($\beta\approx0.098$).** The budget test is `stage1_norm` at 1500 vs 3000.
- **Figure 1 is generated** (`fig1_trajectory.py` -> `fig1_trajectory.png` / `.pdf`, 300 dpi): (a) the loss trajectory for the raw-key model under clip 1.0 / 10.0 / no clip and for the normalized model, (b) the same runs' $E\lVert k\rVert^2$, (c) the per-seed strip for stages 0 and 1 at both budgets. The figure makes the clip independence of the freeze visible (the two finite-clip curves coincide) and shows the norm collapse that is the escape. No new runs: (a) and (b) are replots of `clip_ablation_log.csv` and `rawk_log.csv`, (c) of `budget_match_complete_summary.md`.
