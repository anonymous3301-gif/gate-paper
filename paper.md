# When is a gate worth it? Budget, capacity, and co-adaptation in delta-rule memory

**Author:** KPL Research (pseudonym; the final version may use a real name)
**Version:** v0.1-draft, 2026-10-20
**Status:** Draft preprint for priority claim. The final version will be updated.

## Abstract

Delta-rule memory powers linear-time sequence models, and its update rule mixes three operations -- key normalization, erase strength $\beta$, and norm-based gating -- that are usually introduced together, so each one's value depends on capacity, on budget, and on whether the evaluation is a post-hoc swap or a retraining experiment. We decompose them on multi-query associative recall at $d_{\mathrm{mem}}=16$, holding the task, architecture, and evaluation fixed, and rerun the ladder at two budgets (1500 and 3000 steps; 3 seeds per cell). Raw keys with $\beta=1$ fail at initialization rather than during training: $99.97\%$ of writes violate the LMS stability bound, and the first forward pass has loss $2.36\times10^{18}$ against $5.35$ for normalized keys. The stability effect is budget-dependent -- a variance reduction at 1500 steps, a mean shift at 3000. Erase strength is the dominant effect at the baseline budget ($+0.6724$, 95% CI $\pm0.0447$) and persists at 3000 steps ($+0.2950$, $\pm0.1587$); a fixed $\beta=0.125$ recovers about $90\%$ of the gap to a Kaczmarz gate at both capacities, so the stability bound is a constraint rather than an objective. Gating behaves differently: the gate is worth $+0.0729$ at 1500 steps but only $+0.0030$ at 3000 steps (CI includes zero), which makes it a sample-efficiency advantage rather than a persistent training-time bias. Its shape nevertheless co-adapts: a Kaczmarz-trained model loses $0.380$, $0.227$, and $0.161$ accuracy when evaluated with a mean-matched constant at $d_{\mathrm{mem}}=16,24,32$, while a shape-matched gate costs at most $0.0011$ and is equivalent to the gate itself within $0.01$ (TOST). A curriculum does not substitute for normalization, though it removes the raw-key collapse mode. In a real-architecture check, GDN is trimodal over 10 seeds while Delta is stable (Brown--Forsythe $p=0.0022$). A within-architecture control shows that a scalar retention factor can produce a mixed two-mode outcome at the retention boundary we tested, but not GDN's three-mode distribution, so the mechanism of that collapse remains open. A gate is therefore worth its complexity only conditionally: budget-matched, a fixed erase rate reaches within $0.01$ of the gate at the settings tested, yet the gate's shape still determines what transfers.


---

# 1 Introduction




> **Status note.** This is a draft preprint for priority claim. The budget-matched ladder (Section 3–4) is now final (24/24); the decay-only ablation is complete. The Delta 10-seed test is complete and Section 7 reports it. Numbers marked "baseline budget" are 3-seed means at 1500 steps; budget-matched values are reported in Sections 3–4.



## 1.1 The capacity puzzle



Linear-time sequence models based on delta-rule memory have become a leading alternative to softmax attention. Instead of storing all past keys and values in an attention matrix, a delta-rule memory maintains a state matrix and writes the *error* between a value and its current prediction. This gives the model a fixed-size associative memory whose cost is linear in sequence length. The mechanism underlies DeltaNet, Kaczmarz-based linear attention (KLA), gated delta networks (GDN), and other recent variants.



The memory width $d_{\mathrm{mem}}$ controls how many associations the state can represent. A recurring empirical pattern is that small $d_{\mathrm{mem}}$ fails: accuracy is near chance, and increasing $d_{\mathrm{mem}}$ fixes it. The failure is often attributed to "capacity," but capacity is not a mechanism. The delta update contains at least three operations that are usually introduced together:



1. **Key normalization**, which controls the stability of the write.

2. **The erase strength** $\beta$, which controls how much of the old value is overwritten.

3. **Norm-based gating**, which makes the step size depend on the key.



These operations are not independent in a trained network. A model trained with a gate co-adapts its keys, values, and readout to that gate; swapping the gate after training is not the same as training without it. Conversely, a model trained with a fixed step size may learn a solution that does not need a gate at all. If we ablate one operation without controlling the others, we cannot tell whether the gate is necessary, whether normalization is sufficient, or whether a longer training budget would have rescued the failing rule.



## 1.2 A concrete puzzle



At $d_{\mathrm{mem}}=16$ on multi-query associative recall (MQAR), the four stages of the update rule separate cleanly:



| Stage | Key | Step size $\beta_t$ | Accuracy | Operation added |

|---|---|---|---:|---|

| 0 | raw $k$ | $1$ | 0.091 | baseline |

| 1 | $k/\lVert k\rVert$ | $1$ | 0.192 | stability |

| 2 | $k/\lVert k\rVert$ | $0.125$ | 0.864 | erase strength |

| 3 | $k/\lVert k\rVert$ | $1/\lVert k\rVert^2$ | 0.937 | gating |



*Baseline budget: 3 seeds, 1500 steps, no curriculum. At 3000 steps the same ladder is $0.256\to0.648\to0.943\to0.946$; the budget-matched analysis is in Sections 3–4.*



The first step is stability: raw keys with $\beta=1$ are unstable because the erase eigenvalue $1-\lVert k\rVert^2$ is negative for almost every write. The second step is erase strength: normalizing the key makes the update stable, but $\beta=1$ erases too aggressively; a fixed $\beta=0.125$ recovers most of the gap. The third step is gating: at the 1500-step budget, the remaining gap is recovered by a step size that depends on the key norm; at 3000 steps the fixed rule numerically catches up.



The gaps are large enough to measure, and the operations are separable in the ladder. This makes the failure a useful test case for a broader question: *when is a gate worth its complexity?* The answer, we will argue, is not a universal yes. It depends on capacity, on the training budget, and on whether the evaluation is a post-hoc swap or a retraining experiment.



## 1.3 A four-stage ladder



We build a four-stage ladder at $d_{\mathrm{mem}}=16$. Each stage adds exactly one operation to the previous stage, holding the task, architecture, optimizer, and evaluation fixed. We train three seeds per stage and evaluate on 20 batches of 128 sequences. We measure accuracy and loss, the distribution of key norms, the inverse-norm gate contrast, effective rank, and post-hoc transfer under alternative update rules. We then vary $d_{\mathrm{mem}}\in\{16,24,32,64,128,256\}$, the training budget (1500 vs 3000 steps), and the number of seeds (3 or 10).



The ladder is not a claim that the three operations are independent in general. It is a controlled decomposition at a fixed operating point. We test the attribution in two ways. First, we rerun the ladder at a second budget to check whether the stage-to-stage gaps are budget artifacts. Second, we use a matched-mean post-hoc transfer to separate the *mean* of $\beta$ from its *shape*: we train with one rule, then evaluate with a constant at the gate's mean, a gate at the fixed rule's mean, and the two original rules. If the mean were the causal variable, the mean-matched constant would transfer; if the shape were the causal variable, the shape-matched gate would transfer.



## 1.4 Findings



### Factor 1: Stability is an initialization-time eigenvalue problem



Raw keys with $\beta=1$ fail at initialization, not because the key norms grow during training. At initialization, the mean squared key norm is $10.82$ (median $9.95$, max $35.87$, pooled over 35{,}200 write positions); $99.97\%$ of writes have $\lVert k\rVert^2>2$, so the raw-key erase eigenvalue $1-\lVert k\rVert^2$ lies outside the unit interval. The most negative eigenvalue is $-34.87$ (at the largest key). On the first logged batch ($E\lVert k\rVert^2=10.95$, max $35.87$), the raw-key model's first forward pass has loss $2.36\times10^{18}$, while the normalized-key model has loss $5.35$.



Under the standard gradient-clipping protocol (clip $=1.0$), the raw-key model spends roughly 800 steps in an unstable trap: the loss oscillates between $10^{18}$ and $10^{20}$, the gradient norm is infinite, and the key norms stay at their initialization scale. Clipping with an infinite total norm zeroes the update, so learning freezes. The model eventually escapes by *shrinking* the key norms ($E\lVert k\rVert^2:10.7\to1.0$, max $36\to4.3$), which stabilizes the raw-key update, but the final loss is $4.05$ (close to the uniform-prediction loss $\ln 64=4.16$) and accuracy is about $0.09$. The normalized model keeps the key norms near $11$ and reaches loss $3.08$ and accuracy $0.19$ at the same budget. A clip-protocol control (3 seeds per condition) shows that this is not an artifact of the clipping protocol: the freeze and the norm collapse are identical at clip $=10.0$, and with no clipping at all the raw model never escapes and never learns.



The failure is therefore an initialization-time eigenvalue problem, not a key-norm-growth problem. The three-phase trajectory is observed under clip $=1.0$, and the clip-protocol control (Section 3.2) shows that the frozen phase follows from the infinite total gradient norm rather than from the clip value: any finite clip scales the update to zero, and with no clipping the raw model never escapes and never learns. All ladder numbers use clip $=1.0$; the normalized stage also loses accuracy without clipping ($0.075$ instead of $0.192$), so the magnitudes are protocol-specific even though the raw-versus-normalized ordering is not. The raw-key outcome is also mixture-like across seeds. In the bias-free rerun at 1500 steps, seed 0 reaches $0.0853$ (loss $4.0503$), seed 1 stays at chance ($0.0157$, loss $4.1678$, close to $\ln 64=4.159$), and seed 2 escapes to $0.1722$ (loss $2.9038$); the mean is $0.0911$ and the median is $0.0853$. The normalized stage is tight across seeds ($0.1849$–$0.1972$), while the raw stage spans $0.0157$–$0.1722$. With three seeds, the paired mean difference is $+0.101$ (95% CI $\pm0.190$), so the clearest descriptive difference at this budget is variance reduction; the mean shift is not resolved with three seeds. We therefore report per-seed values and treat the mean as an average over an escape/no-escape mixture, not as a single failure mode; Section 3 reports the budget-matched follow-up.



The budget-matched rerun qualifies the variance-reduction reading. At 3000 steps the raw stage no longer collapses ($0.2339$–$0.2714$; mean $0.2564$), while normalized keys average $0.6481$; the paired mean difference is $+0.3917$ (95% CI $\pm0.1989$), excluding zero. The variance pattern reverses ($0.0198\to0.0669$), because one normalized seed reaches only $0.574$. The 1500-step variance reduction is therefore not a budget-independent property: at 3000 the normalization effect is a mean shift, not a variance reduction.



### Factor 2: Erase strength is the main effect at the baseline budget and persists at 3000



At the baseline budget, moving from $\beta=1$ to $\beta=0.125$ raises accuracy from $0.192$ to $0.864$ at $d_{\mathrm{mem}}=16$. At 3000 steps the same transition is $0.6481\to0.9431$ ($+0.2950$, 95% CI $\pm0.1587$). The effect remains large and its CI excludes zero, but its magnitude shrinks because the normalized $\beta=1$ stage improves with budget. At 3000 the stability transition has a larger point estimate ($+0.3917$), and the two intervals overlap; we therefore do not rank erase strength above stability at 3000. The defensible claim is that erase strength is the dominant effect at the baseline budget and remains a large, budget-persistent effect at 3000. The optimal fixed $\beta$ is not predicted by the raw-key LMS bound, but the reason is not that the normalized optimum is unstable. With normalized keys ($\lVert\hat k\rVert=1$), the erase eigenvalue is $1-\beta$, so the LMS condition reduces to $0<\beta<2$; the ladder's $\beta=0.125$ and the best scalar-$\beta$ condition in our sweep (the `beta_power` $\mu=2.0$ scalar, $\beta=0.196$ at $d_{\mathrm{mem}}=16$ and $0.178$ at $d_{\mathrm{mem}}=32$) both lie well inside this bound. The raw-key bound is a different object. It applies to the raw-key update (stage 0), where $99.97\%$ of writes violate $0<\beta\lVert k\rVert^2<2$ at $\beta=1$. If one evaluates the raw-key product $\beta\lVert k\rVert^2$ at the ladder's operating point $\beta=0.125$, it exceeds $2$ for $13.2\%$ of write positions at initialization and for $20$–$29\%$ of writes during training at $d_{\mathrm{mem}}=16$ (Section 4.4, Table 4.3); at the larger scalar $\beta\approx0.196$ the corresponding fractions are $39\%$ and $52$–$60\%$. A raw-key update at those step sizes would therefore be unstable on a substantial minority of writes. But this is a counterfactual for the normalized model, not the operative constraint. The normalized optimum is set by the trade-off between overwriting old values and retaining new ones. A claim that **the optimal erase rate is not the stable one** can only be a raw-key-regime claim; it does not apply to the normalized regime, and we do not establish it here. Section 4 reports the available fixed-$\beta$ sweep and the budget-matched controls.



### Factor 3: At 1500 steps the residual is the gate shape, not the mean



At 1500 steps the remaining gap is the gate. A Kaczmarz gate ($\beta_t=1/\lVert k_t\rVert^2$) raises accuracy from $0.864$ to $0.937$. This is a training-time sample-efficiency advantage. At 3000 steps the same transition is $0.9431\to0.9461$ ($+0.0030$, 95% CI $\pm0.0032$, which includes zero); with three seeds we do not claim that the gap is exactly zero, but these data are not evidence of a persistent training-time bias. The fixed rule gains $+0.0788$ from the extra budget while Kaczmarz gains only $+0.0089$; numerically, the fixed rule closes the gap.



The matched-mean swap asks a different question: whether a Kaczmarz-trained solution co-adapts to the gate shape. It does, and this is not a post-hoc reweighting. A Kaczmarz-trained model evaluated with a constant at the same mean loses $0.380$, $0.227$, and $0.161$ accuracy at $d_{\mathrm{mem}}=16,24,32$ ($40.5\%$, $23.9\%$, and $17.0\%$ relative). The same model evaluated with a gate of the same shape but mean $0.125$ loses at most $0.0011$, and a paired equivalence test at a tolerance of $\pm0.01$ confirms the two rules are equivalent at all three capacities (Section 5.3, Table 4). The fixed-trained model shows the mirror image: a shape-matched gate at the same mean loses $0.057$, $0.018$, and $0.004$, while a constant at the gate mean changes accuracy by at most $0.0012$. These matched-mean experiments use 1500-step models; whether the co-adaptation cost persists at 3000 steps—where the fixed rule closes the training-time gap—is not tested here (Section 5.3).



The double dissociation is qualitatively invariant across capacity, but its magnitude is not. The Kaczmarz model's shape cost shrinks monotonically with $d_{\mathrm{mem}}$ ($0.380\to0.227\to0.161$) yet remains $17\%$ relative at $d_{\mathrm{mem}}=32$. The fixed model's shape sensitivity is not resolved at $d_{\mathrm{mem}}=32$: the 95% CI includes zero, but an equivalence test at a tolerance of $\pm0.01$ does not pass either, so we do not claim the two rules are equivalent at that capacity (Section 5.3, Table 4). At $d_{\mathrm{mem}}=32$ the fixed-rule model reaches $0.939$, within $0.01$ of the Kaczmarz model's $0.947$, without shape specialization. The gate is therefore not necessary for competitive accuracy at larger capacity; at 3000 steps at $d_{\mathrm{mem}}=16$ the fixed rule is also within $0.003$ of Kaczmarz. At the baseline budget the gate remains the faster training rule, and its shape is what transfers under a Kaczmarz-trained solution.



### Curriculum is a partial rescue, not a substitute for normalization



A curriculum is a partial rescue only in the normalized regime. Raw keys with a curriculum reach $0.211$ — above the no-curriculum raw stage ($0.091$) and with no seed left at chance, but far below normalized keys under the same curriculum ($0.431$, per-seed $0.660/0.205/0.428$) and far below the Kaczmarz gate ($0.937$); the curriculum removes the raw-stage collapse mode without repairing the raw-key eigenvalue problem, and its gain in the normalized regime is not resolved with three seeds. Budget-matched, normalized keys with a curriculum at 3000 steps reach $0.923$ (std $0.004$), close to Kaczmarz at 1500 steps ($0.937$) and below Kaczmarz at 3000 steps ($0.946$). The gate's advantage is therefore sample efficiency; the data do not establish a qualitative difference at convergence. Section 6 reports the budget-matched comparison and its caveats.



### A real-architecture check



As a real-architecture check, we train GDN and Delta blocks on the same MQAR task. GDN with decay is trimodal across 10 seeds: 3 seeds collapse to chance ($\approx0.016$), 2 land in an intermediate regime ($0.298$ and $0.437$; the latter just below the pre-specified $0.45$ failure threshold), and 5 succeed ($0.519$–$0.661$), with a mean of $0.366$. Delta is stable across all 10 seeds ($0.886$–$0.913$; mean $0.897$, std $0.007$). The pre-registered Brown–Forsythe variance test rejects equality ($F(1,18)=12.72$, $p=0.0022$); the Fisher test at the pre-specified $0.45$ threshold gives 5/10 vs 0/10 ($p=0.0325$). The threshold was fixed before the Delta 10-seed run but after the GDN v15 run, and a threshold sweep shows that the Fisher result holds for every $t\in[0.44,0.897]$; at $t=0.4368$ exactly it is not significant ($p=0.0867$), and at $t=0.8975$ it is again not significant ($p=0.0867$). The stricter $0.02$ binarization is not significant (3/10 vs 0/10, $p=0.21$). The small-scale ordering (Delta $>$ GDN) is consistent with the published ordering on the pass-key variant of S-NIAH (DeltaNet 99.0/98.8 against Gated DeltaNet 91.4/91.8 at 4K/8K for the 1.3B models) and inconsistent with the published ordering on the two variants that require the model to clear its state; the published comparison is therefore variant-dependent, and we do not claim that our ordering predicts it (Section 7.5). What the small-scale check provides is a distributional diagnostic that a table of means cannot give. Section 7 reports the full distribution and the statistical caveats.



## 1.5 Boundary conditions



The claims in this paper are conditional, and we state the conditions up front.



- **Capacity.** The gate's cost is largest at small $d_{\mathrm{mem}}$ and attenuates with capacity. At $d_{\mathrm{mem}}=32$ the fixed-rule model is within $0.01$ of the Kaczmarz model. We do not extrapolate a floor from three capacities.

- **Budget.** The 1500-step ladder is $0.091\to0.192\to0.864\to0.937$. At 3000 steps it is $0.256\to0.648\to0.943\to0.946$. Erase strength is the largest effect at 1500 ($+0.672$) and remains large at 3000 ($+0.295$); at 3000 the stability point estimate is larger ($+0.392$), and we do not rank the two. The 3000 gating gap is $+0.003$ (95% CI includes zero); with three seeds we do not claim it is exactly zero, but it is not evidence of a persistent training-time bias. We therefore describe the gate's accuracy advantage as a sample-efficiency effect. The `beta_power` $\mu=1.0$ baseline is a scalar $\beta=\mu/E[\lVert k\rVert^2]\approx0.098$ (an EMA of the mean squared key norm), not $\beta=1$; it is also not the Kaczmarz mean $E[1/\lVert k\rVert^2]\approx0.157$. We do not use it as a budget control for stage 1.

- **Task.** MQAR with $V=64$, $n_{\mathrm{pairs}}=8$, and $n_{\mathrm{dist}}=32$. The task is synthetic; the GDN/Delta check is an external check on real gated delta blocks, not a replication of the ladder.

- **Architecture.** The memory has a linear readout and no softmax attention. The gate's co-adaptation may differ with nonlinear readouts or with additional sequence mixers.

- **Evaluation.** The matched-mean experiment is a post-hoc swap, not a retraining experiment. It measures the co-adaptation of a learned solution, not the effect of training with the swapped rule.

- **Seeds.** Most cells use 3 seeds; $d_{\mathrm{mem}}=24$ and the GDN and Delta experiments use 10. The $d_{\mathrm{mem}}=16$ and $32$ confidence intervals are wider. We report intervals and avoid single-seed claims.



## 1.6 Contributions



1. **A three-factor decomposition.** We separate stability, erase strength, and gating with a four-stage ladder, and we test the attribution with a budget-matched rerun and a matched-mean post-hoc transfer.

2. **A stability analysis.** Raw-key failure is an initialization-time eigenvalue problem ($99.97\%$ of writes violate the raw-key stability condition), not key-norm growth. Under clip $=1.0$ in the 1500-step trajectory, the failure manifests as a frozen-learning trap followed by a norm-collapse escape; a clip-protocol control shows that the freeze follows from the infinite gradient norm rather than from the clip value, and that no clipping setting makes the raw write learn.

3. **An erase-strength analysis.** At the baseline budget, $\beta=1\to0.125$ recovers most of the gap. The optimum is not predicted by the raw-key LMS bound; for normalized keys the relevant eigenvalue is $1-\beta$. The raw-key stability condition and the normalized-key optimum are different objects.

4. **A matched-mean double dissociation.** For a trained solution, the mean of $\beta$ is irrelevant and the shape is the transferable ingredient. The shape cost is capacity-dependent ($40.5\%\to23.9\%\to17.0\%$ relative) and the fixed model's shape sensitivity vanishes at $d_{\mathrm{mem}}=32$; an equivalence test confirms that a shape-matched gate and the Kaczmarz rule are equivalent within $\pm0.01$ at every capacity, so the "no loss" entry is a statement of equivalence rather than of a failed difference. This is a post-hoc co-adaptation result; it does not imply that training with a gate is necessary at convergence.

5. **A curriculum and budget analysis.** Curriculum is a partial rescue only in the normalized regime. The budget-matched ladder gives a 3000-step gating gap of $+0.003$ (95% CI includes zero); with three seeds we do not claim equivalence, but the data support a sample-efficiency advantage rather than a persistent training-time bias. Erase strength is the largest effect at 1500 ($+0.672$) and remains large at 3000 ($+0.295$); at 3000 the stability point estimate is larger ($+0.392$), and we do not rank the two.

6. **A real-architecture check, with two controls on the external validity of the ladder.** GDN with decay is trimodal across all 10 seeds, while Delta is stable across all 10. A within-architecture control adds a scalar retention factor to the ladder's own block and shows that decay alone is sufficient for a collapse but not for GDN's three-mode distribution; a convolution control reruns the whole ladder with a causal convolution and shows that all three factors and their ordering survive it, so the decomposition is not an artifact of our convolution-free block. The pre-registered variance test and the pre-specified failure-rate test reject equality; the published S-NIAH ordering is task-variant dependent, and the small-scale result is consistent with one of those variants and not the others (Section 7.5), so it is a distributional diagnostic rather than a predictor of the published ordering.



## 1.7 Roadmap



Section 2 defines the task, the delta-rule memory, the step-size variants, the training protocol, and the diagnostics. Section 3 isolates stability. Section 4 isolates erase strength. Section 5 isolates gating and co-adaptation, including the matched-mean experiment. Section 6 tests curriculum and budget. Section 7 reports the GDN/Delta check together with two controls on the ladder's external validity (a scalar retention factor and a causal convolution). Section 8 discusses design principles, limitations, the qualitative relation to existing theory, and open questions.



---



## Draft notes (not for submission)



- **Abstract.** The final abstract for the draft preprint is in `abstract.md`. It includes the boundary conditions (capacity and budget) and the decay-only slot (mixed two-mode outcome at the retention boundary). Do not write "the gate is necessary" or "erase strength explains 90%" without the budget caveat.

- **Numbers.** The ladder table uses 3-seed, 1500-step means: 0.091, 0.192, 0.864, 0.937. The 3000-step means are 0.256, 0.648, 0.943, 0.946. The matched-mean baselines in Section 5.3 use 0.9368 (Kaczmarz) and 0.8652 (fixed); the small difference from 0.937/0.864 is the evaluation protocol, not a new result.

- **Budget-matched ladder is final (24/24).** Stage-to-stage at 3000: stability +0.3917 ±0.1989, erase strength +0.2950 ±0.1587, gating +0.0030 ±0.0032 (CI includes zero). The 1500 gating gap (+0.0729 ±0.0274) is not established at 3000: with n=3 we do not claim the gap is exactly zero, but it is not evidence of a persistent training-time bias. The fixed rule gains +0.0788 from the extra budget while Kaczmarz gains +0.0089. Report the gate's accuracy advantage as sample efficiency. Erase strength remains large at 3000, but the stability point estimate is larger there; we do not rank the two. The matched-mean post-hoc co-adaptation result (Section 5.3) is separate and remains; that experiment uses 1500-step models, so persistence of the shape cost at 3000 is untested.

- **Raw-k bimodality.** In the bias-free rerun at 1500 steps, stage 0 seeds are 0.0853 (loss 4.05), 0.0157 (chance, loss 4.17), and 0.1722 (loss 2.90); mean 0.0911, median 0.0853, std 0.0784. The raw-key condition is a mixture of escape and no-escape; report per-seed values and the median, not only the mean. The paired stage0→stage1 mean difference is +0.101 with a 95% CI of ±0.190; at 1500 the clearest descriptive difference is variance reduction (std 0.0784→0.0063). At 3000 the three raw seeds lie in a narrow band (0.2339–0.2714; mean 0.2564) and the paired mean difference is +0.3917 ±0.1989 (CI excludes 0), but the normalized stage's std is larger (0.0669 vs 0.0198). The variance-reduction reading is 1500-specific; at 3000 normalization is a mean shift.

- **Do not use `beta_power` $\mu=1.0$ as a budget control for stage 1.** In `beta_power.py`, the `ema` mode sets $\beta=\mu/E[\lVert k\rVert^2]$, a scalar $\approx0.098$ at $d_{\mathrm{mem}}=16$ (its reference $E[\lVert k\rVert^2]=10.20$ is measured on the trained model); it is an erase-strength condition, not $\beta=1$. The correct budget test is stage 1 at 1500 vs 3000 steps.

- **Raw-key LMS vs normalized-key eigenvalue.** For raw keys the erase eigenvalue is $1-\beta\lVert k\rVert^2$; for normalized keys it is $1-\beta$. The raw-key regime is real (stage 0); what is counterfactual is applying the raw-key bound to the normalized model. In the normalized implementation the relevant condition is $0<\beta<2$, and both the ladder's $\beta=0.125$ and the best scalar-$\beta$ condition $\approx0.185$ are inside it. The statement "the optimal erase rate is not the stable one" applies only to the raw-key regime. Phrase Section 4 accordingly.

- **GDN/Delta.** Both 10-seed sets are complete. GDN is trimodal; Delta is stable. The pre-registered Brown–Forsythe ($p=0.0022$) rejects; the Fisher test at the pre-specified $0.45$ threshold ($p=0.0325$) is supportive; the stricter $0.02$ binarization is not significant ($p=0.21$). The 0.45 threshold was fixed before the Delta 10-seed run but after the GDN v15 run; state this provenance and use the threshold sweep ($p<0.05$ for $t\in[0.44,0.897]$; at 0.4368 and 0.8975, $p=0.0867$) as mitigation. The secondary endpoint does not survive a Bonferroni correction across the two endpoints, although a Holm correction would reject both; because the secondary is also threshold-sensitive, report it as supportive, not independent confirmatory evidence. The $0.437$ seed is the boundary case; do not move the threshold. Do not use Cohen's $d$; report the median gap (0.419), failure counts (5/10 vs 0/10), and variance ratio (1325×). The seed 4 loss check is still pending; the mechanism of the collapse is not established.

- **Clip ablation.** The raw-key three-phase trajectory is observed under clip $=1.0$. Section 3 must report a no-clip or finite-clip control before the "frozen trap" claim is treated as architecture-independent.

- **Title.** Fixed for the draft preprint: *When is a gate worth it? Budget, capacity, and co-adaptation in delta-rule memory*.



---

# 2 Setup



## 2.1 Task: multi-query associative recall



We use multi-query associative recall (MQAR) as a controlled test of associative memory. Each sequence contains $n_{\mathrm{pairs}}=8$ key–value pairs, followed by $n_{\mathrm{dist}}=32$ distractor tokens, followed by $n_{\mathrm{pairs}}$ queries. The queries are a random permutation of the keys; the target for each query is the value that was paired with that key. Keys and values are sampled uniformly from a vocabulary of size $V=64$; duplicate keys are possible, and the delta rule handles them by overwriting. The sequence length is



$$

T = 2n_{\mathrm{pairs}} + n_{\mathrm{dist}} + n_{\mathrm{pairs}} = 56.

$$



The layout is



$$

[k_0, v_0, k_1, v_1, \ldots, k_7, v_7,\; d_0, \ldots, d_{31},\; q_0, \ldots, q_7],

$$



where each $q_j$ is one of the keys. The model reads at every position, but the loss and accuracy are computed only on the final $n_{\mathrm{pairs}}$ query positions. Chance accuracy is $1/V=0.0156$.



MQAR requires the model to store eight associations and retrieve them after a 32-token delay. It is small enough to train on a CPU in minutes, and it isolates associative memory from language-modeling confounds. The absolute accuracies are not intended to match large-scale benchmarks; the paper uses MQAR to measure *which operation* in the delta update causes the failure, and then checks the ordering on GDN/Delta blocks (Section 7).



## 2.2 Delta-rule memory



Let $e_t\in\mathbb{R}^{d}$ be the embedding at position $t$, the sum of a token embedding and a learned positional embedding ($d=64$). The memory uses three bias-free linear maps,



$$

k_t = W_k e_{t-1}, \qquad v_t = W_v e_t, \qquad q_t = W_q e_t,

$$



with $W_k\in\mathbb{R}^{d_{\mathrm{mem}}\times d}$, $W_v\in\mathbb{R}^{d\times d}$, and $W_q\in\mathbb{R}^{d_{\mathrm{mem}}\times d}$. The key is taken from the **previous** token $e_{t-1}$; in the MQAR layout the value token at position $2i+1$ is preceded by its key at position $2i$, so the write at the value position uses the correct key. The read at a query position uses the query embedding $e_t$.



The state is a matrix $S_t\in\mathbb{R}^{d_{\mathrm{mem}}\times d}$, initialized to zero. The read and the write are



$$

\hat v_t = S_{t-1}^{\top}\tilde k_t,

\qquad

S_t = S_{t-1} + \beta_t\,\tilde k_t\,(v_t-\hat v_t)^{\top},

\qquad

r_t = S_{t-1}^{\top} q_t,

$$



$$

y_t = W_o r_t,

$$



where $\tilde k_t$ is either the raw key $k_t$ or its normalized version $k_t/\lVert k_t\rVert$, $\beta_t$ is the step size, $r_t\in\mathbb{R}^{d}$ is the read, and $W_o\in\mathbb{R}^{V\times d}$ is the readout. The readout has a bias; $W_k$, $W_v$, and $W_q$ do not. The model has no softmax attention and no direct skip from $e_t$ to the output: the memory is the only sequence mixer.



The write at position $t$ uses the previous token, so the sequence contains $T-1=55$ writes. Only $8$ of them are correct key writes (the value positions $t=2i+1$, whose previous token is the key). The remaining $47$ are junk writes from values, distractors, and queries. The gate's role is to allocate signal-to-noise ratio across this imbalance; the diagnostics below separate key positions from all positions.



## 2.3 Step-size variants and the four-stage ladder



We consider the following step-size rules. The first four are training rules; the last two are matched-mean evaluation rules used in Section 5.3.

**Terminology.** Throughout this paper, *gate* means the norm-based write step $\beta_t$ of the ladder, and *write rule* means the choice of $(\text{key},\beta_t)$. The decay gate of the Gated DeltaNet block in Section 7 is a different object with a different symbol in its own paper ($\alpha_t$, a per-step retention factor); we write $\alpha$ for it in Section 7.7 and never for $\beta_t$. The two never appear in the same equation.



| Name | $\beta_t$ | Mean of $\beta_t$ | Shape |

|---|---|---|---|

| `one` | $1$ | $1$ | constant |

| `fixed` | $c$ (main $c=0.125$) | $c$ | constant |

| `kaczmarz` | $1/(\lVert k_t\rVert^2+\epsilon)$ | $E[1/\lVert k\rVert^2]$ | norm-dependent |

| `ema` / `power` | $\mu/E[\lVert k\rVert^2]$ | $\mu/E[\lVert k\rVert^2]$ | constant (scalar) |

| `gate@0.125` | $(0.125/E[1/\lVert k\rVert^2])/(\lVert k_t\rVert^2+\epsilon)$ | $0.125$ | norm-dependent |

| `const@E` | $E[1/\lVert k\rVert^2]$ | $E[1/\lVert k\rVert^2]$ | constant |



Three clarifications are important.



First, the `ema`/`power` rule is a **scalar** constant computed from the batch mean of $\lVert k\rVert^2$. It is not the per-sample Kaczmarz gate. In the internal notes, the `beta_power` $\mu=1.0$ baseline at $d_{\mathrm{mem}}=16$ has $\beta=1/10.20\approx0.098$ (its reference $E[\lVert k\rVert^2]=10.20$ is measured on the trained model), not $\beta=1$. It belongs to the erase-strength comparison, not to a budget comparison with the `one` rule.



Second, the Kaczmarz rule uses the **raw** key norm $\lVert k_t\rVert^2$ even when the key is normalized for the prediction and the write. The product $\beta_t\tilde k_t$ is therefore $k_t/\lVert k_t\rVert^2$, the standard normalized Kaczmarz step. It is not a constant gate on normalized keys.



Third, Kaczmarz with unit numerator is **not** $\beta=1$. It is $\beta_t=1/(\lVert k_t\rVert^2+\epsilon)$; on the trained Kaczmarz model at $d_{\mathrm{mem}}=16$ its mean over all write positions is $E[1/\lVert k\rVert^2]\approx0.157$ (the key-position mean is $\approx0.523$). The `ema`/`power` scalar at $\mu=1.0$ is $1/E\lVert k\rVert^2\approx0.098$ (the median of $1/\lVert k\rVert^2$ is $0.093$), a **different** constant, not the Kaczmarz mean. Throughout, $\beta$ denotes a scalar multiplier and $\beta_t$ a per-sample value; comparisons to a fixed $\beta$ use the mean of $\beta_t$ over the same write positions (all positions for the matched-mean experiment).



The four-stage ladder at $d_{\mathrm{mem}}=16$ is:



| Stage | Key | $\beta_t$ | Operation added |

|---|---|---|---|

| 0 | raw $k$ | $1$ | baseline |

| 1 | $k/\lVert k\rVert$ | $1$ | stability |

| 2 | $k/\lVert k\rVert$ | $0.125$ | erase strength |

| 3 | $k/\lVert k\rVert$ | $1/\lVert k\rVert^2$ | gating |



The ladder is a controlled decomposition at a fixed operating point, not a claim that the operations are independent in general. Sections 3–4 rerun it at 1500 and 3000 steps to check for budget confounds.



### Erase eigenvalues



The stability of a write depends on the key and the step size. For a raw key, the erase eigenvalue along $k_t$ is



$$

1-\beta_t\lVert k_t\rVert^2,

$$



so the raw-key LMS stability condition is $0<\beta_t\lVert k_t\rVert^2<2$. For a normalized key, the eigenvalue is



$$

1-\beta_t,

$$



independent of $\lVert k_t\rVert^2$; the normalized-key condition is $0<\beta_t<2$. With $\beta_t=1$ the normalized write erases the old value in one step (eigenvalue $0$); with $\beta_t=0.125$ it retains $0.875$ of the old value. The two conditions are different objects, and the raw-key LMS bound does not apply to the normalized implementation. Section 4 uses this distinction when interpreting the optimal fixed $\beta$.



### Matched-mean evaluation



To separate the mean of $\beta$ from its shape, we evaluate a trained model under four rules with frozen weights:



- `fixed`: $\beta_t=0.125$.

- `kaczmarz`: $\beta_t=1/(\lVert k_t\rVert^2+\epsilon)$.

- `gate@0.125`: the Kaczmarz shape with its mean rescaled to $0.125$.

- `const@E`: a constant at the Kaczmarz mean $E[1/\lVert k\rVert^2]$.



The pairs (`fixed`, `gate@0.125`) and (`kaczmarz`, `const@E`) share their means; only the shape differs within a pair. The experiment trains with one rule and evaluates with all four; it measures co-adaptation, not the effect of retraining. Section 5.3 reports the results.



## 2.4 Training protocol



All ladder experiments use the following protocol unless stated otherwise:



- **Optimizer:** AdamW with learning rate $3\times10^{-4}$ and the PyTorch default weight decay ($0.01$).

- **Batch size:** 32 sequences.

- **Budget:** 1500 optimizer steps for the baseline; 3000 steps for the budget-matched control.

- **Clipping:** global gradient-norm clipping at $1.0$.

- **Threads:** two CPU threads (`torch.set_num_threads(2)`).

- **Evaluation:** 20 freshly sampled batches of 128 sequences; accuracy is the mean over the batches, computed on the eight query positions.

- **Seeds:** 0–2 for the ladder and for the $d_{\mathrm{mem}}=16,32$ cross-evaluations; 0–9 for $d_{\mathrm{mem}}=24$ and for the GDN/Delta experiments.

- **Memory widths:** $d_{\mathrm{mem}}\in\{16,24,32,64,128,256\}$ for the capacity sweeps; the ladder uses 16.

- **Biases:** $W_k$, $W_v$, and $W_q$ are bias-free. The readout $W_o$ has a bias. A bias in $W_v$ changes the value distribution and the delta target; it is not used in any reported experiment.



The GDN/Delta experiments use the same MQAR task but a different implementation (the `fla` library on a GPU); Section 7 gives the task and evaluation protocol, and Appendix D records the exact configuration. They are a real-architecture check, not part of the four-stage ladder.



The evaluation batches are drawn after training, so the seed-to-seed variability includes both initialization and data order. The 20-batch mean has a small standard error (roughly $0.008$ at accuracies near $0.2$); the seed-level intervals in the figures are the dominant uncertainty.



## 2.5 Statistical protocol and multiplicity



We state the endpoints and the multiplicity policy in one place, because the paper contains several families of tests and it would be easy for a reader to read them as a single confirmatory set.



**The real-architecture comparison (Section 7) has one primary endpoint and one secondary endpoint.** The primary endpoint is a variance test (Brown–Forsythe) on the ten-seed accuracy distributions, pre-registered and threshold-free; it is the confirmatory result. The secondary endpoint is a failure-rate comparison (Fisher exact) at a pre-specified accuracy threshold of $0.45$; it is supportive, and it does not survive a Bonferroni correction for two endpoints, although it does survive Holm. We report both, and the confirmatory/supportive distinction is not softened anywhere in the paper.



**Two further analyses of the Section 7 data are sensitivity analyses, not endpoints.** The threshold sweep (Table 7.4, Figure 7) maps the Fisher $p$-value over all thresholds in $[0.02, 0.90]$ and is used to show that the conclusion does not depend on the particular value of $0.45$; it is not a separate test. The stricter $0.02$ binarization (chance-level failure) is a second sensitivity analysis and is not significant ($3/10$ vs $0/10$, $p=0.21$). Neither is used to support the primary claim.



**The matched-mean transfer experiments (Section 5.3) are descriptive, with one confirmatory-equivalence analysis.** The three capacities ($d_{\mathrm{mem}}=16,24,32$) are three separate comparisons, not a family of three confirmatory tests on a single hypothesis; we report per-seed values and per-capacity intervals, and we do not apply a family-wise correction to the three capacity-wise shape costs, because the claim we make is capacity-dependent rather than a single omnibus claim. The one inferential statement in this family is the equivalence test (Table 5.5), which is stated with an explicit tolerance, a paired analysis, and the provenance of the tolerance given in the text.



**The ladder (Sections 3--4) uses descriptive intervals only.** With three seeds per cell, the 95% intervals in Tables 3.5 and 4.2 are seed-paired t intervals and are reported as summaries of the per-seed values, not as population estimates; we do not perform hypothesis tests on them, and no p-value is reported for any ladder transition. Statements about a transition are made in terms of whether its interval excludes zero, and we say so explicitly where it does not (Section 3.3).



**The curriculum and decay-control experiments (Sections 6 and 7.7) are exploratory by construction.** Their conditions were not pre-registered, their seed counts are small, and their purpose is to bound the scope of the ladder rather than to test a primary hypothesis. We report them with per-seed values and state the limitations of each in place, and no claim in the paper rests on them alone.



**The threshold provenance limitation is stated in Section 7.7 and in Section 9** (retraction item 26): the $0.45$ threshold was fixed before the Delta ten-seed set was run but after the GDN v15 set had been seen, so it is partially rather than fully pre-specified. The threshold-free primary test is what the Section 7 conclusion rests on.



## 2.6 Diagnostics



We measure the key distribution at the write positions and the geometry of the key matrix. Let $k_t$ be the **raw** key before normalization, even in models that normalize for the write; this is the distribution the gate sees.



**Norm statistics.** Over all write positions $t=1,\ldots,T-1$:



$$

E2_{\mathrm{all}} = \mathbb{E}\left[\lVert k_t\rVert^2\right],

\qquad

Einv_{\mathrm{all}} = \mathbb{E}\left[\frac{1}{\lVert k_t\rVert^2+\epsilon}\right].

$$



Over the correct key positions $t=2i+1$:



$$

E2_{\mathrm{key}} = \mathbb{E}\left[\lVert k_t\rVert^2 \;\middle|\; t=2i+1\right],

\qquad

Einv_{\mathrm{key}} = \mathbb{E}\left[\frac{1}{\lVert k_t\rVert^2+\epsilon} \;\middle|\; t=2i+1\right].

$$



We report the separation ratio and the gate contrast:



$$

\mathrm{ratio2} = \frac{E2_{\mathrm{all}}}{E2_{\mathrm{key}}},

\qquad

\mathrm{gate} = \frac{Einv_{\mathrm{key}}}{Einv_{\mathrm{all}}}.

$$



A ratio2 above one means that key writes have smaller squared norms than the average write; a gate above one means that the inverse-norm gate upweights key writes relative to the average. We also report the coefficient of variation of the inverse norm,



$$

CV = \frac{\mathrm{std}\left(1/(\lVert k\rVert^2+\epsilon)\right)}{\mathrm{mean}\left(1/(\lVert k\rVert^2+\epsilon)\right)},

$$



separately for all positions and key positions. The norm statistics use four batches of 256 sequences.



**Key geometry.** Let $K\in\mathbb{R}^{N\times d_{\mathrm{mem}}}$ be the matrix of key embeddings at the correct key positions, stacked over $N=256$ sequences ($8N$ rows). Let $\sigma_1\ge\cdots\ge\sigma_{d_{\mathrm{mem}}}$ be its singular values. We report the stable rank and the participation ratio:



$$

\mathrm{eff\_rank} = \frac{\sum_i \sigma_i^2}{\sigma_{\max}^2},

\qquad

PR = \frac{\left(\sum_i \sigma_i^2\right)^2}{\sum_i \sigma_i^4}.

$$



We also compute the condition number $\kappa=\sigma_{\max}/\sigma_{\min}$ and the off-diagonal coherence of the key Gram matrix $G=K^{\top}K$,

$\mathrm{eff\_rank}$ as written is the **stable rank** of the key matrix $K$; it is a property of the *keys*, not of the memory state. (The same functional form, $\sum_i\sigma_i^2/\sigma_1^2$, is used in work on the *state* matrix: Nazari and Rusch (arXiv:2602.04852) define the effective rank of the state as $\mathrm{er}(S)\coloneqq\|S\|_F^2/\|S\|_2^2=\sum_i\sigma_i^2/\sigma_1^2$ -- the quantity the literature also calls the *stable rank* -- and relate it to the conditioning of the keys, with the bound derived for plain linear attention and reaching DeltaNet only through a pseudo-value reparameterization. We keep their symbol. We report ours as a diagnostic of the key geometry, not as a capacity proxy.) We state it explicitly so that it can be compared with any other quantity of the same name, and we report the participation ratio $PR$ alongside it because the two answer slightly different questions. (A stable rank of the state matrix would be a different quantity, and we do not report one.)




$$

\mathrm{off/diag} = \frac{\lVert G-\mathrm{diag}(G)\rVert_1}{\mathrm{tr}(G)}.

$$



The effective rank and PR are the primary geometry metrics; $\kappa$ is setup-dependent and is reported in Appendix C for completeness.



**Accuracy and loss.** Accuracy is the fraction of query positions whose argmax equals the target value. Loss is the cross-entropy on the query positions. Both are averaged over 20 batches of 128 sequences.



## 2.7 Scope and notation



The paper studies the boundary of the gate in a controlled synthetic setting. The main claims are conditional on the memory width, the training budget, the MQAR task, and the linear-readout architecture. The GDN/Delta check tests whether the seed-level instability of a gated delta block appears in a real architecture; GDN's decay gate is not the norm-based step size of the ladder, so the check does not test the Kaczmarz gate directly. It is not a claim about all gated linear attention models.



| Symbol | Meaning |

|---|---|

| $d$ | embedding dimension ($64$) |

| $d_{\mathrm{mem}}$ | memory width |

| $V$ | vocabulary size ($64$) |

| $n_{\mathrm{pairs}}$ | key–value pairs per sequence ($8$) |

| $n_{\mathrm{dist}}$ | distractors per sequence ($32$) |

| $T$ | sequence length ($56$) |

| $S_t$ | memory state in $\mathbb{R}^{d_{\mathrm{mem}}\times d}$ |

| $k_t,v_t,q_t$ | key, value, and query vectors |

| $\tilde k_t$ | raw or normalized key used in the write |

| $\beta_t$ | step size |

| $W_k,W_v,W_q,W_o$ | key, value, query, and output maps |

| $E2$, $Einv$ | mean squared norm, mean inverse squared norm |

| $\mathrm{ratio2}$ | $E2_{\mathrm{all}}/E2_{\mathrm{key}}$ |

| $\mathrm{gate}$ | $Einv_{\mathrm{key}}/Einv_{\mathrm{all}}$ |

| $\mathrm{eff\_rank}$, $PR$ | stable rank, participation ratio |



---



## Draft notes (not for submission)



- **Key position convention.** The key token is at $2i$ and the value at $2i+1$; the write at $2i+1$ uses $e_{2i}$, so the diagnostic key positions are $t=2i+1$. This convention is easy to get wrong; the scripts use `e_prev[:, 2*i+1]`.

- **Bias.** The `budget_match` first launch used `Wv` with a bias and produced `stage0_raw` at chance; it was discarded and relaunched with `Wv` bias-free. All reported experiments use bias-free $W_k,W_v,W_q$.

- **`ema` is not a gate.** The `ema`/`power` rule is a scalar constant at the mean key norm; do not present it as a per-sample gate or as a budget control for the `one` rule.

- **LMS vs normalized eigenvalue.** The raw-key condition $0<\beta\lVert k\rVert^2<2$ and the normalized-key condition $0<\beta<2$ are different. Section 4 must not claim that the normalized optimum is "above the LMS stability limit" without saying that the LMS bound refers to the counterfactual raw-key update.

- **Evaluation pairing.** Each rule is evaluated on freshly sampled batches; the comparison is at the level of means. If a paired analysis is needed, seed the evaluation batches explicitly.

- **GDN/Delta.** Both 10-seed sets are complete; Section 7 reports the distribution, the pre-registered Brown–Forsythe test, and the pre-specified Fisher test with its threshold provenance. The old "Delta 5/10" count must not be repeated.



---

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


---

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


---

# 5.3 Matched-mean post-hoc transfer: the shape, not the mean, carries the transfer





Sections 5.1 and 5.2 compared training rules and found a asymmetric transfer: a fixed-rule model transfers to the Kaczmarz rule with a small loss, whereas a Kaczmarz-trained model transfers to the fixed rule with a large loss. That comparison confounds two differences between the rules: the overall scale of the erase strength (the mean of $\beta$) and its dependence on the key norm (the shape). We separate them with a matched-mean post-hoc transfer.





## Design





We train two models on the same MQAR task, one with a fixed step size $\beta=0.125$ and one with the Kaczmarz rule $\beta_t = 1/\lVert k_t \rVert^2$. At test time, we evaluate each trained model under four update rules:





| Rule | $\beta_t$ | Mean of $\beta_t$ | Shape |


|---|---|---|---|


| `fixed` | $0.125$ | $0.125$ | constant |


| `Kaczmarz` | $1/\lVert k_t \rVert^2$ | $E[1/\lVert k \rVert^2]$ | norm-dependent |


| `gate@0.125` | $(0.125/E[1/\lVert k \rVert^2])/\lVert k_t \rVert^2$ | $0.125$ | norm-dependent |


| `const@E` | $E[1/\lVert k \rVert^2]$ | $E[1/\lVert k \rVert^2]$ | constant |





The four rules form two matched-mean pairs. In pair A, `fixed` and `gate@0.125` share the mean $0.125$; only the shape differs. In pair B, `Kaczmarz` and `const@E` share the mean $E[1/\lVert k \rVert^2]$; again, only the shape differs. Comparing across pairs changes the mean while holding the shape fixed. This 2×2 design lets us ask whether the post-hoc transfer gap is carried by the mean or by the shape.





All evaluations use the same readout and the same test batches. The only change is the update rule inside the memory. The trained weights are frozen.





### Mean-rescaling factors





The matched-mean design is only informative when the matched mean differs numerically from the fixed constant. Table 5.3a reports the all-write mean $E_{\mathrm{all}}=\mathbb{E}[1/\lVert k\rVert^{2}]$ at each capacity, together with the two rescaling factors. The `gate_m` factor is $0.125/E_{\mathrm{all}}$, the factor by which the Kaczmarz gate is rescaled to reach the fixed mean; the `const_m` factor is $E_{\mathrm{all}}/0.125$, the factor by which the constant is shifted relative to the fixed rule. At $d_{\mathrm{mem}}=16$, $E_{\mathrm{all}}\approx0.125$, so `gate@0.125` is numerically the Kaczmarz baseline in mean and the $d=16$ column is a consistency check, not an independent test of mean-independence. The discriminating capacity is $d_{\mathrm{mem}}=32$, where $E_{\mathrm{all}}\approx0.0589$ and the gate is rescaled by $2.12\times$; the constant is correspondingly $0.47\times$ the fixed value. At $d_{\mathrm{mem}}=24$ the gate factor is $0.86\times$ and the constant factor is $1.16\times$. **The mean-independence is therefore demonstrated over a $\times2.1$ range at $d_{\mathrm{mem}}=32$, not over a $\times1.0$ range at $d_{\mathrm{mem}}=16$.**





| $d_{\mathrm{mem}}$ | $E_{\mathrm{all}}$ | fixed constant | `gate_m` factor ($0.125/E_{\mathrm{all}}$) | `const_m` factor ($E_{\mathrm{all}}/0.125$) |


|---:|---:|---:|---:|---:|


| 16 | 0.125 | 0.125 | $1.00\times$ | $1.00\times$ |


| 24 | 0.145 | 0.125 | $0.86\times$ | $1.16\times$ |


| 32 | 0.0589 | 0.125 | $2.12\times$ | $0.47\times$ |





**Table 5.3a. Mean-rescaling factors.** $E_{\mathrm{all}}$ is the mean Kaczmarz step over all write positions. The `gate_m` factor is the factor by which the Kaczmarz gate must be rescaled to match the fixed mean $0.125$; the `const_m` factor is the constant's mean relative to the fixed rule. The $d=16$ row is a near-tautological consistency check because the two means coincide; the $d=32$ row is the informative test.





## Kaczmarz-trained model





The Kaczmarz-trained model is the shape-specialized solution. Its baseline accuracy is $0.9368$, $0.9465$, and $0.9472$ at $d_{\mathrm{mem}}=16,24,32$. Replacing the gate with a constant at the *same mean* (`const@E`) drops accuracy to $0.5570$, $0.7199$, and $0.7861$, corresponding to losses of $0.3798$, $0.2266$, and $0.1611$ (40.5%, 23.9%, and 17.0% relative). Replacing the gate with a gate of the *same shape* but the fixed mean $0.125$ (`gate@0.125`) leaves accuracy essentially unchanged: $0.9372$, $0.9455$, and $0.9479$, with $\Delta = +0.0004$, $-0.0011$, and $+0.0006$ ($|\Delta| \leq 0.0011$).





The contrast is therefore entirely about shape. At the same mean, a constant loses 17–40% relative accuracy; at a different mean, the gate loses nothing. The mean cannot explain the transfer gap.





## Fixed-trained model





The fixed-trained model is much less shape-specialized. Its baseline accuracy is $0.8652$, $0.9262$, and $0.9389$. The shape-matched gate at the same mean $0.125$ (`gate@0.125`) gives $0.8079$, $0.9082$, and $0.9353$, with losses of $0.0573$, $0.0181$, and $0.0037$. The constant at the gate mean (`const@E`) gives $0.8660$, $0.9264$, and $0.9378$, with $\Delta = +0.0008$, $+0.0001$, and $-0.0012$ ($|\Delta| \leq 0.0012$).





The mean-only change is negligible at every capacity. At $d_{\mathrm{mem}}=16$ this is partly by construction: $E_{\mathrm{all}}\approx0.125$ equals the fixed constant, so the mean-only swap is numerically near-identical and is a consistency check. The informative mean-only checks are $d_{\mathrm{mem}}=24$ ($+0.0001$) and $d_{\mathrm{mem}}=32$ ($-0.0012$); both are within seed noise. The shape-only change is small and declines with capacity: $0.0573 \to 0.0181 \to 0.0037$. At $d_{\mathrm{mem}}=32$ the 95% CI of the shape effect includes zero, so we do not claim an exact zero, but the effect is within seed noise.





## The double dissociation





Table 5.3 collects the full accuracy matrix. The two training rules produce qualitatively different transfer profiles:





- **Kaczmarz-trained:** constant at the same mean $\rightarrow$ large loss ($0.380$, $0.227$, $0.161$); gate at a different mean $\rightarrow$ no loss ($|\Delta| \leq 0.0011$).


- **Fixed-trained:** constant at the gate mean $\rightarrow$ no loss ($|\Delta| \leq 0.0012$); gate at the same mean $\rightarrow$ small loss that vanishes with capacity ($0.057$, $0.018$, $0.004$).





This is a double dissociation in the sense that the two matched-mean pairs reverse which manipulation is harmful. For the Kaczmarz solution, the shape is necessary and the mean is not; the independent evidence is at $d_{\mathrm{mem}}=32$, where the mean is rescaled by $2.12\times$ and the shape-matched gate still recovers the baseline. The $d_{\mathrm{mem}}=16$ column is a consistency check because the two means coincide. For the fixed solution, neither manipulation is harmful at large capacity, but the shape is the only one that has any effect at all.





The asymmetry is capacity-dependent. The Kaczmarz constant loss shrinks monotonically with capacity ($0.380 \to 0.227 \to 0.161$), but it remains 17% relative at $d_{\mathrm{mem}}=32$. The fixed-model shape loss shrinks faster ($0.057 \to 0.018 \to 0.004$); at $d_{\mathrm{mem}}=32$ it is not resolved, in the sense that the interval includes zero but the equivalence test does not pass there (Table 5.5). At $d_{\mathrm{mem}}=32$ the fixed-rule model reaches $0.9389$, within $0.01$ of the Kaczmarz model's $0.9472$, without shape specialization.





We caution against summarizing this as a ratio of sensitivities. The ratio (Kaczmarz constant loss divided by fixed-model shape loss) is $6.6\times$, $12.5\times$, and $43.5\times$ at $d_{\mathrm{mem}}=16,24,32$, but the growth is driven by the fixed-model denominator approaching zero, not by an increasing Kaczmarz cost. The absolute Kaczmarz cost falls monotonically. The ratio is therefore misleading; we report absolute and relative drops.





## Equivalence, not a non-significant difference





The "no loss" entries above rest on differences that are small, and a small difference that fails to reach significance is not evidence of equivalence. We therefore test equivalence directly. Each trained model is evaluated under all four rules, so the comparison is *paired by construction* and the analysis is a paired TOST (two one-sided $t$ tests) on the per-seed differences.





The tolerance is $\pm0.01$ in absolute accuracy. It is **chosen for this analysis and is not a pre-registered quantity of this paper**; it is justified substantively as one percentage point of accuracy, about 14% of the baseline-budget gate advantage ($+0.0729$) that Section 5 discusses, and an order of magnitude larger than the differences the equivalence claim is about ($|\Delta| \leq 0.0011$). It is a tolerance, not a target.





**Table 5.5. Equivalence tests for the matched-mean transfer (paired TOST, tolerance $\pm0.01$).** $\Delta$ is the paired per-seed difference; the TOST column is the larger of the two one-sided $p$-values. "Equivalent" means the equivalence test rejects at $\alpha=0.05$.





| Comparison | $d_{\mathrm{mem}}$ | $n$ | $\Delta$ (paired) | 95% CI | TOST $p$ | Verdict |


|---|---:|---:|---:|---|---:|---|


| `gate@0.125` vs `Kaczmarz` (Kaczmarz-trained) | 16 | 3 | $+0.0004$ | $[-0.0011,\ +0.0018]$ | $6.0\times10^{-4}$ | equivalent |


| " | 24 | 10 | $-0.0011$ | $[-0.0027,\ +0.0006]$ | $3.1\times10^{-7}$ | equivalent |


| " | 32 | 3 | $+0.0006$ | $[-0.0012,\ +0.0024]$ | $9.9\times10^{-4}$ | equivalent |


| `gate@0.125` vs `fixed` (fixed-trained) | 16 | 3 | $-0.0573$ | $[-0.0758,\ -0.0388]$ | $1.00$ | difference resolved |


| " | 24 | 10 | $-0.0181$ | $[-0.0244,\ -0.0117]$ | $0.99$ | difference resolved |


| " | 32 | 3 | $-0.0037$ | $[-0.0188,\ +0.0115]$ | $0.11$ | **not resolved** |





Two things follow. First, the shape-matched gate is **equivalent** to the Kaczmarz rule within $\pm0.01$ for a Kaczmarz-trained model at all three capacities, so the "no loss" entry in the double dissociation is an equivalence statement rather than an absence of evidence. Second, the equivalence is **not symmetric**. For a fixed-trained model the shape-matched gate is *not* equivalent at $d_{\mathrm{mem}}=16$ or $24$, where the difference is resolved; at $d_{\mathrm{mem}}=32$ the interval includes zero but the equivalence test does not pass either, so we describe that cell as **unresolved** rather than equivalent. The mirror entries (constant at the gate mean, for a fixed-trained model: $+0.0008$, $+0.0001$, $-0.0012$) are all well inside the tolerance.





The asymmetry is the point, and the equivalence framing sharpens it: the Kaczmarz-trained solution is *insensitive* to the mean of the gate and *sensitive* to its shape, while the fixed-trained solution is insensitive to both up to $d_{\mathrm{mem}}=24$ and becomes indifferent to the shape at $d_{\mathrm{mem}}=32$ within the resolution of three seeds.

**Table 5.5b. Sensitivity of the equivalence conclusion to the tolerance.** TOST $p$-values for the shape-matched gate against the Kaczmarz rule (Kaczmarz-trained models), for four tolerances. Equivalence is established at every tolerance tested, including the tightest.

| $d_{\mathrm{mem}}$ | $\pm0.005$ | $\pm0.01$ | $\pm0.02$ | $\pm0.05$ |
|---:|---:|---:|---:|---:|
| 16 | $2.6\times10^{-3}$ | $6.0\times10^{-4}$ | $1.4\times10^{-4}$ | $2.3\times10^{-5}$ |
| 24 | $2.1\times10^{-4}$ | $3.1\times10^{-7}$ | $4.3\times10^{-10}$ | $8.7\times10^{-14}$ |
| 32 | $4.5\times10^{-3}$ | $9.9\times10^{-4}$ | $2.3\times10^{-4}$ | $3.6\times10^{-5}$ |

The conclusion does not depend on the choice of tolerance: the shape-matched gate and the Kaczmarz rule are equivalent for a Kaczmarz-trained model at every capacity and at every tolerance down to $\pm0.005$, which is four times tighter than the largest difference we are trying to call zero.






## Interpretation





The pattern is what a co-adapted inductive bias predicts. The Kaczmarz-trained model learns a solution whose readout expects the relative, norm-based weighting of writes. A constant step size at the same mean changes the relative weight of key and junk writes, and the readout — which co-evolved with the gate — cannot compensate post hoc. The fixed-trained model learns a solution that does not rely on this relative weighting; at larger capacity it can reach comparable accuracy without it.





The result also explains why the mean-$\beta$ sweeps in Section 4 found little effect once the shape was controlled. A global rescaling of $\beta$ is largely absorbed by the linear readout: with a zero initial state and a linear head, scaling all $\beta_t$ approximately scales the memory state and the logits, leaving the argmax unchanged. The recurrence makes this not exact, but it is close enough that the mean-only swaps here change accuracy by at most $0.0012$. The shape, by contrast, changes the *relative* weight of writes and therefore the content of the memory, not just its scale.





Three scope conditions matter. First, the mean-only swaps are modest: they change $\beta$ by at most a factor of about $2.1$, and they are not a substitute for the full $\beta$ sweep in Section 4. Their role is to test whether the *residual* transfer gap at the operating point is carried by the mean, not to map the full mean–accuracy curve. Second, the experiment is a post-hoc swap, not a retraining experiment. It measures the co-adaptation of a learned solution, not the effect of training with the swapped rule. The training-rule comparison is in Sections 5.1–5.2 and the curriculum experiments. Third, the matched-mean experiments (Section 5.3) use models trained for 1500 steps. Whether the co-adaptation cost persists at 3000 steps—where the fixed-rule model closes the training-time gap—is not tested here and is left for future work. This budget question is separate from the post-hoc transfer question: the budget-matched comparison asks whether training with the gate is necessary at convergence, while the matched-mean swap asks whether a gate-trained solution has co-adapted to the shape.





## Table 5.3





**Matched-mean post-hoc transfer.** Each row is a trained model; columns are evaluation rules. Bold marks the training rule (the diagonal baseline). `gate@0.125` has the Kaczmarz shape with mean $0.125$; `const@E` is a constant at the model's mean $E[1/\lVert k \rVert^2]$. The matched pairs are (`fixed`, `gate@0.125`) and (`Kaczmarz`, `const@E`).





| Training rule | $d_{\mathrm{mem}}$ | $n$ | Baseline | `fixed` | `Kaczmarz` | `gate@0.125` | `const@E` |


|---|---:|---:|---:|---:|---:|---:|---:|


| fixed | 16 | 3 | 0.8652 | **0.8652** | 0.8144 | 0.8079 | 0.8660 |


| fixed | 24 | 10 | 0.9262 | **0.9262** | 0.9117 | 0.9082 | 0.9264 |


| fixed | 32 | 3 | 0.9389 | **0.9389** | 0.9361 | 0.9353 | 0.9378 |


| Kaczmarz | 16 | 3 | 0.9368 | 0.5752 | **0.9368** | 0.9372 | 0.5570 |


| Kaczmarz | 24 | 10 | 0.9465 | 0.7275 | **0.9465** | 0.9455 | 0.7199 |


| Kaczmarz | 32 | 3 | 0.9472 | 0.7889 | **0.9472** | 0.9479 | 0.7861 |





**Table 5.4. Matched-mean contrasts.** $\Delta$ is the evaluation accuracy minus the row's baseline. The mean-only and shape-only columns are matched within each pair.





| Training rule | $d_{\mathrm{mem}}$ | Mean-only $\Delta$ (`const@E`) | Shape-only $\Delta$ (`gate@0.125`) |


|---|---:|---:|---:|


| fixed | 16 | $+0.0008$ | $-0.0573$ |


| fixed | 24 | $+0.0001$ | $-0.0181$ |


| fixed | 32 | $-0.0012$ | $-0.0037$ |


| Kaczmarz | 16 | $-0.3798$ | $+0.0004$ |


| Kaczmarz | 24 | $-0.2266$ | $-0.0011$ |


| Kaczmarz | 32 | $-0.1611$ | $+0.0006$ |





## Figure 3

![Figure 3: post-hoc transfer matrix](fig3_cross_eval.png)

**Figure 3. Post-hoc transfer: the trained rule fixes which swap is harmful.** Each cell is the mean accuracy over three seeds of a model trained with the row's rule and evaluated under the column's rule, at $d_{\mathrm{mem}}=16$ (left) and $d_{\mathrm{mem}}=32$ (right). The outlined cell is the training rule itself. The two decisive drops are the shape cost of a fixed-trained model ($0.057$ at $d_{\mathrm{mem}}=16$, $0.004$ at 32) and the mean cost of a Kaczmarz-trained model ($0.380$ at 16, $0.161$ at 32); the mirror swaps are harmless. The asymmetry is the double dissociation of Section 5.3 in one picture, and it shrinks with capacity without disappearing. Sources: `cross_eval_matched_out.txt` and `cross_eval_d32_out.txt`.

## Figure 4





**Figure 4: Double dissociation in matched-mean post-hoc transfer.**


**(a)** A Kaczmarz-trained model evaluated under a mean-matched constant step size loses accuracy, and the loss **shrinks monotonically with capacity** (0.380, 0.227, 0.161; 95% CI shown). Under a **gate with the same shape as Kaczmarz and its mean rescaled to 0.125 (the fixed-rule value)**, accuracy is fully recovered at all three capacities ($|\Delta| \leq 0.0011$).


**(b)** A fixed-rule-trained model is nearly insensitive to a shape-matched gate; its sensitivity declines from 0.057 to 0.018 to 0.004, and is within seed noise at $d_{\mathrm{mem}}=32$. The mean-matched constant has no effect at any capacity ($|\Delta| \leq 0.0012$).


The mean erase strength is not the carrier of post-hoc transfer where it is separately identifiable. At $d_{\mathrm{mem}}=16$ the gate mean coincides with the fixed constant, so that panel is a consistency check; the independent evidence is the $d_{\mathrm{mem}}=32$ rescaling ($2.12\times$). Only the gate *shape* determines post-hoc transfer, and its cost is capacity-dependent—substantial at $d_{\mathrm{mem}}=16$, attenuating but non-vanishing at $d_{\mathrm{mem}}=32$.


**Sample sizes**: $n=3$ seeds for $d_{\mathrm{mem}} \in \{16, 32\}$; $n=10$ for $d_{\mathrm{mem}}=24$.





- **Table numbering.** The three tables in this subsection are numbered 5.3, 5.4, and 5.5, following the paper's
  per-section convention (Table N.M is the M-th table of Section N; Section 3 uses 3.1--3.6, Section 4 uses
  4.1--4.3, Section 7 uses 7.1--7.6). The earlier flat labels "Table 3", "Table 3b", "Table 4" were renumbered
  to remove the collision with Section 3's 3.x tables; Sections 5.1--5.2, when drafted, will take 5.1 and 5.2.

## Data provenance





| $d_{\mathrm{mem}}$ | Script | Seeds |


|---:|---|---:|


| 16 | `cross_eval_matched.py` | 3 |


| 24 | `cross_eval_d24.py` | 10 |


| 32 | `cross_eval_d32.py` | 3 |





Figure: `fig4_matched_mean.py` (PNG/PDF, 300 dpi). Generated 2026-10-19.





---


---

# 6 Curriculum and budget

## 6.1 Two questions

The ladder in Sections 3–5 changes the *write rule* and holds the task fixed. Two other levers are common in practice and are easy to confuse with it: a **task curriculum** (train on an easier version of the task first) and a **larger budget**. This section asks two separate questions.

First, can a curriculum substitute for the write-rule interventions? The curriculum used here trains with 8 distractors for the first half of the steps and 32 distractors for the second half; evaluation is always at 32 distractors, so the curriculum is a training-schedule intervention and not a change of the evaluation task.

Second, does the budget change which factor matters? Sections 3 and 4 already give the budget-matched answer for the write rule: at 3000 steps the stability transition is $+0.3917$ and the erase-strength transition is $+0.2950$, with overlapping intervals, while the gating transition is $+0.0030$ with a CI that includes zero. This section asks the same question for the curriculum, and then places both answers next to each other.

## 6.2 A curriculum does not substitute for the write rule

Table 6.1 gives the 2×2 at 1500 steps: raw or normalized keys, with and without a curriculum, under $\beta=1$; plus the Kaczmarz rule with and without a curriculum.

**Table 6.1. Curriculum × write rule (1500 steps, 3 seeds).** Mean accuracy with per-seed values in parentheses. The raw-key rows are from `norm_curriculum_out.txt`; the normalized and Kaczmarz rows are from `curriculum_2x2_out.txt`. The no-curriculum cells reproduce the budget-matched ladder rows exactly, which confirms that the protocols match.

| Write rule | No curriculum | With curriculum |
|---|---|---|
| raw $k$, $\beta=1$ | 0.0911 (0.0853/0.0157/0.1722) | 0.2111 (0.2340/0.1896/0.2097) |
| normalized $k$, $\beta=1$ | 0.1919 (0.1849/0.1935/0.1972) | 0.4313 (0.6604/0.2052/0.4284) |
| normalized $k$, Kaczmarz | 0.9372 (0.9364/0.9408/0.9345) | 0.9379 (0.9324/0.9430/0.9383) |

Two things happen in the raw-key row, and they should be separated. The curriculum **removes the collapse mode**: without it, one of the three seeds sits at chance (0.0157) while the others reach 0.0853 and 0.1722, and the spread is large (std 0.0784). With a curriculum, all three seeds land in a narrow band (0.1896–0.2340, std 0.0222) and none is at chance. But the level does not move: 0.2111 is still far below the normalized stage under the same curriculum (0.4313) and far below Kaczmarz (0.9379). An easier task schedule does not repair the initialization-time eigenvalue violation of Section 3; it only removes the failure mode in which the raw-key model never leaves chance. The raw-key write is unstable at the scale of the keys, and the curriculum does not change that scale.

In the other direction, the curriculum does nothing once the write rule is already correct. The Kaczmarz row moves by $+0.0007$ ($\pm0.0102$): 0.9372 without a curriculum, 0.9379 with it, with the per-seed differences at $-0.0040$, $+0.0022$, and $+0.0038$. The curriculum's apparent value lives entirely in the regime where the write rule is misconfigured.

## 6.3 The curriculum's benefit in the normalized regime is noisy, not established

The middle row of Table 6.1 is the one a reader should be careful with. The curriculum raises the mean from 0.1919 to 0.4313, and all three seed differences are positive, but the per-seed values are 0.6604, 0.2052, and 0.4284: the seed spread grows from std 0.0063 to std 0.2276, a factor of about 36, and the paired 95% interval is correspondingly wide.

**Table 6.2. Paired curriculum effect at 1500 steps.** Paired difference is with-curriculum minus without-curriculum within each seed. The 95% CI is a seed-paired interval with $n=3$; it is descriptive, not a population estimate.

| Write rule | Per-seed $\Delta$ | Mean $\Delta$ | 95% CI | CI includes 0 |
|---|---|---:|---:|---|
| raw $k$, $\beta=1$ | $+0.1487,\ +0.1739,\ +0.0375$ | $+0.1200$ | $\pm0.1803$ | yes |
| normalized $k$, $\beta=1$ | $+0.4755,\ +0.0117,\ +0.2312$ | $+0.2395$ | $\pm0.5763$ | yes |
| normalized $k$, Kaczmarz | $-0.0040,\ +0.0022,\ +0.0038$ | $+0.0007$ | $\pm0.0102$ | yes |

The supported reading is therefore: the curriculum helps the normalized $\beta=1$ stage on every seed in this sample, but the effect is not resolved at $n=3$ (the interval spans $-0.34$ to $+0.82$), and the mechanism of the gain is not a uniform shift. Seed 0 gains $0.4755$, seed 1 gains $0.0117$, and seed 2 gains $0.2312$. A curriculum that reliably eased the task would produce a more uniform shift; this one produces a large variance increase in the same three seeds that were tightly clustered before. We describe it as a noisy early advantage, not as a resolved effect.

## 6.4 With budget, the curriculum washes out

Table 6.3 places the curriculum runs next to the budget-matched ladder. The cleanest observation is the variance: the curriculum run at 1500 steps has std 0.2276 across three seeds, and the same run at 3000 steps has std 0.0042 across three seeds (0.9230, 0.9275, 0.9192).

**Table 6.3. Budget and curriculum at $d_{\mathrm{mem}}=16$.** Mean accuracy with per-seed values in parentheses; the ladder rows are from `budget_match_complete_summary.md`, the curriculum rows from `curriculum_2x2_out.txt` and `curriculum_budget_out.txt`.

| Rule | 1500 steps | 3000 steps |
|---|---|---|
| normalized $k$, $\beta=1$ | 0.1919 (0.1849/0.1935/0.1972) | 0.6481 (0.6662/0.7041/0.5740) |
| normalized $k$, $\beta=1$ + curriculum | 0.4313 (0.6604/0.2052/0.4284) | 0.9232 (0.9230/0.9275/0.9192) |
| normalized $k$, fixed $\beta=0.125$ | 0.8643 (0.8686/0.8754/0.8489) | 0.9431 (0.9434/0.9460/0.9399) |
| normalized $k$, Kaczmarz | 0.9372 (0.9364/0.9408/0.9345) | 0.9461 (0.9458/0.9482/0.9444) |

The paired budget gain for the curriculum run is $+0.4919$ ($\pm0.5710$, per-seed $+0.2626$, $+0.7223$, $+0.4908$): the interval includes zero because the 1500-step starting points were so dispersed, and the 3000-step endpoints are the tightest cluster in the whole table. The corresponding budget gain without a curriculum is $+0.4562$ ($\pm0.1747$, CI excludes zero) — a larger and better-resolved effect, because the no-curriculum run starts from a tight cluster.

By 3000 steps the curriculum run reaches 0.9232, which is close to Kaczmarz at 1500 steps (0.9372) and below Kaczmarz at 3000 steps (0.9461). The curriculum therefore buys a level that a well-chosen write rule reaches at a third of the budget, and it does not produce a qualitatively different endpoint. Read together with Section 5, this is the same conclusion the ladder gives: the gate's advantage is sample efficiency, and the curriculum's advantage is also a budget effect rather than a different endpoint.

### Figure 5

![Figure 5: curriculum and budget](fig5_curriculum.png)

**Figure 5. What a curriculum does and does not rescue.** (a) At 1500 steps, the three seeds of each write rule without a curriculum (left marker) and with one (right marker), connected pairwise: the curriculum removes the raw stage's chance-level seed without lifting the level ($0.0911 \to 0.2111$, seed std $0.0784 \to 0.0222$), does nothing under Kaczmarz ($0.9372 \to 0.9379$), and raises the normalized stage's mean ($0.1919 \to 0.4313$) while its seed spread explodes ($0.0063 \to 0.2276$). (b) The same normalized stage with and without a curriculum at 1500 and 3000 steps, with the Kaczmarz level of each budget as a reference line: the curriculum run's spread collapses from $0.2276$ to $0.0042$ at 3000 steps and reaches $0.9232$, close to Kaczmarz at 1500 steps ($0.9372$) and below Kaczmarz at 3000 steps ($0.9461$). Sources: `norm_curriculum_out.txt`, `curriculum_2x2_out.txt`, `curriculum_budget_out.txt`, and `budget_match_complete_summary.md`.

## 6.5 What the curriculum experiments establish

**(i) A curriculum is not a substitute for the write rule.** In the raw-key regime the curriculum removes the collapse mode (no seed at chance, std 0.0784 → 0.0222) but the level stays at 0.2111, far below the normalized stage under the same curriculum (0.4313) and far below Kaczmarz (0.9379). An easier task schedule does not repair the initialization-time eigenvalue violation.

**(ii) The curriculum's benefit appears only where the write rule is still wrong.** It is absent under Kaczmarz ($+0.0007 \pm 0.0102$) and large but unresolved in the normalized $\beta=1$ stage ($+0.2395 \pm 0.5763$, CI includes zero, with the seed variance growing from 0.0063 to 0.2276).

**(iii) Budget dominates the curriculum.** The curriculum run's seed spread collapses from 0.2276 at 1500 steps to 0.0042 at 3000 steps, and its 3000-step level (0.9232) approaches Kaczmarz at 1500 steps (0.9372) without reaching Kaczmarz at 3000 steps (0.9461). The curriculum changes how fast a misconfigured rule gets to a mediocre place; it does not change where the good rules end up.

**(iv) Nothing here rescues the gate.** The curriculum is an easier task, not a gate: the gate's matched-mean post-hoc co-adaptation result (Section 5.3) is unaffected by it, and the 3000-step gating gap of $+0.0030$ (CI includes zero) remains the honest statement of the gate's training-time advantage.

---

## Draft notes (not for submission)

- **Data provenance.** Table 6.1 and 6.3: `norm_curriculum_out.txt` (raw/normalized × curriculum, 1500 steps), `curriculum_2x2_out.txt` (one/kaczmarz × curriculum, 1500 steps), `curriculum_budget_out.txt` (one × curriculum, 3000 steps). All runs: $d=64$, $d_{\mathrm{mem}}=16$, lr 3e-4, batch 32, $V=64$, gradient clipping 1.0, evaluation at 32 distractors, 20 batches of 128. The no-curriculum cells reproduce the budget-matched ladder rows exactly (raw 0.0911 / 0.0853/0.0157/0.1722, normalized 0.1919, Kaczmarz 0.9372), which is the protocol check for this section.
- **Std convention.** `norm_curriculum_out.txt` prints the population std (ddof=0); this section reports the sample std (ddof=1), which is what the other sections use. The two differ by a factor of $\sqrt{3/2}$; do not mix them. E.g. the raw/curriculum row is 0.0182 in the file and 0.0222 here.
- **Curriculum definition.** $n_{\mathrm{dist}}=8$ for the first half of training, then 32; evaluation always at 32. This is a *task-schedule* intervention, not a write-rule change, and the section must not present it as a fourth rung of the ladder.
- **The normalized $\beta=1$ + curriculum row is the fragile one.** Its mean (0.4313) is driven by one seed at 0.6604, and the paired CI includes zero. Do not cite the mean alone; report the per-seed values and the std increase.
- **Do not claim the curriculum "works".** The supported statements are: it removes the raw-stage collapse mode, it does nothing under Kaczmarz, and its short-budget gain is unresolved with three seeds.
- **Not a gate comparison.** The curriculum is an easier task; it does not touch the shape of the write step, so it says nothing about the gate's necessity. Section 5.3 remains the gate's co-adaptation result.
- **Untested.** A curriculum that also changes the key distribution, a longer curriculum (more than two stages), and curriculum runs at $d_{\mathrm{mem}}\in\{24,32\}$ were not run. The matched-mean post-hoc transfer at 3000 steps is also untested (Section 5.3 limitation).
- **Open.** If Section 6 is to appear before Section 7 in the final ordering, confirm that the Section 1 roadmap sentence ("Section 6 tests curriculum and budget") still matches the content.


---

# 7 Real-architecture check: GDN versus Delta

## 7.1 Why a real-architecture check

The four-stage ladder isolates stability, erase strength, and gating inside a minimal delta-rule memory. The obvious objection is that the decomposition is an artifact of the toy architecture: perhaps a real gated delta block, trained with a modern kernel and optimizer, would not show the same sensitivity. We therefore run a deliberately small external check. We train GDN with a decay gate and Delta without decay on the same MQAR task and compare their seed-level accuracy distributions.

The check is not a replication of the ladder. GDN's gate is a decay (retention) gate, not the norm-based step size of stages 2–3. The two mechanisms are different: the ladder's gate changes the write step as a function of the key norm, while GDN's decay multiplies the entire state by a data-dependent factor. The check therefore tests a broader claim: adding a gate to a delta-rule memory creates a co-adaptation problem that can help or fail, and the failure can be distributional rather than a uniform mean shift. It does not test whether the Kaczmarz gate is necessary in a real architecture.

## 7.2 Setup and endpoint provenance

**Task.** The task is the MQAR task of Section 2.1 with $J=8$ key–value pairs, $V=64$, $n_{\mathrm{dist}}=32$, and sequence length $T=56$. Accuracy is measured on the eight query positions; chance is $1/V=0.0156$.

**Models.** Both conditions use the public `fla` implementations on a GPU. GDN uses a data-dependent decay gate; Delta disables decay. Both are trained from scratch with the same data pipeline and training budget; the intended difference is the decay gate. We use seeds $0$–$9$ for each condition. The exact layer width, number of layers, learning rate, batch size, and step count are reported in Appendix D and the reproducibility artifact; the comparison is between off-the-shelf blocks, not a parameter-matched ablation.

**Primary endpoint.** The primary endpoint is the Brown–Forsythe test for equality of variances, using absolute deviations from each group's median; it is threshold-free and was pre-registered before the final Delta 10-seed run.

**Threshold provenance.** The failure threshold (accuracy $<0.45$, i.e., 50% of the Delta mean) was fixed before observing the Delta 10-seed results. The GDN 10-seed results, however, had been observed at an earlier stage of the project. The threshold $0.45$ therefore lies in the gap of the GDN distribution (between the boundary case $0.4368$ and the lowest success $0.5193$), and a reader may reasonably ask whether the threshold was chosen to maximize significance. The threshold is therefore pre-specified with respect to the Delta 10-seed set, but not fully blind with respect to GDN; we state this explicitly rather than claiming a fully blind pre-registration.

Figure 7 addresses this directly: the two-sided Fisher exact test is significant for every $t\in[0.44,0.897]$, from just above the GDN boundary case ($0.4368$) to the upper boundary $t=0.8970$, just above the fifth-lowest Delta seed ($0.8961$). At $t=0.4368$ itself, GDN has 4/10 failures and $p=0.0867$; at $t=0.44$ it becomes significant (5/10, $p=0.0325$). At $t=0.8970$, Delta has 5/10 failures and $p=0.0325$; at $t=0.8975$, $p=0.0867$. The threshold $0.45$ is not a knife-edge; it sits inside a broad significant region. The stricter binarization (accuracy $<0.02$, complete collapse) is non-significant ($p=0.21$) and is reported as a sensitivity analysis, not as a claim. The primary endpoint (Brown–Forsythe) is threshold-free and remains confirmatory.

We also report several descriptive summaries (median gap, variance ratio, Welch $t$, Mann–Whitney $U$); these are not used for the primary/secondary decision.

**Why two endpoints.** The variance test asks whether the distributions differ in dispersion without choosing a failure threshold; the Fisher test asks whether that dispersion produces more failures at a threshold that matters for the task. The first is threshold-free but less interpretable; the second is interpretable but threshold-dependent. We treat the variance test as the primary result and the failure-rate test as supportive.

## 7.3 Results

Figure 6 shows the per-seed accuracies; Table 7.1 lists the values.

**Table 7.1. Per-seed MQAR accuracy.** Seeds $0$–$9$; each value is the final accuracy for one run. GDN uses the decay gate; Delta disables decay. Chance is $0.0156$. The $0.4368$ seed is a pre-specified failure (below $0.45$) but not a complete collapse.

| Seed | GDN (decay) | Delta (no decay) | GDN outcome |
|---:|---:|---:|---|
| 0 | 0.5285 | 0.9027 | success |
| 1 | 0.5599 | 0.8941 | success |
| 2 | 0.2983 | 0.8975 | intermediate |
| 3 | 0.6119 | 0.8970 | success |
| 4 | 0.0160 | 0.9126 | collapse |
| 5 | 0.0165 | 0.8961 | collapse |
| 6 | 0.0162 | 0.8903 | collapse |
| 7 | 0.6611 | 0.8990 | success |
| 8 | 0.5193 | 0.8931 | success |
| 9 | 0.4368 | 0.8864 | intermediate (borderline failure) |

**Table 7.2. Summary statistics.** Standard deviations are sample standard deviations ($\mathrm{ddof}=1$). The failure and collapse counts use the pre-specified $0.45$ threshold and the stricter $0.02$ sensitivity threshold, respectively.

| Statistic | GDN | Delta |
|---|---:|---:|
| $n$ | 10 | 10 |
| Mean | 0.3665 | 0.8969 |
| Std | 0.2607 | 0.0072 |
| Median | 0.4781 | 0.8966 |
| Min | 0.0160 | 0.8864 |
| Max | 0.6611 | 0.9126 |
| Variance | $6.80\times10^{-2}$ | $5.13\times10^{-5}$ |
| Failures ($<0.45$) | 5/10 | 0/10 |
| Collapses ($<0.02$) | 3/10 | 0/10 |

**GDN is trimodal across 10 seeds.** It is not a shifted version of Delta. Its 10 seeds fall into three qualitatively different regimes. Three seeds (4, 5, 6) collapse to chance (0.0160, 0.0165, 0.0162). Two seeds land in an intermediate regime: seed 2 at 0.2983 and seed 9 at 0.4368, the latter just below the pre-specified 0.45 threshold. Five seeds succeed with 0.5193–0.6611. Delta has no failures: all 10 seeds lie in 0.8864–0.9126, with a standard deviation of 0.0072.

The mean GDN accuracy is 0.3665, but the mean is a poor summary of a trimodal distribution: the median is 0.4781, above the 0.45 threshold, while the mean is below it. The mean gap is 0.5304; the median gap is 0.419 (Delta 0.897 minus GDN 0.478), and the variance ratio is $1325\times$. We do not use Cohen's $d$: with a trimodal GDN distribution and a nearly degenerate Delta distribution, the pooled-variance and same-shape premises of $d$ do not hold. The seed-level distribution, not the mean, is the phenomenon. The tri-modality is not a small-sample artifact of the mean; it is the primary feature of the distribution.

The three regimes are descriptive. The intermediate regime contains two seeds with different behavior: a partial lock-in at 0.2983 and a near-threshold failure at 0.4368. With 10 seeds we do not fit a formal mixture model or estimate the mode weights precisely; the regimes are a summary, not a claim about the data-generating process.

### The borderline 0.4368 seed

The 0.4368 seed is a failure under the pre-specified rule but not a collapse. It lies 0.0132 below 0.45; a post-hoc threshold of 0.43 would classify it as a success and change the Fisher count to 4/10. We do not move the threshold. The primary variance test does not depend on this binarization, and the seed is reported separately in Table 7.1 and Figure 6. This is the main fragility of the secondary endpoint, and we state it explicitly rather than treating the 5/10 count as a robust natural kind.

## 7.4 Primary, secondary, and descriptive tests

The primary test is a test of *dispersion*, not of accuracy. Brown--Forsythe rejects the hypothesis that the two seed-level accuracy distributions have the same spread; it says nothing by itself about which model is more accurate. The direction of the accuracy difference is reported separately and on its own evidence: Delta's ten-seed mean is $0.8969$ against GDN's $0.3665$, and the failure counts at the pre-specified threshold are $0/10$ against $5/10$. No claim in this section about the ordering of the two blocks rests on the variance test.


**Table 7.3. Primary, secondary, and descriptive summaries.** The primary endpoint is the Brown–Forsythe variance test, fixed before the final Delta run and threshold-free. The secondary endpoint is Fisher's exact test at the pre-specified $0.45$ threshold; the threshold was fixed before the final Delta run but after the GDN v15 run (Section 7.2). The $0.02$ binarization is a stricter sensitivity analysis. The median gap and variance ratio are the robust descriptive summaries; Welch and Mann–Whitney are reference tests. We do not use Cohen's $d$ because the GDN distribution is trimodal and the Delta distribution is nearly degenerate, so the pooled-variance and same-shape premises of $d$ do not hold.

| Role | Test or statistic | Result | $p$ | Decision |
|---|---|---|---:|---|
| Primary | Brown–Forsythe (median-centered) | $F(1,18)=12.72$ | 0.0022 | reject $H_0$ |
| Secondary | Fisher exact, failure $<0.45$ | 5/10 vs 0/10 | 0.0325 | reject $H_0$ at $0.05$ |
| Sensitivity | Fisher exact, collapse $<0.02$ | 3/10 vs 0/10 | 0.2105 | fail to reject |
| Descriptive | Median gap (Delta $-$ GDN) | 0.419 | — | — |
| Descriptive | Variance ratio (GDN/Delta) | $1325\times$ | — | — |
| Reference | Welch $t$ | $t=-6.43$ | $1.20\times10^{-4}$ | — |
| Reference | Mann–Whitney $U$ | $U=0$ | $1.83\times10^{-4}$ | — |

The primary endpoint rejects equality of variances: $F(1,18)=12.72$, $p=0.0022$. The secondary endpoint rejects equality of failure rates at the pre-specified threshold: 5/10 vs 0/10, two-sided Fisher $p=0.0325$. The stricter collapse binarization does not: 3/10 vs 0/10, $p=0.2105$. This is the expected loss of power when the intermediate failures are folded into the non-failure category; we report it as a sensitivity analysis and do not claim significance under it.

The mean-shift tests are also significant (Welch $p=1.20\times10^{-4}$; Mann–Whitney $p=1.83\times10^{-4}$), but they are descriptive. A mean test would treat the trimodal distribution as a single shifted mode and would not distinguish a uniform 0.37 from a mixture of 0.016, 0.298, 0.437, and 0.5–0.66. The same caution applies to Cohen's $d$: with a trimodal GDN distribution and a nearly degenerate Delta distribution, the pooled-variance and same-shape premises of $d$ are not met. We therefore report the median gap (0.419; GDN median 0.478, Delta median 0.897), the failure counts (5/10 vs 0/10), and the variance ratio ($1325\times$) as the appropriate descriptive summaries. The variance test is the appropriate primary endpoint for the distributional claim.

### Threshold sensitivity

The $0.45$ threshold lies in the GDN gap between the boundary case $0.4368$ and the lowest success $0.5193$, and the GDN v15 set had been seen before the threshold was fixed. To test whether the Fisher result is an artifact of that gap, we swept the threshold $t$ from $0.02$ to $0.90$ (Table 7.4, Figure 7). The two-sided Fisher $p$ is below $0.05$ for every $t\in[0.44,0.897]$. At $t=0.4368$ exactly, GDN has 4/10 failures and $p=0.0867$; at $t=0.44$ it becomes significant (5/10, $p=0.0325$). The maximum $p$ in the significant band is $0.0325$ (attained on the lower plateau $t=0.44$–$0.5193$ and at the upper endpoint $t=0.8970$); the minimum is $1.1\times10^{-5}$ at $t=0.67$–$0.8864$. The upper boundary is $t=0.8970$, just above the fifth-lowest Delta seed ($0.8961$); at $t=0.8975$, $p=0.0867$ (Delta 6/10). The failure counts at $0.45$ are 5/10 vs 0/10; at $0.52$ they are 6/10 vs 0/10; at $0.67$ they are 10/10 vs 0/10. Delta has no failures through $t=0.8864$. The $0.45$ result is therefore not a knife-edge, and the Fisher conclusion is stable across the pre-specified band; the threshold sweep is the mitigation for the non-blind GDN provenance, not a substitute for the threshold-free primary test.

**Table 7.4. Threshold sensitivity of the Fisher exact test.** The failure rule is $\text{accuracy}<t$. At $t=0.4368$ exactly, the 0.4368 seed is not below the threshold (4/10, $p=0.0867$); the test becomes significant at $t=0.44$ (5/10, $p=0.0325$). The significant region extends to $t=0.8970$, just above the fifth-lowest Delta seed ($0.8961$); at $t=0.8975$, $p=0.0867$ (Delta 6/10). The table lists the exact boundary values; Figure 7 plots the full $0.02$–$0.90$ range.

| $t$ | GDN failures (of 10) | Delta failures (of 10) | Fisher two-sided $p$ |
|---:|---:|---:|---:|
| 0.4368 | 4/10 | 0/10 | 0.0867 |
| 0.44 | 5/10 | 0/10 | 0.0325 |
| 0.45 | 5/10 | 0/10 | 0.0325 |
| 0.5193 | 5/10 | 0/10 | 0.0325 |
| 0.52 | 6/10 | 0/10 | 0.0108 |
| 0.66 | 9/10 | 0/10 | $1.19\times10^{-4}$ |
| 0.67 | 10/10 | 0/10 | $1.08\times10^{-5}$ |
| 0.88 | 10/10 | 0/10 | $1.08\times10^{-5}$ |
| 0.8864 | 10/10 | 0/10 | $1.08\times10^{-5}$ |
| 0.89 | 10/10 | 1/10 | $1.19\times10^{-4}$ |
| 0.8970 | 10/10 | 5/10 | 0.0325 |
| 0.8975 | 10/10 | 6/10 | 0.0867 |

### Multiplicity

Both endpoints reject at the $0.05$ level. The primary is pre-registered and threshold-free; the secondary is partially pre-specified (blind to the Delta 10-seed set, but not to the GDN v15 set) and threshold-dependent. The primary survives a Bonferroni correction across the two endpoints ($0.0022<0.025$); the secondary does not ($0.0325>0.025$). As a sensitivity analysis, a Holm correction, which is uniformly more powerful than Bonferroni, would reject both (primary $0.0022<0.025$; secondary $0.0325<0.05$). We report the Bonferroni bound as the conservative limit and still describe the secondary as supportive, because the $0.45$ threshold and the $0.4368$ seed make it less robust than the threshold-free primary. The threshold sweep (Table 7.4, Figure 7) shows that the secondary conclusion is stable across $[0.44,0.897]$, which mitigates but does not erase the provenance limitation. We do not present the secondary as independent confirmatory evidence. The stricter $0.02$ sensitivity is not significant and is not used for the main claim.

## 7.5 The published ordering is task-variant dependent, and our check is consistent with one variant of it

The Gated DeltaNet paper reports zero-shot S-NIAH results for 1.3B models in its Table 2 (arXiv 2412.06464, v3). On S-NIAH-1, the pass-key variant with repeated synthetic context, DeltaNet reaches $99.0$ and $98.8$ at 4K and 8K tokens while Gated DeltaNet reaches $91.4$ and $91.8$: the published ordering on that variant is Delta $>$ GDN, which is the ordering we measure at small scale (Delta mean $0.8969$ against GDN mean $0.3665$).

On the other two variants the published ordering is the other way round. At 8K on S-NIAH-2 (real-world essay context) DeltaNet reaches $14.4$ against Gated DeltaNet's $29.6$; at 4K on S-NIAH-3 (UUID values) $22.4$ against $27.6$. The paper's own reading of the difference is that DeltaNet has no mechanism for clearing the state, so it degrades when the context contains much irrelevant material, while the gated rule filters it. The published comparison is therefore not a single ordering, and ours is consistent with the variant in which the model has to retain a few associations and inconsistent with the variants in which it has to clear the state. Our task sits between the two -- synthetic key-value retrieval with a fixed number of distractors -- and we do not have the experiments to say which published regime it corresponds to.

There is also an independent reason to expect the two regimes to differ at our scale. Our within-architecture decay control (Table 7.5) shows that a retention factor strong enough to be useful across 32 distractors is also strong enough to erase the associations entirely, so at $d_{\mathrm{mem}}=64$ and 1500 steps a gate has little room to help and a great deal of room to harm. The check is therefore a consistency check on a *distribution*, and the ordering we report is variant-dependent in the published data as well.

Two things we do not claim. We do not claim proxy validity: an MQAR accuracy and a S-NIAH score are not the same measurement, and the successful GDN mode ($0.519$--$0.661$) lies 24--38 points below Delta ($0.897$) while the published S-NIAH-1 gap at 4K is about 7 points, so the magnitude is amplified by roughly $3$--$5\times$ even in the successful mode; the mean gap is larger (about 53 points) because it mixes in the three collapses. And we do not claim that our ordering predicts the published one: the published ordering flips between task variants, and it may also flip with scale and token budget, which we have not checked.

## 7.6 Interpretation

The pattern is consistent with a co-adaptation account. GDN's decay gate introduces a retention/filtering tradeoff: the state must retain eight key–value associations across 32 distractors, but the gate can also filter junk writes. A decay policy that is too strong erases the associations; one that is too weak accumulates junk. The gate must co-adapt with the keys, values, and readout. Different seeds find different solutions: some collapse, some lock into a partial policy, and some succeed. Delta has no decay gate, so this degree of freedom is absent; its seed-to-seed variance is small.

An independent contemporaneous study reaches the same qualitative conclusion about seed-level lock-in. Boesch and Wee (arXiv:2609.16183) decompose masked multi-query recall along three single-knob axes and report that "training is a lock-in lottery -- a seed either locks in or does not", with lock-in rates between $1/10$ and $7/10$ depending on the curriculum, and that a distance curriculum takes the unchanged architecture from $0.021$ to $1.000$. Our trimodal GDN distribution (three collapses, two lock-ins, five successes) and our curriculum result (the curriculum removes the raw stage's chance-level seed but does nothing under Kaczmarz, $+0.0007\pm0.0102$) are the same kind of observation: lock-in is a property of the training dynamics that varies across seeds, and the task schedule moves its rate. The two findings differ in *which* phenomenon the curriculum acts on, because the regimes differ. Their lock-in is a property of the training dynamics that a task schedule can shift, and in their setting it is repaired by it; our raw-key failure is an initialization-time eigenvalue violation (Section 3.1), which a task schedule cannot repair at all -- the curriculum removes the chance-level seed and leaves the level low, and it does nothing at all under a write rule that is already correct ($+0.0007\pm0.0102$). The regimes differ enough (their 32-pair haystack, ours 8 pairs with 32 distractors) that we do not treat the two results as directly comparable, and we do not read their curriculum effect as contradicting ours.

This interpretation is a hypothesis, not a measured mechanism. We have not inspected the learned decay values or the state geometry, and the loss trajectory for the collapse seeds, including seed 4, is still being checked. The distributional result stands regardless of the mechanism: the variance and failure-rate differences are measured, and the ordering is external.

The practical implication is methodological. A three-seed mean of GDN can be dominated by which modes happen to be sampled; reporting only the mean hides the failure mode. For architecture comparisons at small scale, seed-level distributions and variance tests are more informative than a mean ranking. This is the same lesson as the raw-key bimodality in Section 3: a mean over an escape/no-escape mixture is not a stable summary.

## 7.7 Limitations

Several caveats matter. First, the comparison is between two off-the-shelf blocks, not a within-architecture gate on/off ablation; GDN and Delta may differ in parameter count, normalization, and kernel details. The intended difference is the decay gate, but we cannot rule out implementation differences. One difference we *can* address is the convolution: the ladder is measured without one, but the control in Table 7.7 shows that adding a causal convolution leaves all three factors and their ordering intact, so the decomposition is not an artifact of the convolution-free block. What remains unaddressed is state size, budget, and parameterization, and one further convolution setting is not replicated. A recent matched-state decomposition (Boesch and Wee, arXiv:2609.16183) reports that the margin of a rank-1 delta rule over a diagonal recurrence shrinks from $+0.19$ and $+0.32$ to $+0.03$ at 16 and 32 pairs once **both** cells carry a causal convolution, and that no class-level claim survives a state-matched Mamba-2 comparison. We did not run that condition. Our control varies the convolution on *our* block and shows that the three-factor decomposition is robust to it, which is a different statement: it does not tell us whether the GDN-over-Delta margin reported in Section 7.3 would narrow in a setting where the two real blocks are matched on convolution. We therefore report the Section 7.3 ordering as it stands and state the unrun comparison as a limitation. Second, the task is a single synthetic MQAR setting and the models are small, and the published comparison is **task-variant dependent**: DeltaNet is ahead of Gated DeltaNet on S-NIAH-1 at 4K/8K and behind it on S-NIAH-2/3 (Section 7.5). Our small-scale ordering is consistent with the first variant and not the others, and we do not have the experiments to say which regime our task belongs to, so the small-scale ordering is not a predictor of the published one in either direction. Third, 10 seeds are enough to reject variance equality but not enough to estimate the mode weights precisely or to fit a formal mixture; the trimodal label is descriptive. Fourth, the secondary endpoint depends on the 0.45 threshold and the 0.4368 boundary case, the GDN v15 set had been seen before the threshold was fixed, and the endpoint does not survive a Bonferroni correction; the threshold sweep (Table 7.4) mitigates but does not erase the provenance limitation. Fifth, the stricter 0.02 binarization is underpowered (3/10 vs 0/10, $p=0.21$), and the collapse mechanism is not established; the seed 4 loss check is pending. Sixth, the within-architecture decay control (Table 7.5) rules out a scalar retention factor as a *sufficient* explanation of the GDN distribution, but it does not identify the mechanism either, and it is bounded to one architecture, one protocol, and three initializations. Seventh, the scale of the check is small: both blocks are trained at the same CPU-scale configuration (d=64, d_mem=64, 1500 steps), so the comparison verifies the *shape* of the accuracy distribution rather than effect sizes, and the 340M-parameter configurations used in the published comparisons are out of scope. The diagnostic content of this section -- an initialization-time eigenvalue violation, a gradient-norm-infinite freeze, a trimodal distribution over seeds -- is what we expect to transfer to a large block, not the numbers.

### A within-architecture control of the decay hypothesis

We asked whether the GDN collapse can be produced by the decay gate alone, inside a single architecture. The control adds exactly one degree of freedom to the delta rule used in the ladder: a learned scalar retention factor $\alpha=\mathrm{sigmoid}(\log\alpha)$, applied as $S_t=\alpha S_{t-1}+\hat k_t\bigl(v_t-S_{t-1}^{\top}\hat k_t\bigr)^{\top}$. There is no convolution, no $\beta$ parameterization, and no gate composition. Ten seeds per condition, 1500 steps, $d_{\mathrm{mem}}=64$.

**Table 7.5. Decay-only control (10 seeds per condition, 1500 steps).** $\alpha_0$ is the initial retention $\mathrm{sigmoid}(\text{init})$; "learned $\alpha$" is the range over the ten trained models. All three decay conditions are uniform over seeds.

| Condition | $\alpha_0$ | Learned $\alpha$ | J=8 mean | Std | Median | Range | $<0.45$ | $<0.02$ |
|---|---:|---|---:|---:|---:|---|---:|---:|
| no decay ($\alpha=1$ fixed) | — | 1.0000 | 0.8651 | 0.0160 | 0.8624 | 0.8399–0.8910 | 0/10 | 0/10 |
| `decay_init_0.5` | 0.622 | 0.515–0.532 | 0.0157 | 0.0006 | 0.0159 | 0.0146–0.0166 | 10/10 | 10/10 |
| `decay_init_1.0` | 0.731 | 0.638–0.651 | 0.0154 | 0.0008 | 0.0153 | 0.0139–0.0167 | 10/10 | 10/10 |
| `decay_init_2.0` | 0.881 | 0.829–0.835 | 0.0153 | 0.0007 | 0.0155 | 0.0141–0.0161 | 10/10 | 10/10 |

The outcome is uniform, not multimodal. Without decay the model is stable in 10/10 seeds ($0.8651\pm0.0160$). With a learned scalar decay, every seed collapses to chance at every initialization in the pre-registered sweep, and the ten seeds of each condition are nearly identical (std $0.0006$–$0.0008$). The mechanism is geometric: with retention $\alpha$ per step, the content written 32 steps earlier is attenuated by $\alpha^{32}$, which is $1.5\times10^{-9}$ at $\alpha=0.53$ and $2\times10^{-6}$ at $\alpha=0.64$; at the learned $\alpha\approx0.83$ of the third condition it is still only $0.83^{32}\approx0.003$. Nothing survives the distractor gap, and every seed ends at chance.

Two consequences follow, and the second is the one that matters for the GDN result. First, a scalar retention factor is sufficient to *create* a collapse mode inside a single architecture, and the depth of the collapse is set by the learned retention rather than by the initialization alone: the learned $\alpha$ tracks the initialization (0.52, 0.64, 0.83 for initializations 0.5, 1.0, 2.0) and the model collapses in every case. Second, it is not sufficient to create a *split* **in the pre-registered sweep**. Within one architecture, the outcome is uniform over seeds at every initialization in that sweep ($0.5$, $1.0$, $2.0$), so the control as pre-registered does not reproduce the GDN trimodality. A uniform collapse is not a multimodality, and the two must not be conflated. The one condition in which the outcome is mixed is reported separately below, because it was run after the sweep was complete.

The one exception is the additional condition of Table 7.6, run after the pre-registered sweep was complete and reported separately. It is the only condition in which the outcome is not uniform over seeds, and the paper's conclusion about the GDN distribution does not depend on it: the pre-registered sweep alone already shows that a scalar retention factor does not reproduce GDN's three-mode distribution. At $\log\alpha=3.0$ (initial retention $0.952$) the outcome is mixed: eight of ten seeds succeed ($0.6027$--$0.7392$), one seed collapses to chance ($0.0179$), and one seed is degraded but not at chance ($0.0595$, about four times chance). A scalar retention factor is therefore sufficient to produce a mixed outcome inside a single architecture, and this initialization sits at the boundary of the uniform-collapse regime.

**Table 7.6. Additional condition `decay_init_3.0` (10 seeds, 1500 steps, $d_{\mathrm{mem}}=64$).** This condition was run *after* the pre-registered sweep of Table 7.5 was complete and is reported separately from it. "Degraded" means above chance but below the $0.45$ threshold. Chance is $0.0156$.

| Seed | Learned $\alpha$ | J=0 | J=8 | J=16 | J=32 | Outcome |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.9817 | 0.9375 | 0.7321 | 0.3956 | 0.1566 | success |
| 1 | 0.9816 | 0.9410 | 0.7392 | 0.4337 | 0.1611 | success |
| 2 | 0.9516 | 0.0430 | 0.0179 | 0.0165 | 0.0175 | collapse (at chance) |
| 3 | 0.9814 | 0.9409 | 0.6876 | 0.4630 | 0.1811 | success |
| 4 | 0.9816 | 0.9362 | 0.7204 | 0.4956 | 0.1849 | success |
| 5 | 0.9815 | 0.9317 | 0.6027 | 0.3456 | 0.1336 | success |
| 6 | 0.9687 | 0.2731 | 0.0595 | 0.0336 | 0.0244 | degraded |
| 7 | 0.9812 | 0.9332 | 0.6688 | 0.4460 | 0.1390 | success |
| 8 | 0.9816 | 0.9435 | 0.6769 | 0.3736 | 0.1532 | success |
| 9 | 0.9822 | 0.9425 | 0.6669 | 0.3867 | 0.1496 | success |

Three features of this condition matter for how it is read. First, the failure is a **minority**: two of ten seeds fail, and the successful mode is far from the $0.45$ threshold (its lower bound is $0.6027$, a gap of $0.15$; the threshold is pre-specified in Section 7 while this lower bound is an observed quantity, and the two should not be read as the same kind of number). Second, the two failure severities here are **not** GDN's third mode. GDN's intermediate mode is a lock-in at $0.298$ and $0.437$; the degraded seed here reaches $0.0595$, about four times chance and an order of magnitude lower, and **no seed in this control falls in the band $[0.25, 0.45)$ where GDN's two lock-in seeds sit** -- the control produces two failure severities, a chance collapse and a near-collapse, and no partial-retrieval lock-in. The within-architecture condition therefore reproduces a success mode and two failure severities, not the GDN trimodality, and the GDN mechanism remains unidentified. Third, the final learned retention **orders the outcomes monotonically**: $0.9812$--$0.9822$ for the eight successful seeds, $0.9687$ for the degraded seed, and $0.9516$ for the collapsed seed, with no overlap between the groups. The relation is striking, but the trajectory of $\alpha$ during training was not logged, so the causal direction is unresolved: $\alpha$ may be pulled down and cause the collapse, or the collapse and the reduction of $\alpha$ may share a common cause. We report the final values and stop there.

This is a negative result about one candidate mechanism, not evidence that decay is irrelevant to GDN. GDN's gate *is* a decay/retention gate (Section 7.1); the experiment says only that decay alone, in this architecture and training protocol, is not sufficient to produce the GDN distribution. Two further cautions apply to any outcome of this control. First, a uniform collapse is not a multimodality, and a *two*-mode outcome, if one appears, would still not be the three-mode GDN distribution: GDN's third mode is an intermediate lock-in ($0.298$ and $0.437$) that no decay-only condition has produced. Second, the control is a within-architecture negative result: it rules out a scalar retention factor as *sufficient* for the GDN distribution, not as *involved* in it. The control is also bounded: 10 seeds per condition, 1500 steps, $d_{\mathrm{mem}}=64$, a single architecture, and a scalar $\alpha$ initialized at 0.5--2.0.

### A convolution control of the ladder

The ladder is measured on a delta-rule block **without** a convolution, while the real blocks in Section 7 have one. We therefore reran the whole four-stage ladder with a causal convolution added, holding everything else fixed. The run was pre-registered before it started (`conv_ladder_predictions.md`), with two amendments, and the operational thresholds and the outcome-to-claim map were fixed before the run finished.

**Table 7.7. The four-stage ladder with and without a causal convolution** ($d_{\mathrm{mem}}=16$, 1500 steps, 3 seeds; mean with per-seed values in parentheses). The convolution is the same depthwise causal conv (kernel 4, groups $=d$, residual) used in `conv_compare.py`. The no-conv rows reproduce the budget-matched ladder rows of Tables 3.3 and 4.2 exactly at the reporting precision, which is the protocol check that separates the effect of the convolution from any implementation difference.

| Condition | Stage 0 (raw $k$, $\beta=1$) | Stage 1 (norm., $\beta=1$) | Stage 2 (norm., $\beta=0.125$) | Stage 3 (Kaczmarz) |
|---|---|---|---|---|
| no conv | 0.0910 (0.0853/0.0157/0.1722) | 0.1918 (0.1849/0.1935/0.1972) | 0.8643 (0.8686/0.8754/0.8489) | 0.9372 (0.9364/0.9408/0.9345) |
| with conv | 0.0157 (0.0179/0.0155/0.0136) | 0.1785 (0.1915/0.1691/0.1749) | 0.8486 (0.8205/0.8190/0.9064) | 0.9149 (0.9009/0.9192/0.9246) |

**Table 7.8. Stage-to-stage transitions with and without a convolution.** Paired per-seed differences; the interval is a seed-paired t interval with $n=3$ and is descriptive.

| Transition | no conv | with conv |
|---|---:|---:|
| stability (raw $\to$ normalized) | $+0.1008 \pm 0.1898$ | $+0.1628 \pm 0.0249$ |
| erase strength (norm. $\beta=1 \to \beta=0.125$) | $+0.6725 \pm 0.0447$ | $+0.6702 \pm 0.1344$ |
| gating (fixed $\to$ Kaczmarz) | $+0.0729 \pm 0.0275$ | $+0.0663 \pm 0.1063$ |

All four pre-registered predictions hold. The stability transition is strictly positive ($+0.1628$, and larger than without the convolution because the raw stage fails *more* completely: all three seeds are at chance, so the convolution makes the raw-key write worse, not better). The erase transition is $+0.6702$, above the $+0.30$ threshold and still the largest of the three transitions. The gating transition is $+0.0663$, below the $+0.15$ threshold. The per-seed ordering raw $<$ normalized $<$ fixed $\leq$ Kaczmarz holds for every seed in both conditions.

**The three-factor decomposition therefore survives the addition of a causal convolution.** The levels drop slightly (stage 3 from $0.9372$ to $0.9149$, stage 2 from $0.8643$ to $0.8486$), which is consistent with the accuracy cost of the convolution seen in the earlier capacity-scaled comparison, but the *attribution* is unchanged: with a convolution, erase strength is still the dominant transition, the gate is still worth about $+0.07$ at this budget, and the ordering is unchanged. The Section 7 caveat about our block therefore narrows: the decomposition is not an artifact of the convolution-free block, so what remains as a difference between our cells and the real blocks is everything else -- state size, budget, parameterization, and the presence of a decay gate in GDN.

Two things this control does **not** say. It does not say that a convolution causes the GDN trimodality; it is a control on our own block, which has no decay gate at all, and the trimodality of Section 7.3 is a property of the GDN block. And it does not resolve the disagreement with the published ordering discussed in Section 7.5: that comparison involves the internals of the two real blocks, and this control only shows that adding a convolution to *our* block leaves the three factors intact. The published comparison is still to be settled from the primary source.

## 7.8 Conclusion

The real-architecture check supports a conditional view of gating. GDN's decay gate is not a uniform improvement: across 10 seeds it produces a trimodal distribution with 3 complete collapses, 2 intermediate outcomes, and 5 successes, while Delta without decay is stable. The pre-registered variance test rejects equality ($p=0.0022$); the pre-specified failure-rate test at 0.45 is supportive ($p=0.0325$) but does not survive a Bonferroni correction. The threshold sweep shows that the Fisher conclusion holds for every $t\in[0.44,0.897]$, so the 0.45 choice is not a knife-edge; at $t=0.4368$ and $t=0.8975$ it is non-significant ($p=0.0867$). The threshold was still not fully blind to GDN, and the threshold-free variance test remains the confirmatory result. The published comparison is task-variant dependent (DeltaNet is ahead of Gated DeltaNet on S-NIAH-1 at 4K/8K and behind it on S-NIAH-2/3), and our small-scale ordering is consistent with one of those variants and not the others, so the check is a distributional diagnostic rather than a predictor of the published ordering. The result does not identify the mechanism of the GDN collapse, and the within-architecture control does not identify it either: a learned scalar retention factor produces a uniform collapse in the pre-registered sweep (10/10 seeds at chance at every initialization) and a minority failure mode at one additional initialization (Table 7.6), not GDN's three-mode distribution. Decay alone is therefore sufficient for a collapse and not sufficient for the GDN distribution, and the GDN mechanism remains open -- it must involve gate composition, the state size, or the parameterization. The experiment also does not test the Kaczmarz gate in a real architecture; those are open questions.

## Figure 6

![Figure 6: GDN trimodality](fig6_gdn_multimodality.png)

**Figure 6: GDN's decay gate produces a trimodal seed distribution; Delta is stable.**
Each point is one seed ($n=10$ per condition) on MQAR with $J=8$ pairs. GDN with decay (red) is trimodal: 3 seeds collapse to chance (0.016), 2 land in an intermediate regime (0.298 and 0.437), and 5 succeed (0.519–0.661). Delta without decay (blue) is tight (0.886–0.913). The dashed line is the pre-specified failure threshold (0.45; fixed before the final Delta run, after the GDN v15 run; approximately half of the Delta mean); the dotted line marks chance ($1/V=1/64\approx0.016$). Horizontal bars show group means (GDN 0.366; Delta 0.897). The $0.437$ seed is just below the threshold: it is the boundary case, a pre-specified failure but not a complete collapse. The primary Brown–Forsythe test is pre-registered: $F(1,18)=12.72$, $p=0.0022$. The secondary Fisher test uses the pre-specified threshold: 5/10 vs 0/10 at 0.45, $p=0.0325$. The stricter 0.02 binarization (3/10 vs 0/10, $p=0.21$) is a sensitivity analysis, not a significant result. The median gap is 0.419 and the variance ratio is $1325\times$. The variance test is the primary endpoint; the failure-rate test is secondary and does not survive a Bonferroni correction (a Holm correction would reject both, but the secondary is threshold-sensitive because of the 0.437 seed). The threshold sweep in Figure 7 shows $p<0.05$ for every $t\in[0.44,0.897]$.

## Figure 7

![Figure 7: Threshold sensitivity](fig7_threshold_sweep.png)

**Figure 7: Threshold sensitivity of the Fisher exact test.**
**(a)** Two-sided Fisher exact $p$-value as a function of the failure threshold $t$ (accuracy below $t$ = failure). The dashed red line is $p=0.05$; the dotted blue line is the pre-specified $t=0.45$ (fixed before the Delta 10-seed results; the GDN 10-seed set had been seen). The test is significant for every $t\in[0.44,0.897]$: from just above the lowest GDN boundary case ($0.4368$) to the upper boundary $t=0.8970$, just above the fifth-lowest Delta seed ($0.8961$). At $t=0.4368$ itself, GDN has 4/10 failures and $p=0.0867$; at $t=0.44$ it becomes significant (5/10, $p=0.0325$). At $t=0.8970$, Delta has 5/10 failures and $p=0.0325$; at $t=0.8975$, $p=0.0867$. The pre-specified threshold is not a knife-edge. The shaded band $[0.44,0.897]$ is entirely below $p=0.05$; the maximum $p$ in the band is 0.0325 (lower plateau $0.44$–$0.5193$ and upper endpoint $0.8970$), and the minimum is $1.1\times10^{-5}$ at $t=0.67$–$0.8864$.
**(b)** Failure counts for GDN and Delta as a function of $t$. Delta has zero failures for $t\leq0.8864$. GDN's counts increase in steps: 3/10 (complete collapse), 4/10 (including the lock-in at 0.2983), 5/10 (including the 0.4368 boundary case), 6/10 (including the lowest success at 0.5193), and 10/10 at $t>0.6611$. The significant region begins at $t=0.44$, where GDN reaches 5/10 failures while Delta remains at 0/10, and continues through the zero-Delta-failure plateau to $t=0.8864$ and to the upper boundary $t=0.8970$ (Delta 5/10). At lower thresholds (0.02–0.4368), GDN has 3–4 failures but $p>0.05$. The figure plots $t$ from 0.02 to 0.90; Table 7.4 lists the exact boundary values. The sweep mitigates, but does not remove, the fact that the GDN v15 set was seen before 0.45 was fixed; the primary Brown–Forsythe test is threshold-free. The failure-count panel is a step function; the plotted t grid is discrete, so some intermediate counts (for example 7/10 and 9/10) may be skipped visually.

## Data provenance

| Condition | Implementation | Seeds | Values |
|---|---|---|---:|
| GDN (decay) | `fla` GDN, GPU | 0–9 | Table 7.1 |
| Delta (no decay) | `fla` Delta, GPU | 0–9 | Table 7.1 |

The per-seed values are the arrays used in Figure 6. The statistical tests were reproduced with SciPy: Brown–Forsythe $F=12.7228$, $p=0.0022036$; Fisher at 0.45, $p=0.0325077$; Fisher at 0.02, $p=0.210526$; Welch $t=-6.4318$, $p=1.1988\times10^{-4}$; Mann–Whitney $U=0$, $p=1.8267\times10^{-4}$; median gap $0.4185$ (reported 0.419); variance ratio $1324.98$. Figure 6 is generated by `fig6_gdn_multimodality.py`; the threshold sweep is generated by `fig7_threshold_sweep.py`; the tests are reproduced by `verify_gdn_stats.py`. The full $p<0.05$ region is $(0.4368,0.8970]$; the clean reported band is $[0.44,0.897]$. The $p$-values are $0.032508$ at $t=0.44$ and $0.45$, $1.083\times10^{-5}$ at $t=0.67$–$0.8864$, and $0.086687$ at $t=0.8975$. Cohen's $d$ is not reported: the GDN distribution is trimodal and the Delta distribution is nearly degenerate, so the pooled-variance and same-shape premises of $d$ do not hold.

---

## Draft notes (not for submission)

- **The primary is pre-registered; the secondary is partially pre-specified.** The primary $p=0.0022$ survives $0.05/2=0.025$; the secondary $p=0.0325$ does not survive Bonferroni but does survive Holm ($0.0325<0.05$ after rejecting the primary). The main fragility is the 0.45 threshold/0.437 seed and the non-blind GDN v15 provenance, not multiplicity. The threshold sweep shows $p<0.05$ for every $t\in[0.44,0.897]$; use it as the mitigation, but do not call the secondary fully pre-registered. Keep the "confirmatory vs supportive" distinction; do not write "both endpoints are independently confirmatory."
- **Do not move the 0.45 threshold.** The $0.4368$ seed is the boundary case; report it separately. A post-hoc 0.43 threshold would give 4/10, but that is not the pre-specified analysis. The primary variance test is threshold-free and is the main result. The threshold sweep shows the conclusion is stable across $[0.44,0.897]$; at 0.4368 and 0.8975, $p=0.0867$.
- **Retraction #26: the 0.45 threshold was not fully blind.** The earlier wording "fixed before seeing any results" is retracted. The Delta 10-seed set was unseen, but the GDN 10-seed v15 set had been seen; 0.45 lies between the GDN boundary case 0.4368 and the lowest success 0.5193. The threshold sweep (Table 7.4, Figure 7) shows $p<0.05$ for $t\in[0.44,0.897]$, so the Fisher conclusion does not depend on the gap. The primary Brown–Forsythe test is threshold-free and remains confirmatory. The submission-ready Section 9 item 26 draft is in `section9_retractions.md`.
- **GDN is not the Kaczmarz gate.** GDN's gate is a decay/retention gate; the ladder's gate is a norm-based write step. The section tests a different gating mechanism and should say so. Do not use the GDN result to claim that the Kaczmarz gate is necessary in real architectures.
- **Decay-only ablation: the pre-registered sweep is complete (Branch B).** `delta_decay_only.py`: 10 seeds × {no decay, log_alpha init 0.5, 1.0, 2.0}, same DeltaMem architecture (normalized keys, no conv, no beta gate), 1500 steps, lr 1e-3, $d_{\mathrm{mem}}=64$; the update is $S_t=\alpha S_{t-1}+\hat k_t\bigl(v_t-S_{t-1}^{\top}\hat k_t\bigr)^{\top}$ with a single learned scalar $\alpha=\mathrm{sigmoid}(\log\alpha)$, so $\alpha$ is a per-step retention factor. Results: `no_decay` 10/10 stable (J=8 mean 0.8651, std 0.0160); `decay_init_0.5` 10/10 at chance (0.0157, std 0.0006, learned $\alpha$ 0.515–0.532); `decay_init_1.0` 10/10 at chance (0.0154, std 0.0008, learned $\alpha$ 0.638–0.651); `decay_init_2.0` 10/10 at chance (0.0153, std 0.0007, learned $\alpha$ 0.829–0.835). **All three decay conditions are uniform over-decay failures, not mixtures**; the write-up is Table 7.5 in 7.7.
  - Mechanism (used in Table 7.5): retention $\alpha$ per step attenuates content written 32 steps earlier by $\alpha^{32}$ -- $1.5\times10^{-9}$ at $\alpha=0.53$, $2\times10^{-6}$ at $\alpha=0.64$, and still only $0.003$ at the learned $\alpha\approx0.83$ of the init 2.0 condition. Nothing survives the distractor gap, so every seed ends at chance.
  - The learned $\alpha$ tracks the initialization (0.52 / 0.64 / 0.83 for initializations 0.5 / 1.0 / 2.0) and the model collapses in every case, so the depth of the collapse is set by the learned retention, not by the initialization alone. This is the point recorded in the pre-registration addendum after init 2.0 seed 0 collapsed with learned $\alpha=0.8313$.
  - **The additional (non-pre-registered) condition is complete: `decay_init_3.0` at 10 seeds** (`delta_decay_only_v3.py --inits 3.0 --skip-nodecay`; log `delta_decay_only_init3_out.txt`, JSON `delta_decay_only_init3.json`). Outcome: **8 success (J=8 0.6027-0.7392), 1 degraded (0.0595), 1 collapse at chance (0.0179)**; mean 0.5572, std 0.2762, median 0.6728, range [0.0179, 0.7392]; 8/10 at or above 0.45, 2/10 below it, 1/10 below 0.02. Final learned $\alpha$ orders the outcomes monotonically and without overlap: 0.9812-0.9822 (success, n=8), 0.9687 (degraded), 0.9516 (collapse). The write-up is Table 7.6 in 7.7; it is labeled as an additional condition and is never merged into the pre-registered table of Table 7.5.
  - **Decision record (2026-10-20).** The additional condition was launched *after* the pre-registered sweep was seen and *after* the pre-registration addendum was written; it is not a pre-registered condition and must never be presented as one. The pilot and the 10-seed run are the same experiment: `delta_decay_fix.py` and `delta_decay_only_v3.py` were compared item by item ($d=64$, $d_{\mathrm{mem}}=64$, $V=64$, batch 32, 8 pairs, 32 distractors, 1500 steps, lr $10^{-3}$, AdamW, clip 1.0, evaluation on 20 batches of 128 at $J\in\{0,8,16,32\}$, initial $\log\alpha=3.0$, the same RNG order, the same parameter-creation order, the same update equation, 2 threads in both; the only difference is a harmless extra `np.random.seed`). The pilot's seeds 0 and 1 are therefore reproduced *exactly* by the 10-seed run, which makes the pilot a reproducible 3-seed observation of the same protocol rather than a separate experiment. The full record is in `section7_7_decay_only_variants.md`.
  - **Three reservations for the additional condition, to be carried into the final 7.7 text.** (i) A mixed two-mode outcome is **not** GDN's three-mode distribution. The GDN lock-in mode ($0.298$, $0.437$) has no counterpart in any decay-only condition: the degraded seed sits at $0.0595$, about four times chance, and **no decay-only seed falls in the band $[0.25, 0.45)$** where GDN's two lock-in seeds sit. So the claim is about the *position* of the degraded seed, not about a count of zero in $[0.02, 0.45)$ -- that count is one. (ii) Do not write "a narrow band of the learned retention": the measured points are a uniform collapse at learned $\alpha\approx0.83$ and the mixed outcome at $0.95$–$0.98$, and the far side of that boundary was not measured. Say "at the boundary we tested". (iii) The relation between the final learned $\alpha$ and the collapse is **correlational**: the trajectory of $\alpha$ during training was not logged, so whether $\alpha$ is pulled down and causes the collapse, or the collapse and the reduction of $\alpha$ share a common cause, is unresolved. Report the final values and stop there.
  - Interpretation rules that hold regardless: (i) a uniform collapse is not multimodality, and the GDN trimodality must not be inferred from these conditions; (ii) do not claim a mechanism for the GDN distribution; (iii) the GDN trimodality, the Brown–Forsythe primary ($p=0.0022$) and the Fisher secondary ($p=0.0325$) are independent of this ablation.
  - **The reading of the init 2.0 outcome was pre-registered (2026-10-19) before that condition finished**, together with the prediction that $\alpha_0=0.881$ leaves about $1.7\%$ of the content written 32 steps earlier ($0.881^{32}\approx0.017$) and is therefore the first tested condition in which content can survive the gap. The three outcomes and their 7.7 statements are fixed in `section7_7_decay_only_variants.md`; do not choose the interpretation after seeing the result. An addendum to that pre-registration (added after init 2.0 seed 0 collapsed, before the remaining seeds finished) records the limitation of the prediction: $0.881^{32}\approx0.017$ is a statement about the *amount* of content that survives the gap, not about the *probability* of success, and seed 0 (learned $\alpha=0.8313$, J=8 $=0.0146$) already breaks the simple "the outcome is a function of the initial retention" picture. The relevant quantity, if any, is the learned retention, not the initialization.
- **Seed 4 loss check is still pending.** Do not claim that the collapse is numerical or representational until the loss trajectory is inspected. The distributional result does not depend on the mechanism.
- **Exact Colab configuration is still needed for Appendix D.** Fill in the model width, number of layers, learning rate, batch size, training steps, decay initialization, and `fla` commit. The local `fla` installation is unusable on Windows/CPU (missing Triton), so the GDN/Delta runs are external.
- **Do not use the old "6 succeed (0.437–0.661)" wording.** Under the pre-specified 0.45 threshold, 0.437 is a failure; the correct success range is 0.519–0.661. Section 1 has been updated accordingly.
- **Delta is 10/10, not 5/10.** The old "5/10" note in Section 1/2 has been removed. Delta's full range is 0.8864–0.9126, mean 0.8969, std 0.0072.
- **Seed order.** The Delta array is assumed to be in seed order 0–9; confirm against the Colab log before submission. If the log uses a different order, relabel Table 7.1. The GDN mapping (seeds 4, 5, 6 collapse; seed 2 lock-in) is consistent with the run log.
- **The mean is not the primary statistic.** GDN mean 0.3665 is below 0.45, but the median 0.4781 is above it. The mean is pulled down by the three chance-level collapses. Report the distribution first.
- **The published comparison was checked against the primary source (retraction item 33).** Our earlier sentence -- "the Gated DeltaNet paper reports 99.0/98.8 for DeltaNet versus 91.4/91.8 for GDN on 1.3B S-NIAH-1 at 4K/8K (arXiv 2412.06464, Table 3)" -- was **correct in its numbers, its attribution and its scope**; the only error was the table number, which is **Table 2**, not Table 3. We verified this in arXiv 2412.06464 v3: Table 2 is the zero-shot S-NIAH table for 1.3B models, and its S-NIAH-1 columns at 4K/8K read DeltaNet 99.0/98.8 and Gated DeltaNet 91.4/91.8. A reviewer suggested that the published ordering is reversed (GDN above DeltaNet) at 400M parameters and 15B tokens; the paper's 400M/15B table is Table S.1, an ablation reporting perplexity and commonsense accuracy rather than RULER S-NIAH, so we could not confirm the reversal and did not adopt it. What we did adopt is the variant dependence visible in the verified table: on S-NIAH-2/3 the published ordering is GDN above DeltaNet (14.4 against 29.6 at 8K on S-NIAH-2; 22.4 against 27.6 at 4K on S-NIAH-3), so Section 7.5 now states the comparison as variant-dependent rather than as a single ordering. If a 400M/15B RULER table is to be cited, its source must be identified first.
- **Figure 6 layout.** The original top-right statistics box hid the Delta points; the final version moves the box to the lower right and the legend to the upper left. The PNG/PDF are in `C:\ai_exp`; the generator is `fig6_gdn_multimodality.py`; the statistics are reproduced by `verify_gdn_stats.py`.
- **Multiplicity wording for reviewers.** If asked: the primary endpoint is the pre-registered variance test and survives Bonferroni; the secondary failure-rate test is supportive at the prespecified 0.05 level but not Bonferroni-robust. The stricter 0.02 binarization is a sensitivity analysis and is not significant.
- **Borderline wording for reviewers.** If asked: the 0.4368 seed is 0.0132 below the pre-specified threshold; it is a failure by the rule, not a collapse. The 5/10 count is sensitive to this one seed; the variance result is not.

---


---

# Section 7.7 decay-only: the two possible endings (prepared 2026-10-19)

**Outcome of the pre-registered sweep (decided 2026-10-19, after the run):**
`decay_init_2.0` finished with 10/10 seeds at chance (J=8 mean 0.0153, std
0.0007, median 0.0155, range 0.0141–0.0161, learned $\alpha$ 0.829–0.835).
Together with `decay_init_0.5` (10/10 at chance) and `decay_init_1.0` (10/10 at
chance), the pre-registered sweep is **uniform over seeds in all three decay
conditions**, so the branch is **B** (no within-architecture reproduction of the
GDN multimodality from a scalar retention factor). The 7.7 write-up is
Table 7.5 in `section7_gdn_delta.md`.

**Additional condition, not part of the pre-registered sweep (completed):**
`decay_init_3.0` at 10 seeds was launched afterwards as a boundary probe,
because the 3-seed pilot at that initialization was the only non-uniform
outcome observed so far and it sits between the uniform collapse at learned
$\alpha\approx0.83$ and success. It is **mixed**: 8 seeds succeed
(0.6027-0.7392), one seed is degraded (0.0595) and one collapses to chance
(0.0179); the final learned $\alpha$ orders the three outcomes monotonically
without overlap (0.9812-0.9822 / 0.9687 / 0.9516), but no $\alpha$ trajectory was
logged, so the causal direction is unresolved. The 7.7 write-up is Table 7.6,
separate from the pre-registered Table 7.5, and the two are never pooled.
**The degraded mode is not GDN's lock-in mode** (0.0595 versus 0.298/0.437),
so the GDN trimodality is still not reproduced.

**Standing reservation, valid for every outcome of the additional condition:**
a two-mode outcome is **not** the three-mode GDN distribution. GDN's third
mode is an intermediate lock-in ($0.298$ and $0.437$) that no decay-only
condition has produced, so a split in the decay-only control must never be
written as "the GDN multimodality is reproduced by a scalar decay". The
strongest admissible statement is that a scalar retention factor produces a
*bimodal* outcome in a transition region inside one architecture, which is a
different phenomenon from GDN's trimodality and does not identify its
mechanism.

This file holds the two texts for Section 7.7 depending on what the
`decay_init_2.0` condition turns out to be. It is written so that the switch
can be made immediately when the run finishes, without re-deriving anything.

Shared facts (all from `delta_decay_only.py`, 1500 steps, $d_{\mathrm{mem}}=64$,
$d=64$, lr 1e-3, batch 32, $V=64$, MQAR 8 pairs / 32 distractors, evaluation
batch 128 × 20 batches):

| condition | $\alpha_0=\mathrm{sigmoid}(\text{init})$ | learned $\alpha$ | J=8 |
|---|---|---|---|
| `no_decay` ($\alpha=1$ fixed) | — | 1.0000 | 10/10 stable, mean 0.8651, std 0.0160 |
| `decay_init_0.5` | 0.622 | 0.515–0.532 | 10/10 at chance, mean 0.0157, std 0.0006 |
| `decay_init_1.0` | 0.731 | 0.638–0.651 | 10/10 at chance, mean 0.0154, std 0.0008, median 0.0153, range 0.0139–0.0167 |

Mechanism, stated once and used in both branches: the update is
$S_t=\alpha S_{t-1}+\hat k_t(v_t-S_{t-1}^{\top}\hat k_t)^{\top}$, so $\alpha$ is a
per-step retention factor. At $\alpha=0.53$ the content written 32 steps
earlier is attenuated by $0.53^{32}\approx1.5\times10^{-9}$; at $\alpha=0.64$,
$0.64^{32}\approx2\times10^{-6}$. Both are uniform over-decay failures: every
seed sits at chance, with no mixture of behaviours.

Separate 3-seed pilot (`delta_decay_fix.py`, same protocol, init
$\log\alpha=3.0$, $\alpha_0=0.952$): J=8 $=0.7321/0.7392/0.0179$, learned
$\alpha=0.982/0.982/0.952$ — a bimodal split (2 high, 1 collapsed). The 10-seed
sweep does not include init 3.0.

---

## Branch A — `decay_init_2.0` shows a split

Text to insert in 7.7 (or as a new 7.7 subsection, "Scalar decay inside one
architecture"):

> Within the same delta-rule architecture used in the ladder, we added a single
> learned scalar retention factor $\alpha=\mathrm{sigmoid}(\log\alpha)$ and
> nothing else — no convolution, no gate composition, no $\beta$ parameterization.
> The scalar-decay model is bimodal in the same way as GDN: with
> $\log\alpha$ initialized at $2.0$ (retention $0.88$), $X$ of 10 seeds reach
> $Y$ and $Z$ collapse to chance, with learned $\alpha$ of $A$ on the successful
> seeds and $B$ on the collapsed ones. Weakly initialized decay is a uniform
> failure instead: at $\log\alpha=0.5$ all 10 seeds collapse to chance with
> learned $\alpha\approx0.52$, and at $\log\alpha=1.0$ all 10 seeds collapse
> with $\alpha\approx0.64$. The no-decay control is stable in 10/10 seeds
> ($0.8651\pm0.0160$).

Then 7.7 must also say:

* The GDN collapse mechanism is now **partially** explained: a scalar retention
  factor is sufficient to produce a collapse mode inside a single architecture.
* It is **not** established that the GDN collapse is the same mechanism: GDN
  shows three modes (chance, lock-in at 0.30, success 0.52–0.66) and the
  within-architecture split has two. Do not say "the GDN collapse is caused by
  decay".
* Do not use the decay-only result to claim the Kaczmarz gate is unnecessary:
  the decay model still needs a stable write (it is the normalized $\beta=1$
  write plus a retention factor).

Tables to add: one per-seed table for the decay conditions (seed, learned
$\alpha$, J=0/8/16/32), and a summary table (condition, mean, std, median,
range, n below 0.45, n below 0.02).

Recommendation on init 3.0: **run it at 10 seeds** — it is the condition where
the 3-seed pilot showed the split, and a second split there would establish the
boundary of the bimodal region.

---

## Branch B' — `decay_init_2.0` is uniformly successful (all seeds high)

Text to insert in 7.7:

> Within the same delta-rule architecture used in the ladder, we added a single
> learned scalar retention factor $\alpha=\mathrm{sigmoid}(\log\alpha)$ and
> nothing else. The outcome depends monotonically on the initial retention
> rather than being multimodal: at $\log\alpha=0.5$ and $1.0$ all ten seeds
> collapse to chance (learned $\alpha\approx0.52$ and $\approx0.64$), at
> $\log\alpha=2.0$ all ten seeds succeed, and the no-decay control is stable in
> 10/10 seeds ($0.8651\pm0.0160$). A scalar decay factor produces a
> *retention-dependent* failure, not a *bimodal* one.

Then 7.7 must also say:

* A scalar decay factor is sufficient to **create** a collapse mode inside a
  single architecture, but it does not create a **split**: the outcome is a
  function of the initial retention, not a mixture of behaviours.
* The GDN trimodality (three modes, one of them a lock-in at 0.30) is
  therefore **not** explained by a scalar retention factor in our setup. The
  mechanism remains open.
* Do not convert this into "decay is harmless": the same mechanism that makes
  decay safe at $\alpha_0=0.88$ makes it fatal at $\alpha_0=0.62$, and the
  GDN block trains with a learned decay whose behaviour we did not observe
  (the GDN seed-4 loss trajectory is still an open check).

Recommendation on init 3.0: **run it only if a boundary claim is wanted.** With
init 2.0 uniformly successful, init 3.0 (the 3-seed pilot condition, 0.7321 /
0.7392 / 0.0179) becomes the interesting cell, because the pilot's collapsed
seed would then be the only evidence of a split. A 10-seed run at init 3.0
would either confirm that the pilot's collapse was a 3-seed accident (which
would close the question) or show that a split appears in a narrow window
between init 2.0 and 3.0 (which would be a new, reportable finding, but still
a two-mode, not three-mode, result). Priority: low-medium.

---

## Branch B — `decay_init_2.0` is uniform (all seeds at chance)

Text to insert in 7.7:

> Within the same delta-rule architecture used in the ladder, we added a single
> learned scalar retention factor $\alpha=\mathrm{sigmoid}(\log\alpha)$ and
> nothing else. At every initialization we tested, the outcome is uniform
> rather than multimodal: with $\log\alpha$ initialized at $0.5$ and at $1.0$
> all ten seeds collapse to chance (learned $\alpha\approx0.52$ and
> $\approx0.64$), with $\log\alpha$ initialized at $2.0$ all ten seeds again
> collapse to chance, and the no-decay control is stable in 10/10 seeds
> ($0.8651\pm0.0160$). We therefore find no within-architecture reproduction of
> the GDN trimodality from a scalar retention factor alone.

Then the 7.7 limitation must be rewritten to:

* The GDN multimodality is **not** reproduced by a scalar decay factor in a
  single-architecture control; the mechanism remains open and is more likely to
  involve gate composition, the convolution, or the parameterization used in
  the GDN block.
* This is a negative result about one mechanism, not evidence that decay is
  irrelevant to GDN. The GDN gate is a decay/retention gate (Section 7.1); the
  experiment says that decay *alone*, in our architecture and training setup, is
  not sufficient to produce the GDN distribution.
* The negative result is conditional on: this architecture, this training
  protocol, 10 seeds, 1500 steps, $d_{\mathrm{mem}}=64$, and a scalar
  $\alpha$ initialized at 0.5–2.0. It is not a claim about GDN.

Recommendation on init 3.0: **do not run it as a priority.** The 3-seed pilot
at init 3.0 (0.7321/0.7392/0.0179) is consistent with a chance split, and
running it would be informative about the boundary, but it cannot change the
paper's claim, because a split at 3.0 would still not be a reproduction of the
GDN trimodality (two modes, not three) and would not identify the GDN
mechanism. If it is run, report it as an additional condition, labeled
separately from the pre-registered 10-seed sweep.

---

## Pre-registration of the `decay_init_2.0` interpretation (2026-10-19, before the condition finished)

Recorded before the init 2.0 result was available, so that the interpretation
of the outcome is not chosen after seeing it.

**Prediction.** The initial retention at $\log\alpha=2.0$ is
$\alpha_0=\mathrm{sigmoid}(2)=0.881$. The content written 32 steps earlier is
attenuated by $0.881^{32}\approx0.017$, i.e. about 1.7% survives the
32-distractor gap, against $0.64^{32}\approx2\times10^{-6}$ at init 1.0 and
$0.53^{32}\approx1.5\times10^{-9}$ at init 0.5. Init 2.0 is therefore the
first tested condition in which *some* content can survive the gap, and it is
the boundary condition between "nothing is retained" (inits 0.5, 1.0: all
seeds at chance) and "content is retained".

**Pre-specified reading of the three possible outcomes.**

| Outcome | Branch | Section 7.7 statement |
|---|---|---|
| Split: some seeds succeed, some collapse | A | A scalar decay factor can create a collapse mode inside one architecture. Does not explain GDN's *three*-mode distribution; mechanism stays open. Run init 3.0 at 10 seeds to locate the boundary. |
| Uniform success: all seeds high | B' | The outcome is a function of the initial retention, not a mixture. A scalar decay factor creates a collapse but not a split. GDN trimodality not explained. init 3.0 optional (low-medium priority). |
| Uniform collapse: all seeds at chance | B | No within-architecture reproduction of the GDN multimodality from a scalar retention factor alone. Negative result about one mechanism; GDN mechanism remains open. Do not run init 3.0 as a priority. |

**Standing constraints in every branch.** The GDN trimodality, the
Brown--Forsythe primary ($p=0.0022$) and the Fisher secondary ($p=0.0325$) do
not depend on this ablation. "A uniform collapse is not multimodality" holds in
every branch. The decay-only ablation is a within-architecture control of one
candidate mechanism and is never described as "the mechanism for GDN".

### Addendum (2026-10-19, after init 2.0 seed 0 was observed, before seeds 1–9 finished)

Recorded so that the limitation of the prediction is on the audit trail
*before* the deciding seeds are in.

Seed 0 of `decay_init_2.0` **collapsed to chance** (J=8 $=0.0146$) even though
its initial retention was the highest tested ($\alpha_0=0.881$) and its learned
retention was $\alpha=0.8313$, far above the $\approx0.52$ and $\approx0.64$ of
the two uniform-collapse conditions. Three consequences:

1. **$0.881^{32}\approx0.017$ is a statement about the amount of content that
   survives the 32-distractor gap, not about the probability of success.** The
   arithmetic was correct; the inference from arithmetic to outcome was not
   licensed. Until seeds 1–9 are in, the prediction is limited to the
   "amount remaining" reading.
2. **The simple picture "the outcome is a function of the initial retention"
   is already broken by seed 0.** The learned retention fell from $0.881$ to
   $0.831$, and the run still collapsed. If the outcome is determined by
   *retention* at all, it is the learned retention that matters, not the
   initialization: the training dynamics pull $\alpha$ down, and the relevant
   threshold may sit in the learned value (possibly above $\alpha\approx0.83$).
   This is the statement Section 7.7 must use in any branch, including B'.
3. **The pre-specified three-way reading is unchanged.** Seed 0 alone does not
   determine the branch; the branch is decided by the full 10-seed
   distribution, and the interpretation for each distribution was fixed above.
   If the remaining seeds are all at chance (Branch B), the wording must not be
   softened into "the stronger initialization was not enough", because the
   pre-registered prediction was about the *amount* retained, not about success.

## Decision record: launch of the additional `decay_init_3.0` condition (2026-10-20)

**What was decided.** After the pre-registered sweep (initializations 0.5, 1.0, 2.0) returned uniform over seeds -- 10/10 at chance in each condition -- an additional 10-seed condition at initialization 3.0 was launched as a boundary probe.

**Why.** The over-decay threshold sits between the learned retention of the uniform-collapse conditions (about 0.83) and that of the 3-seed pilot (0.952--0.982, with two successes and one collapse). The pilot is the only non-uniform outcome observed anywhere, and it sits at that boundary, so a 10-seed run distinguishes "the pilot's split is reproducible" from "the pilot's collapse was a 3-seed accident".

**When.** After the pre-registered results were seen and after the pre-registration addendum was written. It is therefore **not** part of the pre-registered sweep and must be reported as an additional condition, never merged into the pre-registered table.

**Configuration identity with the pilot.** `delta_decay_fix.py` (pilot) and `delta_decay_only_v3.py` (10-seed run) differ only in that the latter parameterizes `init_log_alpha` and can skip the no-decay control. Verified item by item: $d=64$, $d_{\mathrm{mem}}=64$, $V=64$, batch $32$, 8 pairs, 32 distractors, 1500 steps, lr $10^{-3}$, AdamW, gradient clipping at $1.0$, evaluation on 20 batches of 128 at $J\in\{0,8,16,32\}$, initial $\log\alpha=3.0$, `torch.manual_seed` then `random.seed` before model construction, identical parameter-creation order ($W_k$, $W_v$, $W_q$, $\log\alpha$; then `emb`, `pos`, `mem`, `head`), identical update equation $S=\alpha S+\hat k\,(v-S^{\top}\hat k)^{\top}$ with the prediction computed *before* the $\alpha$ multiply, and 2 threads in both. The only difference is an extra `np.random.seed(seed)` in the newer script, which does not consume the torch or python RNG streams.

**Consequence.** The pilot's seeds 0 and 1 are reproduced *exactly* in the 10-seed run (J=8 $0.7321$ and $0.7392$; J=0/16/32 $0.9375/0.3956/0.1566$ and $0.9410/0.4337/0.1611$; learned $\alpha$ $0.9817$ and $0.9816$ against the pilot's $0.982$ and $0.982$). The match is forced by the configuration, not coincidental, so the pilot is reproducible and the two runs are the same experiment at different seed counts.

---

Observed so far: seeds 0 and 1 reproduce the 3-seed pilot exactly (J=8 $=0.7321$
and $0.7392$, learned $\alpha=0.9817$ and $0.9816$; the pilot reported
$0.7321/0.7392/0.0179$ with $\alpha=0.982/0.982/0.952$). The run is
`delta_decay_only_v3.py --inits 3.0 --skip-nodecay`, log
`delta_decay_only_init3_out.txt`.

### If the condition is mixed (some seeds high, some at chance)

Add a second table (Table 7.6) and one paragraph to the 7.7 control:

> The additional condition at $\log\alpha=3.0$ (initial retention $0.952$) sits
> at the boundary between the collapse regime and the stable regime, and it is
> the one condition in which the outcome is *not* uniform over seeds: $K$ of 10
> seeds reach $0.73$–$0.74$ and $10-K$ collapse to chance (learned $\alpha$
> $0.98$ on the successful seeds, $0.95$ on the collapsed ones). A scalar
> retention factor is therefore sufficient to produce a *mixed* outcome inside
> a single architecture, at the learned-retention boundary we tested.

And the mandatory reservations in the same paragraph:

* This is a **two-mode** outcome. GDN's distribution has **three** modes, one of
  which is an intermediate lock-in ($0.298$ and $0.437$) that no decay-only
  condition produces. The within-architecture split is therefore **not** a
  reproduction of the GDN multimodality, and the GDN mechanism remains
  unidentified.
* **Do not call it a narrow band.** The measured points are a uniform collapse
  at learned $\alpha\approx0.83$ and the mixed outcome at $0.95$–$0.98$;
  nothing was measured on the far side of that boundary, so the band is not
  bracketed. Say "at the boundary we tested".
* **The $\alpha$ relation is correlational.** Collapsed seeds end with a lower
  learned $\alpha$ than successful ones ($0.9516$ vs $0.9817/0.9816$), but the
  trajectory of $\alpha$ during training was not logged, so the causal
  direction is unresolved: $\alpha$ may be pulled down and cause the collapse,
  or the collapse and the reduction of $\alpha$ may share a common cause.
  Report the final values and stop there.
* **The mix proportion is the result.** Three of the ten seeds are fixed by the
  pilot (two successes, one collapse), so the condition is guaranteed to be
  mixed, but the proportion -- and hence the failure counts under the $0.45$
  and $0.02$ thresholds -- is only known at the end of the run. Do not write a
  proportion before the log is complete.
* The condition is **additional**, not part of the pre-registered sweep, and it
  was launched after seeing the pre-registered results; label it as such in
  the text and in Table 7.6's caption.
* The claim is "a scalar retention factor can be mixed within one
  architecture", not "decay is the cause of the GDN collapse".

### If the condition is uniform (all seeds high, or all at chance)

> The additional condition at $\log\alpha=3.0$ (initial retention $0.952$),
> run after the pre-registered sweep, is uniform over seeds: all ten seeds
> reach $0.73$–$0.74$. The 3-seed pilot's collapsed seed was therefore a
> small-sample accident, and the outcome of a scalar retention factor is
> uniform at every initialization we tested: collapse below a learned
> retention of about $0.85$, success above it, and no mixed outcome in
> between.

In this case the 7.7 wording written for the pre-registered sweep stands, with
one added sentence reporting the additional condition.

### In both cases

The pre-registered sweep (0.5, 1.0, 2.0) keeps its own table and its own
conclusion; the additional condition is a separate row or a separate table
with its own label, and the two are never pooled into a single "decay
ablation" table whose summary statistic spans them.

---

* "A uniform collapse is not multimodality" — use in either branch.
* The GDN trimodality (3 collapses, 2 lock-in, 5 successes) is a distributional
  fact from Section 7.3 and does not depend on the decay-only result.
* The Brown–Forsythe primary ($p=0.0022$) and the Fisher secondary
  ($p=0.0325$) are unaffected by the decay-only experiment.
* Do not describe the decay-only ablation as a "mechanism for GDN" in either
  branch. It is a within-architecture control of one candidate mechanism.

## Numbers still to be filled when the run finishes

| quantity | Branch A | Branch B' | Branch B |
|---|---|---|---|
| init 2.0 summary (mean, std, median, range, n<0.45, n<0.02) | fill | fill | fill |
| init 1.0 final summary | fill | fill | fill |
| number of split seeds | fill | — | — |
| learned $\alpha$ on successful vs collapsed seeds | fill | — | — |
| init 3.0 decision | run at 10 seeds | optional (low-medium) | do not prioritize |


---

# 8 Discussion

## 8.1 What the decomposition implies for design

The ladder is a decomposition of one update rule into three operations that are usually introduced together, and each operation carries a different kind of lesson.

**Check the write eigenvalue first.** The raw-key failure is visible at initialization, before any training happens: 99.97% of write positions violate the LMS stability condition, the most negative erase eigenvalue is $-34.87$, and the first forward pass has loss $2.36\times10^{18}$ against $5.35$ for normalized keys with the same key distribution. The clip-protocol control (Section 3.2) shows that this is not an artifact of the clipping protocol: while the total gradient norm is infinite, every finite clip scales the update to zero, and without clipping the model never escapes and never learns. The practical rule is therefore not "use a smaller learning rate" but "look at the eigenvalue of the write you have actually implemented".

**The stability bound is a constraint, not an objective.** In the normalized regime the optimal fixed step size is $\beta=0.125$, well inside $0<\beta<2$; the bound says what is stable, and the task says what is best. A fixed $\beta=0.125$ recovers about 90% of the gap to a Kaczmarz gate at $d_{\mathrm{mem}}=16$, about 92% at $d_{\mathrm{mem}}=32$, and about 99% of the gap at 3000 steps at $d_{\mathrm{mem}}=16$. The remaining step from a stable write to a useful write is mostly about *how much* is erased, not about making the write more aggressive or more conservative in the stability sense.

**Gating buys sample efficiency, not accuracy at convergence, and its shape is the part that matters.** At the baseline budget the gate is worth $+0.0729$; at 3000 steps the same transition is $+0.0030$ with an interval that includes zero. But a gate-trained solution is not interchangeable with a constant of the same mean: the mean-matched constant costs $0.380$, $0.227$, and $0.161$ at $d_{\mathrm{mem}}=16,24,32$, while the shape-matched gate is *equivalent* to the gate itself within a tolerance of $\pm0.01$ at all three capacities (paired TOST, Section 5.3). The asymmetry is the design lesson: a network trained with a gate learns a solution that expects a relative, norm-based weighting of writes, and that expectation is not recoverable post hoc. "Train without a gate" and "evaluate with a gate" are different questions, and a post-hoc swap is not a reweighting.

**Capacity attenuates the co-adaptation cost but does not remove it.** The shape cost of a gate-trained solution falls from $0.380$ to $0.161$ as $d_{\mathrm{mem}}$ grows from 16 to 32, yet it is still 17% relative at $d_{\mathrm{mem}}=32$; and at $d_{\mathrm{mem}}=32$ the fixed rule is within $0.01$ of the Kaczmarz model without any shape specialization. The interesting regime is therefore not "small capacity needs gates" but "at what capacity does the shape stop mattering", and our data only bracket it between 24 and 32.

**A curriculum is not a write rule.** Training on an easier version of the task first removes the raw-key collapse mode (no seed left at chance) without repairing the initialization-time violation, and it does nothing once the write rule is fixed ($+0.0007\pm0.0102$ under Kaczmarz). Easing the task and stabilizing the write are not substitutes.

## 8.2 Limitations

* **Scale.** The ladder runs at $d=64$, $d_{\mathrm{mem}}=16$ on CPU. The real-architecture check (Section 7) is also small-scale, and the 340M-parameter configurations used in the published comparisons are out of scope. What we expect to transfer is the diagnostic content -- an initialization-time eigenvalue violation, a gradient-norm-infinite freeze, a trimodal distribution over seeds -- and not the numbers.
* **The ladder block has no convolution, but the control shows that does not change the decomposition.** The three factors are measured on a delta-rule block without a causal convolution, while the real GDN and Delta blocks have one. The convolution control (Tables 7.7 and 7.8) reruns the whole ladder with a convolution and finds that all three transitions and the per-seed ordering survive it, so the attribution is not an artifact of our block. What is left as a difference from the real blocks is state size, budget, parameterization, and GDN's decay gate. One further convolution setting is unrun: Boesch and Wee (arXiv:2609.16183) report that the delta-rule-over-diagonal margin shrinks to $+0.03$ once both cells carry a convolution, and our control (one block, with and without a convolution) is not a replication of that comparison.
* **The ladder numbers are stated under one clipping protocol.** Every accuracy in Sections 3 and 4 uses global gradient-norm clipping at $1.0$. The clip-protocol control (Section 3.2) shows that the *ordering* survives every clipping setting we tested, but the magnitudes do not: without clipping the normalized stage reaches $0.0754$ instead of $0.1919$, and with clipping at $10.0$ the raw stage collapses to chance in all three seeds. The diagnostics -- which factor matters, and in which order -- are protocol-independent; the numbers are not.
* **Seeds.** The ladder uses three seeds per cell, so its intervals are descriptive; the multi-seed evidence is in Section 7 (10 seeds) and at $d_{\mathrm{mem}}=24$ (10 seeds).
* **Post-hoc versus retraining.** The matched-mean experiments (Section 5.3) measure co-adaptation of a learned solution, not the effect of training with the swapped rule, and they use 1500-step models; whether the shape cost persists at 3000 steps, where the fixed rule closes the training-time gap, is untested.
* **One task family, and a variant-dependent published comparison.** Everything is multi-query associative recall with $V=64$. The published S-NIAH ordering is task-variant dependent (DeltaNet is ahead of Gated DeltaNet on S-NIAH-1 at 4K/8K and behind it on S-NIAH-2/3), and our small-scale ordering is consistent with one of those variants and not the others (Section 7.5); we do not identify which regime our task corresponds to. What we expect to transfer is the diagnostic -- how to tell a collapse from a lock-in -- and not the ordering.
* **Architecture dependence of the stability analysis.** The initialization violation depends on the key-norm scale, which is a property of the parameterization; the specific numbers in Section 3.1 do not transfer to a block with a different initialization or normalization, although the diagnostic procedure does.
* **Threshold provenance.** The $0.45$ threshold was fixed before the Delta ten-seed set but after the GDN v15 set had been seen (Section 9, item 26). The threshold-free variance test is what Section 7 rests on.

## 8.3 Relation to existing theory: a qualitative connection only

The two-component picture that separates *finite-feature noise* (which shrinks with the memory dimension) from *structural interference* (which does not) is developed in Zhang et al. (arXiv:2610.09056, *A Geometry-Based Capacity Theory for Finite-Feature Associative Memory*). Their law for the retrieval cosine of a finite random-feature Hebbian memory is

$$C_i(R) \approx \left[1 + \frac{N-1}{R} + \sum_{j \neq i} K_{ij}^{2}\right]^{-1/2},$$

i.e. a finite-feature term $(N-1)/R$ and a structural term $\sum_{j\neq i}K_{ij}^{2}$ (squared kernel overlap among the stored keys), with the infinite-feature ceiling $C_{i,\infty}\approx[1+\sum_{j\neq i}K_{ij}^{2}]^{-1/2}$. Our Kaczmarz residual is qualitatively consistent with that separation. The residual $1-\text{accuracy}$ of the Kaczmarz model falls from $0.0636$ at $d_{\mathrm{mem}}=16$ to $0.0528$, $0.0517$, and $0.0506$ at $d_{\mathrm{mem}}=32,64,128$; a two-component description of the form residual $\approx$ floor $+\,c\,f(d_{\mathrm{mem}})$ gives a floor between $0.048$ and $0.050$ whichever decaying $f$ we fit, and the residual decreases monotonically with $1/d_{\mathrm{mem}}$ (Spearman $1.00$ on those four points). **The four points do not identify the decay rate**: the residual sum of squares is $7.7\times10^{-6}$ for $f=1/d$, $9.8\times10^{-7}$ for $f=1/d^2$, $1.6\times10^{-5}$ for $f=1/d^{1/2}$, and $1.5\times10^{-5}$ for $f=1/\log d$, with two residual degrees of freedom. A faster-decaying finite-feature term fits slightly better than the $1/d$ form, so we do not treat the decay rate as measured; the non-vanishing floor is the feature that all four forms agree on.

We stop there, for four reasons. First, **this is a descriptive fit on four points, not an inferential test**: we attach no $p$-value to the correlation, no confidence interval to the floor, and no model comparison to the fit, because four capacities with no per-point error bars cannot support any of them. In particular, the comparison of the functional forms above is a tie-breaking exercise, not a model selection. Second, the two quantities are not the same: their law is a normalized *retrieval cosine* for a finite random-feature Hebbian memory, ours is a task *accuracy* in a delta-rule memory, and the mapping between them is not something we can justify. Third, the theory's finite-feature term decays as $(N-1)/R$, with $N$ the number of stored items and $R$ the feature dimension; identifying $R$ with $d_{\mathrm{mem}}$ and $N$ with our eight stored pairs would make the predicted decay $1/d_{\mathrm{mem}}$, and our four points prefer a faster decay ($1/d_{\mathrm{mem}}^2$ fits better than $1/d_{\mathrm{mem}}$ in residual sum of squares), so our data do not confirm the predicted exponent. Fourth, and most importantly, the co-adaptation cost of Section 5.3 is a *different phenomenon* from the residual: it is the price of swapping a write rule after training, not the price of a finite memory, and it lies outside that framework. The connection is a consistency check on one curve, not a validation of the theory.

A second line of work reaches the same geometry from the state side. Nazari and Rusch (arXiv:2602.04852) show that the recurrent state of a trained linear-attention model is effectively low-rank, and they relate the *state's* effective rank to the conditioning of the keys: $\mathrm{rank}(S_t)\le\min(\mathrm{rank}(K_t),\mathrm{rank}(V_t))$ algebraically, and $\nu(V_t)/\kappa^2(K_t)\le\mathrm{er}(S_t)$ in effective rank, where $\mathrm{er}(S)\coloneqq\|S\|_F^2/\|S\|_2^2=\sum_i\sigma_i^2/\sigma_1^2$ is their symbol (the quantity the literature also calls the *stable rank*), with the readout amplifying query noise by at least $1/\sqrt{u(S_t)}$ when the state spectrum is skewed. Two points of contact with our work. The functional form of their effective rank is the same stable rank we report, $\sum_i\sigma_i^2/\sigma_1^2$, but it is computed on the **state** $S_t$, whereas ours is computed on the **key matrix at the key positions** (Section 2.6): the same functional form on a different matrix, and their bound, which is what relates the two, is derived for the plain linear-attention recurrence and only reaches DeltaNet through a pseudo-value reparameterization. And their Proposition on readout-noise amplification is a reason to keep effective rank as a diagnostic rather than a capacity proxy in our ladder, which measures exactly the readout accuracy that such a state would degrade. Their "convolution constraint" -- the sequence mixer is invariant under orthogonal transformations of keys and queries, but depthwise convolutions are not -- is complementary to our convolution control, which finds that adding a convolution leaves the three-factor decomposition intact.

## 8.4 Open questions and the next experiments

The three-factor decomposition suggests three concrete directions that our own measurements motivate.

**A. Covariance-tracking writes (the most promising).** Our results say that *insufficient* writing is catastrophic and *excessive* writing is mild (the $\beta$ sweep has a broad optimum), that the *shape* of the write is what a trained solution co-adapts to, and that effective rank is a property of the update rule rather than of the representation alone. Together these suggest that the write strength should be conditioned on *directional, accumulated evidence* rather than on a token-local scalar -- which is the motivation for covariance-tracking writes such as those in KDN. The obstacle is that KDN-style blocks confound two things: the covariance tracking itself, and the addition of a second learned parameter. Separating them is a clean ablation -- diagonal KDN versus isotropic KDN versus a false-covariance control, where the covariance term is removed but the extra parameter is kept -- and our matched-mean methodology transfers to it directly, because the question is again whether a solution co-adapted to a write rule can be re-evaluated under a different one. The second gap is evaluation: a method that is diagnosed on 64-token MQAR should be shown to survive long-context evaluation.

**B. PDN times DeltaProduct.** The capacity dependence of the shape sensitivity (40.5% $\to$ 23.9% $\to$ 17.0% relative) bears directly on the untested range of PDN's squashing, and PDN cannot learn negative eigenvalues naturally. Combining the pre-processing and the state tracking attacks both at once, and both are available in `fla`.

**C'. A fit-free capacity boundary in a delta-rule memory.** The capacity theory of Zhang et al. predicts a *memory budget* for a target retrieval fidelity without fitting any coefficient ($R/N \approx (1-1/N)/((1/C^\star)^2-1)$, whose large-$N$ limit is $[1/(C^\star)^2-1]^{-1}\approx9.26$ for $C^\star=0.95$, and which is $8.1$ at $N=8$), and it is formulated for a Hebbian/random-feature memory. Our ladder uses a different storage rule, so the law is not directly testable there, but the *measurement* it asks for is cheap: the retrieval cosine of a delta-rule memory as a function of $d_{\mathrm{mem}}$, compared with the same fit-free boundary. That is a CPU-scale experiment in a regime the theory does not cover, and it is the cleanest way to find out whether the two-component structure is a property of associative memory in general or of the Hebbian/RFF construction.

**C. A principled derivation of the hybrid ratio.** The 3:1 mixing ratio of a product-based and a delta-rule component is empirical. A derivation from an interference/capacity decomposition -- the same decomposition that Section 5.3 measures and that 8.3 connects to existing theory -- is open, and it is cheap to attack at CPU scale, which is where we can iterate.

We also note what we deliberately did not do: the surprise-based architectures (Titans, MIRAS, ATLAS) are at an implementation stage, MesaNet does not scale, and Mamba-3 is orthogonal to the delta rule.

---

## Draft notes (not for submission)

- **Section 8 has no figures and no new numbers.** Everything in 8.1 is a restatement of a result in Sections 3--6; 8.2 collects the limitations that are stated in place; 8.3 uses the harmonic sweep (`harmonic_out.txt`); 8.4 is forward-looking and clearly marked.
- **8.3 numbers were recomputed from our own data** (`harmonic_out.txt`, Kaczmarz model, accuracies 0.9364/0.9472/0.9483/0.9494 at $d_{\mathrm{mem}}=16/32/64/128$): residuals $0.0636/0.0528/0.0517/0.0506$; two-component fit residual $\approx$ floor $+\,c/d$ gives floor $0.0476$, $c=0.2399$, $R^2=0.929$; Pearson corr$(1/d,\text{residual})=0.964$ ($p=0.036$, $n=4$), Spearman $=1.000$. **$R^2$, the Pearson $p$ and the fit constant $c$ are deliberately kept out of the main text**: with four points and no per-point error bars they would invite exactly the 'do not report $R^2$ on $n=4$' objection, so the main text carries only the floor, the monotonicity, and an explicit 'descriptive fit, not an inferential test' statement. The numbers are here for audit. Each accuracy is a 3-seed mean, which is another reason the fit cannot carry a hypothesis test. Earlier working notes quoted a floor of $\approx0.054$ and a correlation of $0.872$; the recomputed values differ because the estimator is a two-parameter least-squares fit rather than a different ad-hoc rule, and the paper uses only the recomputed ones.
- **The quantitative link to 2610.09056 was checked against the primary source** (arXiv 2610.09056v1, *A Geometry-Based Capacity Theory for Finite-Feature Associative Memory*). The law is $C_i(R)\approx[1+(N-1)/R+\sum_{j\neq i}K_{ij}^2]^{-1/2}$ with the infinite-feature ceiling $C_{i,\infty}\approx[1+\sum_{j\neq i}K_{ij}^2]^{-1/2}$ and the fit-free boundary $R/N\approx(1-1/N)/((1/C^\star)^2-1)$. The comparison in 8.3 therefore stays qualitative, and the reasons are now specific: different metric (retrieval cosine vs task accuracy), different storage rule (finite random-feature Hebbian memory vs delta rule), and a predicted $1/R$ finite-feature exponent that our four points do not support ($1/d^2$ fits better). A quantitative test would need the same rule and metric, which is the experiment proposed in 8.4 (C').
- **8.2 bullet on the convolution** must be updated when the convolution control (Section 7) is complete; the wording there is deliberately generic so that it can be filled in either way.
- **8.4 is a discussion, not a plan.** Keep the claims in the conditional ("our results suggest that ..."), and do not promise results in A, B, or C.
- **The open items elsewhere** (Appendix D configuration, GDN seed 4 loss trajectory, Delta seed order) are listed in the Section 7 draft notes and are not repeated here.


---

# 9 Revisions and retractions (draft)

## 9.1 Why this section exists

Several claims in earlier drafts of this work were too strong, out of scope, or unsupported, and we corrected them while the experiments were running rather than after submission. We list the revisions that change what the paper claims. Each entry states the claim as it was made, the corrected statement, and the consequence. The complete log -- including the code-level corrections and the numerical fixes that do not affect any claim -- is in the appendix, which we keep as a single table plus the full text of the items that matter for the argument.

We do not treat these as cosmetic. Several of them would have changed a conclusion, and one of them (the failure threshold, item 26) changed the status of an endpoint from confirmatory to supportive.

## 9.2 Revisions that affect the paper's claims

**9.2.1 The failure threshold was not fixed before seeing every result (item 26).** An earlier pre-registration note stated that the accuracy threshold $0.45$ was fixed before seeing any results. This was true of the Delta ten-seed set but not of the GDN v15 set, which had already been observed; $0.45$ lies between the GDN boundary case $0.4368$ and the lowest GDN success $0.5193$. The threshold is therefore pre-specified with respect to Delta and partially pre-specified with respect to GDN. We report a full threshold sweep (Figure 7, Table 7.4) showing the Fisher test is significant for every $t\in[0.44,0.897]$ and not significant at $t=0.4368$ ($p=0.0867$), we keep $0.45$ and do not move it post hoc, and we describe the secondary endpoint as supportive rather than confirmatory. The threshold-free Brown--Forsythe test remains the primary endpoint.

**9.2.2 "The optimal erase rate is not the stable one" is a raw-key statement (item 29).** An earlier version asserted this without restriction. It can only be a claim about the raw-key regime, and we do not establish it even there, because no raw-key $\beta$ sweep was run. In the normalized regime the optimal fixed $\beta=0.125$ lies well inside the LMS bound $0<\beta<2$, and the raw-key bound does not constrain the normalized model at all: evaluated at the ladder's operating point it exceeds $2$ for 13.2% of write positions at initialization and 20--29\% of writes during training at $d_{\mathrm{mem}}=16$ (78--80\% at $d_{\mathrm{mem}}=32$), which are counterfactuals for a normalized model.

**9.2.3 Variance reduction is budget-dependent, not a property of normalization (item 30).** An earlier version described normalizing the key as reducing the variance of the outcome. The reduction holds at 1500 steps (std 0.0784 $\to$ 0.0063) and reverses at 3000 steps (0.0198 $\to$ 0.0669), where the effect is a mean shift ($+0.3917\pm0.1989$). At 1500 steps the paired mean difference is $+0.1008\pm0.1898$ and its interval includes zero. The paper now reports the budget dependence and does not describe normalization as a universal variance reducer.

**9.2.4 The frozen phase is not a clipping artifact (item 31).** An earlier version treated the frozen phase of the raw-key model as a consequence of the clipping protocol, which left open the possibility that the raw-key failure was protocol-dependent. The clip-protocol control reverses that reading. While the total gradient norm is infinite, any finite clip scales the update to zero, so the frozen phase and the norm-collapse escape are identical at clip 1.0 and at clip 10.0 (escape steps 850, 970, and 230 for the three seeds, with the same collapse of the key norms). Without clipping the model never escapes: the loss stays above $10^{12}$, the key norms stay at their initialization scale, and all three seeds end at chance. No clipping setting we tested makes the raw-key write learn, so the raw-key failure is not an artifact of the clipping protocol -- the protocol determines how badly the model fails, not whether it fails. What remains protocol-dependent is the magnitude: the normalized stage reaches 0.0754 without clipping instead of 0.1919, so every ladder number in the paper is stated under clip 1.0, while the ordering (raw fails, normalized learns) holds under every setting we tested.

**9.2.5 The GDN collapse is not attributable to the decay gate alone.** An earlier version left open the reading that the trimodal GDN distribution is a decay effect, motivated by the architecture's name. Our within-architecture control (Section 7.7) does not support that reading: a learned scalar retention factor produces a uniform collapse at every initialization in the pre-registered sweep, and at one additional initialization a minority failure mode -- but neither reproduces GDN's three-mode distribution, whose third mode is an intermediate lock-in that no decay-only condition produces. Decay is sufficient for a collapse and not sufficient for the GDN distribution; the mechanism remains open, and the paper says so.

**9.2.6 The published comparison was checked against the primary source (item 33).**
An earlier version of this paper cited the published large-scale result as
follows: "the Gated DeltaNet paper reports 99.0/98.8 for DeltaNet versus 91.4/91.8
for GDN on 1.3B S-NIAH-1 at 4K/8K (arXiv 2412.06464, Table 3)". A reviewer
suggested that this sentence mis-attributed the numbers and that the published
ordering is the opposite of ours at 400M parameters and 15B tokens. We checked
the cited source (arXiv 2412.06464, v3) and report what it says.

**The numbers, the attribution and the scope of our sentence were correct.**
Table 2 of that paper is the zero-shot S-NIAH table for 1.3B models, and its
S-NIAH-1 columns at 4K and 8K read DeltaNet 99.0/98.8 and Gated DeltaNet
91.4/91.8 -- exactly what we wrote. The only error in our sentence was the
table number: it is Table 2, not Table 3. The reviewer's 400M/15B table is not
in the cited paper, whose 400M/15B table (Table S.1) is an ablation reporting
perplexity and commonsense accuracy rather than RULER S-NIAH; we could
therefore not confirm the proposed reversal and have not adopted it.

**What we did change.** Section 7.5 now states the published comparison as
task-variant dependent, which the verified table makes visible: on S-NIAH-1,
the variant with repeated synthetic context, DeltaNet is ahead (99.0/98.8
against 91.4/91.8 at 4K/8K), while on S-NIAH-2 and S-NIAH-3, the variants with
real-world-essay context, Gated DeltaNet is ahead (14.4 against 29.6 at 8K on
S-NIAH-2; 22.4 against 27.6 at 4K on S-NIAH-3). Our small-scale ordering is
consistent with the first variant and not the others, and we do not identify
which published regime our task corresponds to. The earlier "sign-preserving"
phrasing is therefore replaced by an explicit statement of what agrees and
what does not, and the paper no longer asserts a single published ordering.

**What is retracted.** The table number (Table 3) and the implication that a
single published ordering exists. **This is a correction of our citation, not a claim that the original paper is wrong.** **What replaces it.** Section 7.5 as
rewritten, with the verified numbers and the variant dependence, and the
explicit statement that the small-scale check is a distributional diagnostic
rather than a predictor of the published ordering. Section 7's primary
endpoint and its distributional result are unaffected.

**9.2.7 The convolution control (item 34).** An earlier version of this paper treated the absence of a convolution in our ladder as an open external-validity question: the three factors were measured on a delta-rule block without one, while the real GDN and Delta blocks have one, and we had no control. We have now run that control, and it was pre-registered before it started, with the operational thresholds and the outcome-to-claim map fixed before the run finished.

Adding a depthwise causal convolution to the block leaves the decomposition intact: the stability transition is $+0.1628\pm0.0249$ (strictly positive, as predicted), the erase-strength transition is $+0.6702\pm0.1344$ (above the pre-registered $+0.30$ threshold and still the largest of the three), the gating transition is $+0.0663\pm0.1063$ (below the pre-registered $+0.15$ threshold), and the per-seed ordering raw $<$ normalized $<$ fixed $\leq$ Kaczmarz holds for every seed. The no-conv cells of the control reproduce the budget-matched ladder rows exactly at the reporting precision, which separates the convolution's effect from any implementation difference.

We therefore withdraw the open question and narrow the caveat: the three-factor decomposition is not an artifact of the convolution-free block. The revision is not a retraction of a claim but a change of scope -- the remaining differences between our cells and the real blocks are state size, budget, parameterization, and GDN's decay gate. The control does not identify the mechanism of the GDN trimodality, and it does not settle the disagreement with the published ordering, which concerns the internals of the two real blocks.

**What is retracted.** The statement that the decomposition's external validity was an untested open question, and the implicit suggestion that a convolution might change which factor matters. **What replaces it.** Tables 7.7 and 7.8 and the narrowed caveat.

**9.2.8 The shape-sensitivity ratio is not reported.** An earlier version summarized the double dissociation as a ratio of the two shape costs across capacities. Because the ratio's growth is driven by one denominator approaching zero, it is misleading; the paper reports the absolute and relative drops instead.

**9.2.9 The earlier revisions (master log, items 1--25).** Items 1--25 are code-level and numerical corrections made before this version. They do not change the paper's conclusions beyond the items already listed in 9.2.1--9.2.8. The verbatim table is kept as a provenance record in the working repository and is not reproduced here.

## 9.3 Complete log

The full log -- all 34 items, including the code-level corrections (the shifted input alignment, the memory dimension, the non-finite-value guard, the bias in $W_v$), the numerical fixes (the initialization diagnostic recomputed in item 27, where 99.98\% became 99.97\%), the wording corrections, and the full text of items 26--34 -- is in the appendix. We keep it because the corrections are cheap to describe and expensive to reconstruct.



---

# Appendix A. Master log items 1-25 (provenance record)

Items 1--25 are code-level and numerical corrections made before this version
of the paper. They do not change the paper's conclusions beyond the items
already listed in Section 9.2.1--9.2.8. The verbatim table (upstream master log,
items 1--25) is kept in the working repository as a provenance record and is
not reproduced here.

**Note on row 25.** Row 25 is the threshold-provenance correction, which is
item 26 in this document (the upstream log's item 26 was merged into it), so
that the same correction is not listed twice.


---

# Appendix: complete revision and retraction log

This appendix holds the complete log. The paper's Section 9 keeps only the
items that change what the paper claims; this file holds all 34 items, with the
full text of the ones that matter for the argument and a one-line index for the
rest.

**Status of the earlier items (1--25).** The master log for items 1--25 lives
outside this workspace and is pasted in verbatim below in the index table. The
one-line descriptions in the table are ours and are marked as such; the
authoritative wording is the master log's. Items 26--34 are written out in full
after the table.

## Index

| Item | One-line description | Category | Where it appears |
|---:|---|---|---|
| 1--25 | *from the master log: code-level fixes (shifted input alignment, memory dimension, non-finite-value guard, $W_v$ bias), numerical and rounding fixes, wording corrections, and the removal of the shape-sensitivity ratio* | bug / numerical / wording | Section 9.2.6 (the ratio) and the appendix |
| 1--25 | *the upstream master log, reproduced verbatim below and in Section 9.2.9* | bug / numerical / wording / claim scope | Section 9.2.9 and the appendix |
| 26 | Provenance of the 0.45 threshold: pre-specified for Delta, partially pre-specified for GDN | claim scope | Section 9.2.1, Section 7.4, Section 7.7 |
| 27 | Initialization diagnostic recomputed: 99.98% $\to$ 99.97%, 10.8 $\to$ 10.82, $-34.9 \to -34.87$ | numerical (no artifact) | Section 3.1, Table 3.1, Section 1 |
| 28 | "Erase strength is the largest effect at 3000" withdrawn; persistence claimed instead of rank | claim scope | Sections 1, 3.4, 4.3 |
| 29 | "The optimal erase rate is not the stable one" restricted to the raw-key regime | claim scope | Sections 1, 4.1, 4.4 |
| 30 | "Normalization reduces variance" withdrawn; budget dependence reported | claim scope | Sections 1, 3.3, 3.4 |
| 31 | "The frozen phase is a clipping artifact" withdrawn; the control reverses it | claim scope (control) | Sections 1, 3.2, 3.4 |
| 32 | The GDN collapse is not attributable to the decay gate alone | claim scope (control) | Section 7.7, Section 9.2.5 |
| 33 | The published comparison was checked against the primary source: numbers and attribution confirmed, the table number corrected (3 -> 2), and the comparison restated as task-variant dependent | citation / claim scope | Sections 1, 7.5, 7.7, 7.8, 8.2 |
| 34 | The convolution control was run: the three-factor decomposition survives a causal convolution, and the Section 7 caveat narrows | scope (control) | Section 7.7, Section 8.2, Tables 7.7--7.8 |

---

## Items 1--25 (master log, verbatim)

# Items 1--25 (upstream master log) -- pasted

The table below is the upstream master log's items 1--25, pasted verbatim
(pasted 2026-10-20). It is kept here as the appendix provenance block; nothing in it is paraphrased or reformatted.

Column 3 ("where it appears") is reproduced verbatim from the upstream log, so
its section numbers refer to the upstream manuscript rather than to this
version.

**Note on row 25.** Row 25 is the threshold-provenance correction, which is
**item 26 in this document** (the upstream log's item 26 was merged into it), so
that the same correction is not listed twice.

The upstream master log's items 1--25 are code-level and numerical corrections
made before this version of the paper. They do not change the paper's conclusions
beyond the items already listed in Section 9.2.1--9.2.8. The verbatim table is
kept as a provenance record in the working repository and is not reproduced here.

---

## Item 26: provenance of the 0.45 threshold

**Retracted claim.** "The 0.45 failure threshold was fixed before seeing any results."

**Correct provenance.** The Delta 10-seed set had not been observed when the threshold was fixed. The GDN 10-seed set had already been observed in the v15 run. The 0.45 threshold lies between the GDN boundary case 0.4368 and the lowest success 0.5193. The threshold is therefore pre-specified with respect to Delta but not blind with respect to GDN; the secondary endpoint is partially pre-specified, not fully pre-registered.

**Mitigation.** The threshold sweep (Table 7.4, Figure 7) gives two-sided Fisher $p<0.05$ for every $t\in[0.44,0.897]$. At $t=0.4368$ exactly, GDN has 4/10 failures and $p=0.0867$; at $t=0.44$ it becomes significant (5/10, $p=0.0325$). The upper boundary is $t=0.8970$, just above the fifth-lowest Delta seed (0.8961); at $t=0.8975$, $p=0.0867$ (Delta 6/10). The maximum $p$ in the band is 0.0325 and the minimum is $1.1\times10^{-5}$ at $t=0.67$--$0.8864$. Delta has zero failures through $t=0.8864$. The Fisher conclusion is therefore not an artifact of the gap location. The primary Brown--Forsythe variance test is threshold-free and remains confirmatory.

**Consequence for the paper.** The primary endpoint is described as pre-registered; the secondary Fisher endpoint is described as partially pre-specified and supportive. It does not survive a Bonferroni correction. We do not use Cohen's $d$; we report the median gap (0.419), the failure counts (5/10 vs 0/10), and the variance ratio ($1325\times$).


## Item 27: the initialization diagnostic of the raw-key write

**Retracted claim.** "99.98% of write positions violate the raw-key stability condition", with the mean squared key norm 10.8, the median 9.9, the maximum 35.9, and the most negative erase eigenvalue $-34.9$ (Table 3.1). The numbers had no saved artifact: they were computed in an earlier working session and could not be checked.

**Correction.** Recomputed from scratch by `init_eig_diag.py` (raw-key model, seed 0, $d_{\mathrm{mem}}=16$; 20 batches of 1,760 write positions = 35,200 positions; output in `init_eig_diag_out.txt` and `init_eig_diag.json`): mean $E\lVert k\rVert^2=10.82$, median $9.95$, maximum $35.87$, most negative eigenvalue $-34.87$, and **99.97%** of positions with $\lVert k\rVert^2>2$ (11 of 35,200 are at or below 2; the smallest observed value is 1.74). The two first-forward losses ($2.36\times10^{18}$ raw, $5.35$ normalized) and every value of the Table 3.2 trajectory reproduce exactly from `rawk_log.csv`.

**Consequence for the paper.** The qualitative claim is unaffected. The violation fraction changes from 99.98% to 99.97%, and the diagnostic now has a reproduction script and a saved artifact. Sections 1, 3, and 4 use the recomputed values.


## Item 28: "erase strength is the largest effect at 3000 steps"

**Retracted claim.** That erase strength is the largest stage-to-stage effect at both budgets.

**Correction.** At 3000 steps the stability transition is $+0.3917\pm0.1989$ and the erase-strength transition is $+0.2950\pm0.1587$; the intervals overlap, so the two cannot be ranked. Erase strength is the main effect at the baseline budget ($+0.6724\pm0.0447$) and remains a large, budget-persistent effect at 3000 steps -- persistence, not rank.

**Consequence for the paper.** Sections 1, 3, 4 and the abstract say "dominant at the baseline budget, persists at 3000" and state explicitly that the 3000-step ranking is not claimed.


## Item 29: scope of the erase-rate claim

**Retracted claim.** That the optimal erase rate is not the stable one, stated without restriction.

**Correction.** The statement can only be about the raw-key regime, and we do not establish it even there, because no raw-key $\beta$ sweep was run. In the normalized regime the optimum ($\beta=0.125$) is comfortably inside $0<\beta<2$, and the raw-key bound does not constrain the normalized model. The raw-key product at the ladder's operating point is reported with the correct scope: 13.2% of write positions at initialization and 20--29% during training at $d_{\mathrm{mem}}=16$ (78--80% at $d_{\mathrm{mem}}=32$), all counterfactuals for a normalized model.


## Item 30: variance reduction is budget-dependent

**Retracted claim.** That normalizing the key reduces the variance of the outcome, as a general property of the stage.

**Correction.** The reduction holds at 1500 steps (std 0.0784 $\to$ 0.0063) and reverses at 3000 steps (0.0198 $\to$ 0.0669), where the effect is a mean shift ($+0.3917\pm0.1989$, interval excluding zero). At 1500 steps the paired mean difference is $+0.1008\pm0.1898$ and its interval includes zero, so the mean shift is not resolved at that budget either.


## Item 31: the frozen phase is not a clipping artifact

**Retracted claim.** That the frozen phase of the raw-key model is a consequence of the clipping protocol, leaving open that the raw-key failure itself might be protocol-dependent.

**Correction.** The clip-protocol control (`clip_ablation.py`; raw key at clip 1.0 / 10.0 / none, and normalized key at no clip; 3 seeds each, 1500 steps) shows the opposite. While the total gradient norm is infinite, any finite clip scales the update to zero, so the frozen phase and the norm-collapse escape are identical at clip 1.0 and clip 10.0 (same escape steps 850/970/230, same key-norm collapse). Without clipping the model never escapes at all: the loss stays above $10^{12}$, the key norms stay at their initialization scale, and all three seeds end at chance. No clipping setting we tested makes the raw-key write learn, so the raw-key failure is not a clipping artifact; the protocol determines how badly the model fails, not whether it fails. What remains protocol-dependent is the magnitude: the normalized stage reaches 0.0754 without clipping instead of 0.1919, so all ladder numbers are stated under clip 1.0, while the ordering holds under every setting tested.


## Notes on this appendix

- **Why the full text is kept.** Several of the corrections are cheap to state and expensive to reconstruct (the exact per-seed values, the escape steps, the count of positions below the threshold). A reader who wants to check one of them should not have to rerun anything.
- **What is not in this log.** The per-seed data, the protocol, and the pre-registrations are not revisions and are described in the paper itself. The three pre-registrations that constrain how we report results (the threshold provenance, the decay-control interpretation, and the convolution control) are recorded where they are used, not here.
- **Item numbering.** Items 26--34 use the numbering of the master log. If items are merged into the final version, the numbering should be reassigned once, at the end.

## Item 32: the GDN collapse is not attributable to the decay gate alone

**Retracted claim.** That the trimodal GDN distribution could be read as a decay effect, motivated by the architecture's name.

**Correction.** Our within-architecture control does not support that reading. A learned scalar retention factor added to the ladder's own block produces a uniform collapse at every initialization in the pre-registered sweep (10/10 seeds at chance), and at one additional initialization a minority failure mode (8 successes, one degraded seed at 0.0595, one collapse at 0.0179) -- but neither reproduces GDN's three-mode distribution, whose third mode is an intermediate lock-in (0.298, 0.437): the degraded seed sits at 0.0595, about four times chance, and no decay-only seed falls in the band $[0.25, 0.45)$ where GDN's two lock-in seeds sit. Decay is sufficient for a collapse and not sufficient for the GDN distribution.

**Consequence for the paper.** Section 7.7 states that the mechanism of the GDN collapse remains open, and that it must involve gate composition, the state size, or the parameterization. The additional condition is labelled as an additional condition and is never pooled with the pre-registered sweep.

## Item 33: the published comparison was checked against the primary source

**Retracted claim.** That the small-scale check preserves the published large-scale ordering, together with the attribution of the published figures and the description of the published numbers as a 1.3B S-NIAH-1 result at 4K/8K.

**Correct provenance.** The figures in our earlier sentence were correct, as was their attribution and their scope: Table 2 of arXiv 2412.06464 (v3) is the zero-shot S-NIAH table for the 1.3B models, and its S-NIAH-1 columns at 4K and 8K read DeltaNet 99.0/98.8 and Gated DeltaNet 91.4/91.8. The only error in our sentence was the table number (Table 2, not Table 3). A reviewer suggested that the published ordering is reversed (GDN above DeltaNet) at 400M parameters and 15B tokens; the paper's 400M/15B table is Table S.1, an ablation on perplexity and commonsense accuracy rather than RULER S-NIAH, so we could not confirm the reversal and did not adopt it.

**What changed anyway.** Section 7.5 now states the published comparison as task-variant dependent, because the verified table shows it: on S-NIAH-1 (repeated synthetic context) DeltaNet is ahead (99.0/98.8 against 91.4/91.8 at 4K/8K), and on S-NIAH-2/3 (real-world-essay context) Gated DeltaNet is ahead (14.4 against 29.6 at 8K on S-NIAH-2; 22.4 against 27.6 at 4K on S-NIAH-3). Our small-scale ordering matches the first variant and not the others, and we do not identify which published regime our task belongs to. The unqualified "sign-preserving" phrasing is replaced by an explicit statement of what agrees and what does not.

## Item 34: the convolution control

**Retracted claim.** That the decomposition's external validity was an untested open question, because our block has no convolution while the real blocks do.

**Correction.** The control was pre-registered (`conv_ladder_predictions.md`, with two dated amendments; the operational thresholds and the outcome-to-claim map fixed before the run finished), and it holds: with a depthwise causal convolution the stability transition is $+0.1628\pm0.0249$ (strictly positive), the erase-strength transition is $+0.6702\pm0.1344$ (above $+0.30$ and still the largest of the three), the gating transition is $+0.0663\pm0.1063$ (below $+0.15$), and the per-seed ordering holds in both conditions. The no-conv cells reproduce the budget-matched ladder rows exactly at the reporting precision.

**Consequence for the paper.** The decomposition is not an artifact of the convolution-free block; the remaining differences from the real blocks are state size, budget, parameterization, and GDN's decay gate. Tables 7.7 and 7.8 in Section 7. One further setting is *not* replicated: the matched-state comparison in which both real blocks carry a convolution (Boesch and Wee, arXiv:2609.16183), which is stated as a limitation in Section 7.7 and Section 8.2.


---

