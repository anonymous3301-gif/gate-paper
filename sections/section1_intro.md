# 1 Introduction



**Title:** *When is a gate worth it? Budget, capacity, and co-adaptation in delta-rule memory*



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

