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

