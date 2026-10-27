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
