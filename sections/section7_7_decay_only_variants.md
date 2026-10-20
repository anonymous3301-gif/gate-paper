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
