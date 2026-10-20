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
