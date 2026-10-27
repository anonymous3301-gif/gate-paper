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
