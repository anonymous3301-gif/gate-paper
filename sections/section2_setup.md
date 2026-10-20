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

