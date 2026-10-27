# When is a gate worth it? Budget, capacity, and co-adaptation in delta-rule memory

**Author:** KPL Research (pseudonym; the final version may use a real name)  
**Version:** v0.1-draft, 2026-10-20  
**Status:** **Draft preprint for priority claim. The final version will be updated.**  
**DOI (v0.1.1-draft):** [10.5281/zenodo.23686059](https://doi.org/10.5281/zenodo.23686059)  
**DOI (v0.1-draft):** [10.5281/zenodo.23493605](https://doi.org/10.5281/zenodo.23493605)  
**Concept DOI (all versions):** [10.5281/zenodo.23493604](https://doi.org/10.5281/zenodo.23493604)

[![DOI (Zenodo)](https://zenodo.org/badge/DOI/10.5281/zenodo.23686059.svg)](https://doi.org/10.5281/zenodo.23686059)

This repository contains the Markdown manuscript, figures, code, and key result
files for the paper. It is a draft: the text is readable and the claims are
supported by the included artifacts, but it is not a final journal submission.

This paper is independent of the KPL verification project (`kpl-prereg`, DOI: `10.5281/zenodo.23492899`). The two projects share methodology but address different questions; this paper concerns delta-rule memory gating, while KPL concerns Kaczmarz-Product Layer design.

## Abstract

Delta-rule memory powers linear-time sequence models, and its update rule mixes
three operations -- key normalization, erase strength $\beta$, and norm-based
gating -- that are usually introduced together, so each one's value depends on
capacity, on budget, and on whether the evaluation is a post-hoc swap or a
retraining experiment. We decompose them on multi-query associative recall at
$d_{\mathrm{mem}}=16$, holding the task, architecture, and evaluation fixed,
and rerun the ladder at two budgets (1500 and 3000 steps; 3 seeds per cell).
Raw keys with $\beta=1$ fail at initialization rather than during training:
$99.97\%$ of writes violate the LMS stability bound, and the first forward pass
has loss $2.36\times10^{18}$ against $5.35$ for normalized keys. The stability
effect is budget-dependent -- a variance reduction at 1500 steps, a mean shift
at 3000. Erase strength is the dominant effect at the baseline budget
($+0.6724$, 95% CI $\pm0.0447$) and persists at 3000 steps ($+0.2950$,
$\pm0.1587$); a fixed $\beta=0.125$ recovers about $90\%$ of the gap to a
Kaczmarz gate at both capacities, so the stability bound is a constraint rather
than an objective. Gating behaves differently: the gate is worth $+0.0729$ at
1500 steps but only $+0.0030$ at 3000 steps (CI includes zero), which makes it a
sample-efficiency advantage rather than a persistent training-time bias. Its
shape nevertheless co-adapts: a Kaczmarz-trained model loses $0.380$, $0.227$,
and $0.161$ accuracy when evaluated with a mean-matched constant at
$d_{\mathrm{mem}}=16,24,32$, while a shape-matched gate costs at most $0.0011$
and is equivalent to the gate itself within $0.01$ (TOST). A curriculum does
not substitute for normalization, though it removes the raw-key collapse mode.
In a real-architecture check, GDN is trimodal over 10 seeds while Delta is
stable (Brown--Forsythe $p=0.0022$). A within-architecture control shows that a
scalar retention factor can produce a mixed two-mode outcome at the retention
boundary we tested, but not GDN's three-mode distribution, so the mechanism of
that collapse remains open. A gate is therefore worth its complexity only
conditionally: budget-matched, a fixed erase rate reaches within $0.01$ of the
gate at the settings tested, yet the gate's shape still determines what
transfers.

## Repository contents

| Path | Content |
|---|---|
| `paper.md` | Single-file manuscript (abstract + Sections 1-9 + appendices) |
| `abstract.md` | Title, author, abstract, draft status |
| `sections/` | Per-section Markdown sources |
| `figures/` | Figure 1-7 PDFs |
| `code/` | Scripts used for the ladder, controls, statistics, and figures |
| `data/` | Key result JSON/Markdown files |
| `PUBLICATION_NOTES.md` | Draft status, known limitations, retraction scope, figure notes |
| `LICENSE` | CC-BY-4.0 for the text and figures |
| `LICENSE-CODE` | MIT for the code |

## Code map

The main entry points are:

- `code/budget_match.py` -- budget-matched four-stage ladder
- `code/conv_ladder.py` -- convolution control
- `code/clip_ablation.py` -- clip-protocol control
- `code/init_eig_diag.py` -- initialization diagnostic
- `code/cross_eval_matched.py`, `cross_eval_d24.py`, `cross_eval_d32.py` -- post-hoc transfer
- `code/tost_matched_mean.py` -- TOST for the matched-mean comparison
- `code/delta_decay_only.py`, `delta_decay_only_v3.py` -- decay-only control
- `code/verify_gdn_stats.py` -- GDN vs Delta statistics
- `code/fig1_trajectory.py` ... `code/fig7_threshold_sweep.py` -- figures
- `code/section6_analysis.py`, `code/seed_stats.py`, `code/npairs_sweep.py`, `code/harmonic.py`, `code/beta_power.py`, `code/rawk_log.py` -- supporting analyses

## Draft status and limitations

- This is **v0.1-draft**. The final version will be updated.
- The retractions section (`sections/section9_retractions.md`) contains
  **self-retractions of earlier drafts of this work**, not retractions of prior
  literature. It is kept because the corrections are part of the scientific
  record.
- Figure 7 panel (b) is a step function; the plotted threshold grid is discrete,
  so some intermediate failure counts may be skipped visually. The caption says
  so explicitly.
- The GDN collapse mechanism remains open; the paper does not claim to identify
  it.
- The published comparison is task-variant dependent; the paper does not assert
  a single published ordering.

## License

Text and figures: CC-BY-4.0. Code: MIT. See `LICENSE` and `LICENSE-CODE`.
