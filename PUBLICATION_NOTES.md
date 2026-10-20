# Publication notes — v0.1-draft

**Date:** 2026-10-20  
**Status:** Draft preprint for priority claim. The final version will be updated.  
**Author:** KPL Research (pseudonym; the final version may use a real name). Solo author; no coauthors. The GitHub account is anonymous3301-gif; the paper author name is separate.

This paper is independent of the KPL verification project (`kpl-prereg`, DOI: `10.5281/zenodo.23492899`). The two projects share methodology but address different questions; this paper concerns delta-rule memory gating, while KPL concerns Kaczmarz-Product Layer design.

## What this version is

This is the first public draft of the paper
*When is a gate worth it? Budget, capacity, and co-adaptation in delta-rule
memory*. It is published to establish a timestamped priority record. It is not
a final journal submission.

## Changes made for this draft

1. **Abstract finalized.** `abstract.md` is the final abstract for v0.1-draft,
   based on the working skeleton. The decay sentence uses the observed
   mixed two-mode outcome at the retention boundary (8 success, 1 degraded,
   1 collapse), not a generic "minority failure mode".
2. **Section 9.2.8 completed.** The "To be completed from the master log"
   placeholder was removed. The paragraph now states that the shape-sensitivity
   ratio is not reported because its growth is driven by a denominator
   approaching zero; absolute and relative drops are reported instead.
3. **Section 9.2.9 moved to Appendix A.** The verbatim master-log items 1-25
   are now in `sections/appendix_master_log_items_1_25.md`; Section 9.2.9 is a
   short pointer. This keeps the main text readable.
4. **Status notes reconciled.** `section1_intro.md` now says "draft preprint
   for priority claim", the decay-only ablation is complete, and the abstract
   points to `abstract.md`. The old "working draft / still running" note and
   the "abstract deferred" note were removed.
5. **Title fixed.** The title is
   *When is a gate worth it? Budget, capacity, and co-adaptation in delta-rule
   memory*.
6. **Figure 7 caption clarified.** Panel (b) is a step function; the plotted
   threshold grid is discrete, so some intermediate counts may be skipped
   visually.

## Retraction scope (important)

`section9_retractions.md` contains **self-retractions of earlier drafts of this
work**, not retractions of prior literature.

- It retracts or narrows the authors' own earlier claims: the threshold
  provenance, the raw-key "optimal erase rate" statement, the variance-reduction
  claim, the clipping-artifact reading, the decay-only attribution of GDN's
  collapse, the table number in the published comparison, and the convolution
  open question.
- The published comparison is checked against the primary source. The numbers
  and attribution were found correct; the table number (Table 3 -> Table 2) and
  the implication of a single published ordering are retracted. No external
  paper is named as wrong.
- There is no legal or ethical claim against a third party in this section.
- **Exact scope:** the retractions concern the author's own prior claims and interpretations. They do not name or assert that any prior work is wrong.
- **9.2.6 specifically:** the numbers, attribution, and scope in the original source were correct. The error was in this manuscript's citation (the table number, Table 3 -> Table 2) and in the implication that a single published ordering exists. It is not a claim that the original paper is wrong.

## Known limitations of this draft

- The text is a draft; the final version will be updated.
- The GDN collapse mechanism remains open. The paper does not identify it.
- The published comparison is task-variant dependent (S-NIAH-1 vs S-NIAH-2/3);
  the paper does not assert a single published ordering.
- Figure 7 panel (b) is a step function; the discrete threshold grid can skip
  intermediate failure counts visually. The caption states this.
- The paper is distributed as Markdown (`paper.md`) plus figure PDFs. No
  compiled PDF is included in v0.1-draft.

## IP and affiliations

Per the author, this is a personal project using a home PC and AI assistance.
No institutional or corporate IP review is required. This note is included for
transparency; it is not a legal representation.

## Data and code

- Code: `code/` (MIT).
- Key result files: `data/` (JSON/Markdown).
- Figures: `figures/` (PDF).
- Text: `paper.md`, `sections/`, `abstract.md` (CC-BY-4.0).

## DOI

Zenodo DOI: pending `v0.1-draft` release.
