# Pairing results status

## Current implementation

The current implementation starts at commit `75e849d`, which changed both the
partner distances used for radii and the ordering that associates radii with
blocked centers. The consolidation work also adds a corrected-random control
and recalibrates the continuous LID strategy.

No canonical all-model result snapshot for this implementation has been
validated yet.

Phase 2 development runs additionally evaluate radius reassignment and
training-only coverage calibration. Their current five-seed results are
documented in `fase2.md` as bounded diagnostic evidence. They are not a full
result snapshot and do not yet demonstrate an improvement over the original
ADERH baseline.

The subsequent random-subspace experiment produced the first positive Phase 2
result. `random_subspace_10_shuffled` was selected with seeds `0–4`, frozen,
and then evaluated on held-out seeds `5–9`. On the held-out seeds it improved
Average Precision by `+0.0346` at 50D and by `+0.1162` at 100D; the 100D ROC-AUC
gain was `+0.1227`. A focused eight-dimensional, ten-seed run also shows large
gains at 100D, 200D and 500D. These are current implementation results, but
the ten-seed significance tests are exploratory because the first five seeds
were used for development. The exact tables and limitations are in
[`../../fase2.md`](../../fase2.md).

The reviewed focused snapshot is tracked as
[`results/current/subspace-full_raw_results.csv`](results/current/subspace-full_raw_results.csv),
[`results/current/subspace-full_summary.csv`](results/current/subspace-full_summary.csv)
and
[`results/current/subspace-full_metadata.json`](results/current/subspace-full_metadata.json).
It was reproduced from clean commit `e917c79`; all 59 comparable non-timing
columns matched the development run exactly. This does not replace the still
pending canonical all-model snapshot.

The [pre-subspace retest](results/pre_subspace/README.md) evaluates this same
frozen configuration on synthetic seeds 10–29 and eleven real datasets from
the ADERH paper. It reports paired effect sizes, confidence intervals and
multiplicity-adjusted sign tests, alongside published and rerun baselines.

## Historical snapshots

The tracked `full_*`, `smoke_*`, and `rank_based/full_*` artifacts were produced
before those corrections. In particular, the experimental variants calculated
pair distances through positions in the full input matrix and combined blocked
centers with interleaved radii.

The `original` rows still describe the unmodified `aderh==1.0.0` baseline, but
rows for Local, Hybrid and LID-Adaptive variants are not evidence for the
current code.

Historical files are retained for provenance and must not be overwritten.
Fresh runs are written below `results/current/` and `figures/current/`.
