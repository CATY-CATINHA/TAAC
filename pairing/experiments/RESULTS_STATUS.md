# Pairing results status

## Current implementation

The current implementation starts at commit `75e849d`, which changed both the
partner distances used for radii and the ordering that associates radii with
blocked centers. The consolidation work also adds a corrected-random control
and recalibrates the continuous LID strategy.

No full result snapshot for this implementation has been validated yet.

Phase 2 development runs additionally evaluate radius reassignment and
training-only coverage calibration. Their current five-seed results are
documented in `fase2.md` as bounded diagnostic evidence. They are not a full
result snapshot and do not yet demonstrate an improvement over the original
ADERH baseline.

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
