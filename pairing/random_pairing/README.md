# Corrected Random Pairing ADERH

`CorrectedRandomADERH` uses the same anchor sampling, random permutation and
fixed-point removal as ADERH 1.0.0. It differs only in pair representation:

- distances are calculated between each anchor and its actual sampled partner;
- squared radii are `distance_squared / 2`;
- radii use the same blocked order as centers `[anchors; partners]`.

This is the control required to distinguish representation corrections from
the effect of Local, Hybrid or LID-Adaptive partner selection.

## Phase 2 radius controls

Three additional controls isolate why the official geometry obtains more
coverage even when its median radius is similar to `corrected_random`.

### `ShuffledRadiusRandomADERH`

- uses exactly the same centers and partners as `corrected_random`;
- preserves the complete per-estimator distribution of squared radii;
- randomly reassigns pair radii between center pairs;
- duplicates the reassigned radius consistently for both centers of a pair.

This tests whether reducing the correlation between center location and radius
is useful without changing radius scale.

### `CoverageCalibratedRandomADERH`

- keeps corrected, aligned radii;
- fits an unmodified ADERH reference on the same training matrix;
- measures reference coverage on seeded samples of training points and
  estimators;
- selects the smallest squared-radius multiplier `gamma >= 1` that attains at
  least the same sampled mean estimator coverage;
- never shrinks the corrected base radii when their coverage already meets the
  reference target;
- never uses labels, test observations, ROC-AUC or Average Precision.

The selected scale and target/achieved coverage are retained as fitted
attributes and recorded by the experiment pipeline.

### `ShuffledCoverageCalibratedRandomADERH`

First reassigns radii between pairs, then applies the same training-only
coverage calibration. This isolates whether reassignment and scale provide
complementary effects.

These classes are experimental controls. Matching the original detector's
training coverage is not yet the final deployment rule; it is a causal test of
the center/radius mechanism identified in `fase2.md`.
