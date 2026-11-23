# ADERH pairing variants

This package compares the official `aderh==1.0.0` implementation with pairing
variants that use a consistent pair representation.

Every experimental variant:

- samples `n` anchors per estimator;
- selects a one-to-one derangement of those sampled anchors as partners;
- stores centers in blocked order `[anchors; partners]`;
- calculates squared radii from the actual pairs as `distance_squared / 2`;
- stores radii in the same blocked order as the centers;
- reuses ADERH's scoring implementation.

The comparison includes:

| Name | Pairing policy |
|---|---|
| `original` | Unmodified ADERH 1.0.0 |
| `corrected_random` | Random pairing with actual pair distances and aligned radii |
| `shuffled_radius` | Corrected random centers with the same radii reassigned between pairs |
| `coverage_calibrated_random` | Corrected random radii scaled to original training coverage |
| `shuffled_coverage_calibrated` | Radius reassignment followed by training-only coverage calibration |
| `local` | All anchors prefer nearby sampled anchors |
| `hybrid` | A fixed fraction of anchors prefer local partners |
| `lid_adaptive_*` | LID determines which anchors prefer local partners |

`corrected_random` is an essential control. It separates the effect of fixing
the pair/radius representation from the effect of introducing locality.

See the variant READMEs and [`experiments/README.md`](experiments/README.md) for
the exact policies and execution instructions.

The corrected-random control is documented in
[`random_pairing/README.md`](random_pairing/README.md).

The Phase 2 radius controls are also documented there. They use no test labels
and are intended to isolate center/radius effects before introducing another
adaptive pairing policy.
