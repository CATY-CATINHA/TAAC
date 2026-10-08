# Internal ADERH analysis (official implementation 1.0.0)

This project depends on the official `aderh==1.0.0` package and does not copy,
modify, subclass, or vendor its detector. The findings below come from the
[official implementation](https://github.com/Walid10010/ADERH/blob/main/aderh/_aderh.py)
and its
[implementation notes](https://github.com/Walid10010/ADERH/blob/main/IMPLEMENTATION_NOTES.md).

## What is accessible after fitting

The fitted detector retains sphere centers in `_centers` and squared radii in
`_sq_radii`. It also retains per-estimator random seeds in `_seeds`. These names
are private (underscore-prefixed), so they are inspectable in version 1.0.0 but
are not a stable public API.

Pitch, NDensity, nearest-sphere assignments, coverage masks, and individual
ensemble contributions are local variables inside `_raw_scores`. They are not
stored on the fitted object or returned by `decision_function`. Therefore this
baseline records none of them.

## Verified distance conventions

ADERH works in **squared Euclidean distance** space. For an anchor/partner pair
whose squared distance is `d_squared`, the stored squared radius is
`d_squared / 2`. Thus its ordinary Euclidean radius is `d / sqrt(2)`, not
`d / 2`.

Version 1.0.0 intentionally preserves three reference-code details:

1. Partner points for pair-distance calculation come from `X[perm]`, among the
   first `n` input rows, rather than from `anchors[perm]`. This makes the model
   sensitive to training-row order.
2. Centers are stacked block-wise as `[anchors; anchors[perm]]`.
3. Radii are duplicated in interleaved order. Consequently, some duplicated
   centers are paired with a radius from a different pair.

Pitch is the covered point's squared center distance divided by the selected
sphere's squared radius. NDensity counts assignments to each selected sphere,
divides those counts by squared radius, then normalizes by the largest density
within that estimator. The contribution for a covered point is
`(1 - NDensity) * Pitch`; uncovered points retain a contribution of one.
Contributions are averaged across estimators. Higher final values are more
anomalous.

## Later instrumentation approach

A separate, explicitly version-pinned diagnostic module could compute and
export the scoring intermediates from the fitted private `_centers` and
`_sq_radii`, without changing the installed ADERH package. It should be treated
as an independent analysis implementation, include score-equivalence tests
against `decision_function` at tight numerical tolerance, and fail fast if the
ADERH version or private state shape changes. That work is deliberately deferred
until after the uninstrumented baseline is validated.

