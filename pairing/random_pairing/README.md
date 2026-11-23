# Corrected Random Pairing ADERH

`CorrectedRandomADERH` uses the same anchor sampling, random permutation and
fixed-point removal as ADERH 1.0.0. It differs only in pair representation:

- distances are calculated between each anchor and its actual sampled partner;
- squared radii are `distance_squared / 2`;
- radii use the same blocked order as centers `[anchors; partners]`.

This is the control required to distinguish representation corrections from
the effect of Local, Hybrid or LID-Adaptive partner selection.
