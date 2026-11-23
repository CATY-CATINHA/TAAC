# Local Pairing ADERH

`LocalADERH` makes all sampled anchors prefer local partners while preserving a
one-to-one pairing.

For each estimator it:

1. samples `n` anchors from the training set;
2. computes squared Euclidean distances between those sampled anchors;
3. restricts every anchor to its `k_neighbors` nearest sampled anchors;
4. uses a minimum-cost assignment to obtain a derangement in which every
   sampled anchor appears exactly once as a partner;
5. expands the permitted neighbourhood when the requested `k_neighbors` does
   not admit a complete assignment;
6. creates centers `[anchors; partners]` and matching squared radii
   `distance_squared / 2`;
7. reuses the ADERH scoring mechanism.

Partners therefore come from the sampled anchors, not from the complete
training set. They cannot repeat and an anchor cannot partner with itself.

```python
from pairing.local_pairing.local_aderh import LocalADERH

model = LocalADERH(
    n_estimators=256,
    n=18,
    contamination=0.1,
    random_state=42,
    k_neighbors=5,
)
model.fit(X_train)
```

`k_neighbors` must be between `1` and `n - 1`. Local search and assignment add
cost relative to the official random strategy. The effective neighbourhood can
also become larger than requested when a one-to-one assignment is infeasible.
