# Fixed Hybrid Pairing ADERH

`HybridADERH` combines local preferences with random pairing. `alpha` controls
the fixed proportion of sampled anchors that receive a local constraint.

For each estimator:

1. sample `n` anchors;
2. randomly select `round(alpha * n)` local positions;
3. restrict those positions to their `k_neighbors` nearest sampled anchors;
4. assign random costs to the remaining positions;
5. solve one global minimum-cost assignment, expanding the local neighbourhood
   if necessary;
6. construct center-aligned radii from the actual pairs.

The assignment is a one-to-one derangement. Consequently, once `alpha > 0`,
the random and local decisions are coupled by the global assignment; the
remaining random component is not an independent copy of the official random
permutation.

Special cases:

- `alpha=0` is the corrected-random control: centers match the official ADERH
  centers for the same seed, while radii use actual partners and aligned order;
- `alpha=1` has the same partner-selection policy as `LocalADERH`.

```python
from pairing.hybrid_pairing.hybrid_aderh import HybridADERH

model = HybridADERH(
    n_estimators=256,
    n=18,
    contamination=0.1,
    random_state=42,
    k_neighbors=5,
    alpha=0.5,
)
model.fit(X_train)
```

`alpha` must be in `[0, 1]`; `k_neighbors` must be between `1` and `n - 1`.
