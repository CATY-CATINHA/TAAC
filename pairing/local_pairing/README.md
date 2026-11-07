# Local Pairing ADERH

## Overview

`LocalADERH` is an experimental variant of ADERH that replaces the original random partner selection with a local pairing strategy.

Instead of pairing each sampled anchor with another randomly selected anchor, Local Pairing selects a partner from the anchor's `k_neighbors` nearest neighbours in the training dataset.

The objective is to investigate whether using local geometric information when constructing hyperspheres can improve anomaly detection, particularly when the data has locally dense regions or complex neighbourhood structures.

## How it works

For each estimator:

1. **Anchor sampling:** Randomly select `n` distinct observations from the training dataset.
2. **Neighbour calculation:** For each anchor, calculate its squared Euclidean distance to every observation in the training dataset.
3. **Self-exclusion:** Exclude the anchor itself from its candidate neighbours.
4. **Neighbour selection:** Identify the `k_neighbors` nearest observations.
5. **Partner selection:** Randomly select one of those nearest neighbours as the anchor's partner.
6. **Hypersphere construction:** Create two hyperspheres, one centred at the anchor and the other at its selected partner. Both hyperspheres use half the squared distance between the pair as their squared radius.
7. **Scoring:** Apply the original ADERH scoring mechanism.

The implementation reuses the scoring mechanism from the official ADERH implementation through the shared `BasePairingADERH` class.

## Parameters

`LocalADERH` inherits the main ADERH parameters and introduces one additional parameter:

| Parameter | Default | Description |
|---|---:|---|
| `n_estimators` | 256 | Number of estimators in the ensemble. |
| `n` | 18 | Number of anchors sampled per estimator. |
| `contamination` | 0.1 | Expected proportion of anomalies, used to determine the decision threshold. |
| `random_state` | `None` | Controls random sampling and partner selection. |
| `k_neighbors` | 5 | Number of nearest neighbours considered when selecting a partner. |

`k_neighbors` must satisfy:

\[
1 \leq k\_neighbors < n\_samples
\]

## Implementation details

- Neighbours are selected from the **entire training dataset**, not only from the sampled anchors.
- The anchor itself is excluded from its candidate neighbours.
- Partner selection is performed independently for each anchor.
- Different anchors may select the same partner.
- Consequently, pairs are not guaranteed to be disjoint or one-to-one.
- When `k_neighbors=1`, each anchor is always paired with its nearest neighbour.

The implementation uses squared Euclidean distances to identify the nearest neighbours.

## Usage

From the repository root:

```python
from pairing.local_pairing.local_aderh import LocalADERH

model = LocalADERH(
    n_estimators=256,
    n=18,
    contamination=0.1,
    random_state=42,
    k_neighbors=5
)

model.fit(X_train)
```

`X_train` must be a two-dimensional numerical array.

## Relationship with the original ADERH

The original ADERH randomly pairs sampled anchors. Local Pairing changes only the partner-selection strategy while reusing the original scoring mechanism.

One implementation detail is important when comparing the variants: the official ADERH implementation calculates pair radii using `X[perm]`, while its hypersphere centres use `anchors[perm]`. Local Pairing calculates radii using the actual selected partners.

Therefore, differences in results may reflect both the pairing strategy and this difference in radius calculation.

## Limitations and considerations

- The nearest-neighbour search calculates distances from each anchor to the entire training dataset, which introduces additional computational cost.
- The parameter `k_neighbors` controls the locality of the pairing strategy and may influence the resulting hypersphere sizes.
- Independent partner selection allows repeated partners and overlapping pair assignments.
- The effect of local pairing on anomaly detection performance must be assessed empirically.

## Related implementation

Shared functionality is implemented in:

```text
pairing/common/base_aderh.py
```

See also:

- `pairing/hybrid_pairing/README.md` — Fixed Hybrid Pairing.
- `pairing/Local_Hybrid_Pairing_Notes.md` — Additional implementation notes.