# Fixed Hybrid Pairing ADERH

## Overview

`HybridADERH` is an experimental ADERH variant that combines the original random pairing strategy with Local Pairing.

For each estimator, a proportion of the sampled anchors uses local partners, selected from their nearest neighbours in the training dataset. The remaining anchors retain the original random pairing strategy.

The parameter `alpha` controls the proportion of anchors assigned to local pairing.

The objective is to investigate whether combining local and random pairings can preserve the multi-scale behaviour of the original ADERH while incorporating local geometric information.

## How it works

For each estimator:

1. **Anchor sampling:** Randomly select `n` distinct observations from the training dataset.
2. **Random pairing:** Generate a random permutation of the sampled anchors and eliminate fixed points using the same adjacent-swap procedure as the original ADERH implementation.
3. **Local pairing proportion:** Calculate the number of anchors assigned to local pairing according to `alpha`.
4. **Anchor selection:** Randomly select which anchors will use local pairing.
5. **Neighbour calculation:** For each selected anchor, calculate its squared Euclidean distance to every observation in the training dataset, excluding itself.
6. **Local partner selection:** Identify its `k_neighbors` nearest neighbours and randomly select one.
7. **Partner replacement:** Replace the original random partner of each selected anchor with its locally selected partner.
8. **Hypersphere construction:** Construct two hyperspheres per pair, centred at the anchor and its selected partner. Both use half the squared distance between the pair as their squared radius.
9. **Scoring:** Apply the original ADERH scoring mechanism.

The implementation reuses the shared `BasePairingADERH` class, which handles hypersphere construction and scoring.

## Parameters

`HybridADERH` inherits the main ADERH parameters and introduces two additional parameters:

| Parameter | Default | Description |
|---|---:|---|
| `n_estimators` | 256 | Number of estimators in the ensemble. |
| `n` | 18 | Number of anchors sampled per estimator. |
| `contamination` | 0.1 | Expected proportion of anomalies, used to determine the decision threshold. |
| `random_state` | `None` | Controls random sampling and partner selection. |
| `k_neighbors` | 5 | Number of nearest neighbours considered for local pairing. |
| `alpha` | 0.5 | Proportion of anchors assigned to local pairing. |

The parameter `alpha` must be in the interval:

\[
0 \leq \alpha \leq 1
\]

The number of locally paired anchors is calculated as:

\[
n_{local} = \lfloor \alpha \cdot n + 0.5 \rfloor
\]

The remaining anchors retain random pairing.

## Understanding alpha

`alpha` controls the balance between random and local pairing:

| Alpha | Local anchors | Random anchors (when `n=18`) |
|---:|---:|---:|
| 0.00 | 0 | 18 |
| 0.25 | 5 | 13 |
| 0.50 | 9 | 9 |
| 0.75 | 14 | 4 |
| 1.00 | 18 | 0 |

The number of local anchors is rounded to the nearest integer using the formula above.

### Special cases

- **`alpha=0`:** All anchors retain their randomly selected partners. The resulting hypersphere centres match those of the original ADERH implementation when using the same random state and configuration.
- **`alpha=1`:** All anchors use locally selected partners. The strategy becomes equivalent to Local Pairing in terms of partner-selection behaviour.

These cases are useful when comparing the hybrid approach against its random and local components.

## Implementation details

- Random pairing follows the original ADERH permutation and fixed-point elimination procedure.
- Local partners are selected from the entire training dataset, not only from the sampled anchors.
- The anchor itself is excluded from its candidate neighbours.
- Local partner selection is independent for each selected anchor.
- Multiple anchors may select the same local partner.
- Local partner replacement can break the one-to-one pairing structure of the original random permutation.
- The parameter `k_neighbors` must satisfy:

\[
1 \leq k\_neighbors < n\_samples
\]

## Usage

From the repository root:

```python
from pairing.hybrid_pairing.hybrid_aderh import HybridADERH

model = HybridADERH(
    n_estimators=256,
    n=18,
    contamination=0.1,
    random_state=42,
    k_neighbors=5,
    alpha=0.5
)

model.fit(X_train)
```

`X_train` must be a two-dimensional numerical array.

## Important implementation note: radius calculation

Although `HybridADERH(alpha=0)` reproduces the original ADERH hypersphere centres, its radii and anomaly scores do not necessarily match the official implementation.

The reason is a detail in the official ADERH reference code:

- Hypersphere centres are constructed using `anchors[perm]`.
- Squared radii are calculated using `X[perm]`.

In contrast, the experimental variants calculate squared radii using the actual selected partners.

This distinction is intentional in the current implementation and must be considered when interpreting comparisons with the original ADERH.

## Limitations and considerations

- `alpha` determines a fixed proportion of locally paired anchors for every estimator. It does not adapt automatically to the data distribution.
- `k_neighbors` influences the local pairing behaviour.
- The random and local components do not guarantee disjoint pairs after local partner replacement.
- Local pairing requires distance calculations against the entire training dataset for the selected anchors, introducing additional computational cost.
- The impact of different `alpha` and `k_neighbors` values must be evaluated empirically.

## Related implementation

Shared functionality is implemented in:

```text
pairing/common/base_aderh.py
```

See also:

- `pairing/local_pairing/README.md` — Local Pairing.
- `pairing/Local_Hybrid_Pairing_Notes.md` — Additional implementation notes.