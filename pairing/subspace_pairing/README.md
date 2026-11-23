# Random-subspace ADERH

`RandomSubspaceADERH` builds each ensemble member in an independently sampled
feature subspace. The current candidate uses at most ten features per
estimator:

- if the dataset has ten or fewer features, the estimator uses every feature;
- otherwise, it samples ten features without replacement;
- anchors and partners are selected by corrected random pairing;
- centers, pair distances, radii, coverage and scores are all calculated in
  that estimator's subspace;
- the optional `shuffle_radii` ablation preserves each estimator's radius
  distribution while randomly reassigning radii between center pairs.

The shuffled configuration is the frozen Phase 2 candidate. Ten features were
selected using development seeds `0–4`; validation seeds `5–9` were kept out
of that choice. The detector never receives the identities of the two
informative synthetic features, labels, or test metrics.

The method addresses ambient noise directly: pair distances no longer
accumulate contributions from every irrelevant feature, and the ensemble
samples many alternative views of the data. In the controlled experiment this
restores coverage at high dimensionality. It does not, however, guarantee
that every estimator contains informative features.

Run the held-out and focused comparisons from the repository root:

```bash
python -m pairing.experiments.experiments subspace-validation
python -m pairing.experiments.experiments subspace-full
```

The complete rationale and current results are documented in
[`../../fase2.md`](../../fase2.md).
