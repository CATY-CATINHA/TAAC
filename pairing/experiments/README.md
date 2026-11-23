# Pairing comparison experiments

The pipeline compares thirteen configurations on identical nested datasets,
train/test splits, preprocessing and random seeds:

1. official ADERH;
2. corrected random pairing;
3. corrected random centers with shuffled radii;
4. corrected random with training-only coverage calibration;
5. shuffled radii with training-only coverage calibration;
6. corrected random pairing in ten-feature random subspaces;
7. ten-feature random subspaces with shuffled radii;
8. local pairing;
9. fixed hybrid pairing;
10. rank LID-adaptive, high LID local;
11. rank LID-adaptive, high LID random;
12. continuous LID-adaptive, high LID local;
13. continuous LID-adaptive, high LID random.

The corrected-random configuration is the control for the actual-partner
distance and radius-order corrections shared by all experimental variants.

## Phase 2 geometry diagnostics

New runs also record diagnostics needed before changing the pairing policy
again:

- overlap between neighbours in the complete ambient space and neighbours in
  the known two-dimensional informative geometry;
- pair-distance and ordinary-radius distributions;
- radius/center-eccentricity correlation;
- sampled training coverage per estimator;
- sampled number of covering estimators and uncovered fraction.
- effective subspace dimension and, for the controlled synthetic dataset,
  how often sampled subspaces contain either or both informative features.

The informative two-dimensional prefix is used only to diagnose the controlled
synthetic experiment. It is never supplied to a detector and is not a tuning
signal that would be available on real data. Coverage diagnostics use seeded
samples whose sizes are recorded in every row and in the metadata.

The complete decision record and experimental roadmap are in
[`../../fase2.md`](../../fase2.md).

## Run

From the repository root, after installing development dependencies:

```bash
python -m pytest
python -m pairing.experiments.experiments smoke
python -m pairing.experiments.experiments phase1
python -m pairing.experiments.experiments subspace-validation
python -m pairing.experiments.experiments subspace-full
python -m pairing.experiments.experiments full
```

`phase1` is the bounded development comparison defined in `fase2.md`: it runs
dimensions `2, 20, 50, 100`, seeds `0–4`, and only the non-LID controls:
`original`, `corrected_random`, the three radius controls, the two subspace
controls, `local`, and `hybrid`.

`subspace-validation` evaluates the two subspace controls and the original on
the held-out seeds `5–9`. `subspace-full` is the focused eight-dimensional,
ten-seed comparison between the original and the frozen shuffled-subspace
candidate. The seeds `0–4` remain development evidence, so statistical tests
over all ten seeds must be described as exploratory rather than as a wholly
held-out confirmatory test.

Diagnostic sampling can be changed with
`--diagnostic-neighbour-k`, `--diagnostic-max-points`, and
`--diagnostic-max-estimators`.

New runs default to `results/current/` and `figures/current/`. These directories
are ignored until a run has been reviewed. Metadata records the checked-out Git
revision, environment, model count and implementation convention.

The files directly under `results/`, `figures/`, and `results/rank_based/` are
historical snapshots generated before the final partner-distance and
radius-alignment correction. They must not be reported as evidence for the
current implementation. See [`RESULTS_STATUS.md`](RESULTS_STATUS.md).

## Pre-subspace retest

The frozen `random_subspace_10_shuffled` configuration is called **pre-subspace**.
The retest compares it with `aderh==1.0.0` on fresh synthetic seeds and eleven
tabular ADBench datasets used by Durani et al. Run from the repository root:

```bash
python -m pairing.experiments.pre_subspace fetch
python -m pairing.experiments.pre_subspace real
python -m pairing.experiments.pre_subspace synthetic
python -m pairing.experiments.pre_subspace_stats
```

Only `fetch` needs a network connection on the first run. It downloads the
manifest's NPZ files from a fixed ADBench revision and verifies SHA-256,
shape, binary labels and anomaly counts. Later runs reuse the local cache.
All commands accept the default paths from the repository root. To use
another output directory pass the same `--output DIR` to both evaluation
commands and to the statistics command. New runs default to
`results/current/pre_subspace/`. The reviewed run tables and paired summaries
are in [results/pre_subspace/](results/pre_subspace/).

## First LID selector checkpoint

The [frozen protocol](lid_gate_protocol.md) defines the training-only
selection rule, its three controls and the commands for both suites.
The [paired comparison](results/lid_gate_v1/README.md) records the decisions
and effects relative to each control. New executions default to
`results/current/lid_gate_v1/`.
