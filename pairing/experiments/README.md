# Pairing comparison experiments

The pipeline compares eleven configurations on identical nested datasets,
train/test splits, preprocessing and random seeds:

1. official ADERH;
2. corrected random pairing;
3. corrected random centers with shuffled radii;
4. corrected random with training-only coverage calibration;
5. shuffled radii with training-only coverage calibration;
6. local pairing;
7. fixed hybrid pairing;
8. rank LID-adaptive, high LID local;
9. rank LID-adaptive, high LID random;
10. continuous LID-adaptive, high LID local;
11. continuous LID-adaptive, high LID random.

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
python -m pairing.experiments.experiments full
```

`phase1` is the bounded development comparison defined in `fase2.md`: it runs
dimensions `2, 20, 50, 100`, seeds `0–4`, and only the non-LID controls:
`original`, `corrected_random`, the three radius controls, `local`, and
`hybrid`. Candidate policies are admitted to the full experiment only after
this bounded comparison has been reviewed.

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
