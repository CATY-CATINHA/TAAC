# Pairing comparison experiments

The pipeline compares eight configurations on identical nested datasets,
train/test splits, preprocessing and random seeds:

1. official ADERH;
2. corrected random pairing;
3. local pairing;
4. fixed hybrid pairing;
5. rank LID-adaptive, high LID local;
6. rank LID-adaptive, high LID random;
7. continuous LID-adaptive, high LID local;
8. continuous LID-adaptive, high LID random.

The corrected-random configuration is the control for the actual-partner
distance and radius-order corrections shared by all experimental variants.

## Run

From the repository root, after installing development dependencies:

```bash
python -m pytest
python -m pairing.experiments.experiments smoke
python -m pairing.experiments.experiments full
```

New runs default to `results/current/` and `figures/current/`. These directories
are ignored until a run has been reviewed. Metadata records the checked-out Git
revision, environment, model count and implementation convention.

The files directly under `results/`, `figures/`, and `results/rank_based/` are
historical snapshots generated before the final partner-distance and
radius-alignment correction. They must not be reported as evidence for the
current implementation. See [`RESULTS_STATUS.md`](RESULTS_STATUS.md).
