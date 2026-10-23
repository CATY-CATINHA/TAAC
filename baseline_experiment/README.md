# ADERH distance-concentration experiment

This repository is a reproducible baseline study of how adding independent,
irrelevant dimensions affects the **original** ADERH detector. It uses the
official `aderh==1.0.0` package unchanged. The experiment measures outcomes; it
does not assume that predictive performance must degrade with dimensionality.

The completed 80-configuration baseline and its interpretation are documented
in [`docs/full_experiment_report.md`](docs/full_experiment_report.md).

Primary sources:

- [Official ADERH repository](https://github.com/Walid10010/ADERH)
- [NeurIPS 2025 paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f2b6bc2104f6f4481eef101f6a32c085-Abstract-Conference.html)
- [Official implementation notes](https://github.com/Walid10010/ADERH/blob/main/IMPLEMENTATION_NOTES.md)

## Environment and installation

Python 3.10 or newer is required. From a PowerShell prompt:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
```

`pyproject.toml` pins the official detector to ADERH 1.0.0 and constrains the
numerical/plotting dependencies to compatible major releases. Every run also
writes exact package versions, Python version, platform, configuration, and
sampling details to `results/<mode>_metadata.json`.

## Protocol

For each seed, one nested dataset is generated with 2,000 observations and
exactly 5% anomalies. The two informative features follow:

- normals: `N([0, 0], I)`;
- anomalies: `N([4, 4], I)`.

All later features are independent `N(0, 1)` noise for both classes. Generation
is feature-major and every dataset is a column prefix of the largest one. Thus
the observations, labels, informative coordinates, and earlier noise
coordinates are exactly fixed as dimensionality grows.

A seeded, stratified 70/30 split is reused across all dimensionalities within a
seed. Labels are used only by the splitter, evaluation metrics, and the separate
normal-only geometric diagnostic. `MinMaxScaler` is fit on the training split
only, then applied to train and test data. ADERH receives no labels.

MinMax scaling follows the official experiment protocol. Because it acts on
each feature independently, adding columns does not change already-present
scaled training columns. Test values outside the training range may transform
outside `[0, 1]`; this is intentional and prevents leakage. Scaling does change
the geometry being studied: each feature's contribution is normalized by its
training range, so conclusions apply to this documented, official-style
preprocessing regime.

The model is constructed as `ADERH(random_state=seed)`. Its documented defaults
are retained: `n_estimators=256`, `n=18`, and `contamination=0.1`. The synthetic
prevalence is 0.05, but ADERH's `contamination` parameter only sets its binary
threshold; this study evaluates the untouched continuous `decision_function`
scores. Official semantics and an integration test confirm that higher scores
mean more anomalous observations.

### Metrics

Predictive metrics are test ROC-AUC and Average Precision. The plots include a
0.5 ROC-AUC random-ranking baseline and the observed test anomaly prevalence as
the Average Precision baseline.

Relative Contrast (RC) is a separate geometric metric, evaluated on scaled
normal training observations:

```text
RC = mean distance between random distinct normal reference pairs
     --------------------------------------------------------------
       mean nearest-neighbor distance among those references
```

For each configuration, up to 512 normal references are selected without
replacement. The numerator uses 4,096 seeded, uniformly sampled ordered pairs
with self-pairs excluded. The denominator uses a nearest-neighbor index and
discards the self-match. The same seed, reference rows, and random-pair indices
are used across dimensionalities; no full pairwise matrix is formed. Raw CSV
columns and metadata record the actual counts and both component means.

Fit timing covers the official `fit` call (including ADERH's training-score
calculation). Scoring timing covers only `decision_function` on the test split;
data generation, scaling, and metrics are excluded.

The raw results also record score diversity and coverage saturation. Under the
official 1.0.0 scorer, each estimator contribution starts at one and becomes
strictly smaller only if at least one hypersphere covers the observation.
Consequently, a final score exactly equal to one means that the observation was
uncovered by every hypersphere in every ensemble member. Overall, normal-class,
and anomaly-class uncovered fractions are reported separately.

## Run

Validate the inexpensive path first:

```powershell
.venv\Scripts\python -m aderh_distance.experiments smoke
```

After activating the virtual environment, the shorter equivalent is
`aderh-experiment smoke`.

Smoke mode runs dimensions `[2, 50]` for seed `0`. Only after it succeeds, run
the complete 80-configuration experiment (8 dimensions × 10 seeds):

```powershell
.venv\Scripts\python -m aderh_distance.experiments full
```

The full dimensions are `[2, 5, 10, 20, 50, 100, 200, 500]`; seeds are
`0` through `9`. Use `--help` for output-directory and RC sampling options.

After a full run, reproduce the trend and score-collapse checks with:

```powershell
.venv\Scripts\python -m aderh_distance.analysis
```

## Artifacts

Each mode writes:

- `results/<mode>_raw_results.csv`: one row per dimension and seed, including
  predictive/geometric metrics, timings, split sizes, preprocessing, score
  direction, and serialized model hyperparameters;
- `results/<mode>_summary.csv`: means and population standard deviations by
  dimensionality;
- `results/<mode>_metadata.json`: exact environment and protocol metadata;
- `results/full_analysis.json` and `results/full_seed_trends.csv`: endpoint,
  monotonicity, rank-correlation, random-baseline, and score-collapse checks;
- `figures/<mode>_*.png` and `.pdf`: Relative Contrast, ROC-AUC, Average
  Precision, uncovered fraction, and runtime plots with variability across
  seeds.

Generated results and figures are git-ignored by default so that arbitrary
local runs cannot be mistaken for validated evidence. The tracked `full_*`
artifacts are an intentional frozen snapshot of the completed baseline reported
in `docs/full_experiment_report.md`. Do not report a new trend until its raw CSV
has actually been produced and checked.

## ADERH internals

The inspection is documented in
[`docs/aderh_internal_analysis.md`](docs/aderh_internal_analysis.md). In short,
fitted centers and squared radii exist only as private attributes. Pitch,
NDensity, and estimator contributions are not exposed or retained. The official
implementation uses squared Euclidean distances and stores squared radii as
half the squared pair distance, corresponding to ordinary radius `d / sqrt(2)`.
Instrumentation is intentionally deferred to a later, separately validated
phase.
