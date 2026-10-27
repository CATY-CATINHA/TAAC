# LID Experiment Report

## 1. Objective

This document describes the implementation, validation, and experimental analysis of the **Local Intrinsic Dimensionality (LID)** component.

The purpose of this stage is to determine whether LID can provide a stable point-level measure of local dimensional complexity that may later be used to adapt ADERH's pair-selection strategy.

The future adaptive mechanism is conceptually based on:

```text
α(x) = f(LID(x))
```

where `α(x)` will determine the balance between Local Pairing and Random Pairing.

At this stage, the function `f` is not defined. The goal is first to establish that the selected LID estimator behaves reliably and captures meaningful local variation.

---

## 2. Estimator Selection

The selected estimator is the **Maximum Likelihood Estimator of Local Intrinsic Dimensionality (MLE-LID)**.

For an observation `x`, the estimator considers the distances to its `k` nearest neighbours:

```text
r1 ≤ r2 ≤ ... ≤ rk
```

and estimates the local dimensionality from the growth pattern of these distances.

The selected initial configuration is:

```text
Estimator:       MLE-LID
Distance:        Euclidean
k:               20
Self-neighbour:  excluded
Supervision:     unsupervised
Preprocessing:   MinMaxScaler fitted on training data only
```

The value `k = 20` is used as a reference configuration rather than as an assumed optimal value.

A later sensitivity analysis will evaluate:

```text
k ∈ {10, 20, 30, 50}
```

to determine how strongly the conclusions depend on neighbourhood size.

---

## 3. MLE-LID Implementation

The estimator is implemented in:

```text
lid_analysis/lid.py
```

through:

```python
estimate_lid(X, k=20, reference=None)
```

For a dataset containing `n` observations, the function returns:

```text
[LID(x1), LID(x2), ..., LID(xn)]
```

so that each observation receives its own local dimensionality estimate.

The implementation performs several validation steps:

- verifies that input data are two-dimensional;
- rejects NaN and infinite values;
- checks whether `k` is valid;
- excludes the observation itself from its neighbourhood;
- applies numerical safeguards;
- detects degenerate neighbourhoods.

The `reference` parameter also allows future estimation of the LID of new observations relative to a training reference set.

---

## 4. Unit Testing

Technical validation was implemented in:

```text
lid_analysis/tests/test_lid_unit.py
```

These tests use small synthetic datasets generated directly in memory.

They are intended to answer:

> Does the implementation behave technically as expected?

The tests verify:

1. exactly one LID estimate is returned per observation;
2. all estimates are finite;
3. all estimates are positive;
4. repeated execution on identical data produces identical results;
5. invalid neighbourhood sizes are rejected.

Execution:

```bash
python -m pytest lid_analysis/tests/test_lid_unit.py -v
```

Result:

```text
4 passed
```

This confirms the technical correctness of the basic implementation.

---

## 5. Synthetic Validation

A second validation stage was implemented in:

```text
lid_analysis/tests/test_lid_synthetic.py
```

Unlike the unit tests, this stage evaluates the behaviour of the estimator.

Synthetic Gaussian datasets with known dimensionalities were generated:

```text
2D, 5D, 10D, 20D
```

The objective was to verify that:

- estimated LID increases when dimensionality increases;
- the estimates remain positive and finite;
- estimates in low-dimensional settings remain reasonably close to the known dimensionality.

Execution:

```bash
python -m pytest lid_analysis/tests/test_lid_synthetic.py -v
```

Result:

```text
3 passed
```

The synthetic validation therefore confirmed that MLE-LID responds in the expected direction before being applied to the ADERH experimental setting.

---

## 6. Integration with the ADERH Baseline

The main experimental analysis is implemented in:

```text
lid_analysis/analyze_lid_baseline.py
```

A central methodological decision was to avoid creating a separate synthetic scenario.

Instead, the LID analysis deliberately reuses the existing:

```text
baseline_experiment/
```

protocol.

The following baseline components are reused directly:

- `make_nested_datasets()`;
- `DEFAULT_DIMENSIONS`;
- `FULL_SEEDS`;
- `SMOKE_DIMENSIONS`;
- `_split_indices()`;
- `scale_train_test()`.

This ensures direct comparability between the LID results and the previously obtained ADERH results.

---

## 7. Experimental Protocol

The controlled baseline contains:

```text
2000 observations
5% anomalies
2 informative dimensions
additional independent Gaussian noise dimensions
```

The dimensionalities analysed are:

```text
2, 5, 10, 20, 50, 100, 200, 500
```

The complete analysis uses:

```text
seeds 0–9
```

giving:

```text
8 dimensions × 10 seeds = 80 configurations
```

For each seed:

1. the nested synthetic datasets are regenerated;
2. the same stratified 70/30 train/test split is reproduced;
3. the same indices are reused across dimensionalities;
4. MinMaxScaler is fitted only on the training split;
5. LID is calculated for all scaled training observations.

The training set contains:

```text
1400 observations
```

per configuration.

Therefore, the full experiment produces:

```text
80 × 1400 = 112,000 point-level LID estimates
```

---

## 8. Use of Ground-Truth Labels

Ground-truth labels are used only to reproduce the stratified train/test split of the original baseline.

They are **not** supplied to the LID estimator.

Conceptually:

```text
Labels
   ↓
Reproduce baseline split only

X_train_scaled
   ↓
MLE-LID
   ↓
LID values
```

This preserves the unsupervised nature required for future integration with ADERH.

---

## 9. Smoke Experiment

Before running the complete experiment, a smoke configuration was executed using:

```text
seed = 0
dimensions = 2, 50
```

Results:

```text
2D
Median LID = 2.156
Mean LID   = 2.252

50D
Median LID = 28.168
Mean LID   = 29.183
```

This confirmed that:

- the baseline imports worked correctly;
- the data-generation pipeline was reproduced;
- preprocessing was working;
- MLE-LID could be calculated successfully;
- estimated LID increased substantially from 2D to 50D.

The corresponding smoke values were later confirmed to match exactly the same configurations in the full experiment.

---

## 10. Full Experiment Outputs

The complete experiment generates four main outputs.

### 10.1 `full_lid_points.csv`

Stores all individual LID estimates.

Each row contains:

```text
dimensionality
seed
k
train_position
original_index
lid
```

This file contains the complete point-level distribution and preserves all 112,000 LID estimates.

It supports future:

- histograms;
- boxplots;
- distribution analysis;
- local diagnostics;
- point-dependent pairing policies.

---

### 10.2 `full_lid_by_seed.csv`

Stores one row per dimensionality/seed configuration.

For each configuration it contains:

```text
Mean LID
Median LID
Standard deviation
Minimum
Q25
Q75
Maximum
```

This file is particularly useful for studying reproducibility and variability between seeds.

---

### 10.3 `full_lid_summary.csv`

Aggregates the results across all ten seeds for each dimensionality.

This is the main file used for the global analysis of LID behaviour.

---

### 10.4 `full_lid_metadata.json`

Stores the complete experiment configuration, including:

- estimator;
- neighbourhood size;
- distance metric;
- dimensions;
- seeds;
- dataset generator;
- preprocessing;
- split;
- LID population;
- package versions;
- Python version;
- operating system.

This file was included to support reproducibility.

---

## 11. Consistency Checks

The generated files were checked for internal consistency.

The full experiment contains:

```text
80 dimensionality/seed configurations
112,000 individual LID estimates
```

No invalid, negative, infinite, or duplicated point-level estimates were identified.

The descriptive statistics recalculated from `full_lid_points.csv` agree with those stored in `full_lid_by_seed.csv`.

Likewise, aggregation of the seed-level results agrees with `full_lid_summary.csv`.

The smoke and full runs also produce identical point-level results for the overlapping:

```text
seed 0 / 2D
seed 0 / 50D
```

configurations.

This confirms consistent execution of the experimental pipeline.

---

## 12. Main Results

The aggregated LID results for `k = 20` are:

| Dimensionality | Mean LID | Median LID |
|---:|---:|---:|
| 2 | 2.269 | 2.166 |
| 5 | 5.502 | 5.274 |
| 10 | 9.795 | 9.423 |
| 20 | 16.236 | 15.640 |
| 50 | 29.321 | 28.320 |
| 100 | 44.142 | 42.674 |
| 200 | 65.029 | 62.928 |
| 500 | 107.492 | 104.007 |

The main trend is:

```text
Increasing dimensionality
        ↓
Increasing estimated LID
```

For every one of the ten seeds:

```text
LID(2D)
    <
LID(5D)
    <
LID(10D)
    <
LID(20D)
    <
LID(50D)
    <
LID(100D)
    <
LID(200D)
    <
LID(500D)
```

The Spearman correlation between dimensionality and mean/median LID is:

```text
ρ = 1.0
```

for every seed.

This indicates an extremely consistent monotonic relationship in the controlled experimental setting.

---

## 13. Stability Across Random Seeds

The estimator shows low variability across the ten seeds.

Approximate standard deviations of Mean LID are:

```text
2D    → 0.012
10D   → 0.050
50D   → 0.145
100D  → 0.346
200D  → 0.793
500D  → 1.976
```

Absolute variability increases as LID becomes larger, but relative variability remains low.

The coefficient of variation remains approximately between:

```text
0.5% and 1.8%
```

for Mean LID across the analysed settings.

Even at 500 dimensions, relative variability remains below approximately 2%.

Therefore, the observed dimensional trend is stable and not driven by individual random seeds.

---

## 14. Behaviour in High Dimensions

In lower-dimensional settings, the MLE-LID estimate is relatively close to the dimensionality of the synthetic construction.

Examples:

```text
2D  → Mean LID ≈ 2.27
5D  → Mean LID ≈ 5.50
10D → Mean LID ≈ 9.79
```

However, the estimator increasingly underestimates the absolute dimensionality at higher dimensions:

```text
20D  → Mean LID ≈ 16.24
50D  → Mean LID ≈ 29.32
100D → Mean LID ≈ 44.14
200D → Mean LID ≈ 65.03
500D → Mean LID ≈ 107.49
```

The approximate ratio between Mean LID and total dimensionality decreases substantially:

```text
2D   → ~1.13
5D   → ~1.10
10D  → ~0.98
20D  → ~0.81
50D  → ~0.59
100D → ~0.44
200D → ~0.33
500D → ~0.21
```

This means the estimated LID should not be interpreted as an exact reconstruction of absolute dimensionality in high-dimensional regimes.

For this project, a more appropriate interpretation is:

> **LID is used as a relative indicator of local dimensional complexity.**

The relevant information is therefore the ordering and local variation of LID rather than exact equality with the synthetic dimensionality.

---

## 15. Point-Level Distribution

The complete distributions are preserved in:

```text
full_lid_points.csv
```

The point-level values show that observations within the same global dimensionality can have different estimated LID values.

Across the analysed dimensions:

```text
Mean LID > Median LID
```

which indicates some right-skewness in the distributions.

For example, at 500 dimensions the typical distribution is approximately:

```text
Q25     ≈ 90.15
Median  ≈ 104.01
Q75     ≈ 120.67
```

This point-level variation is important for the future adaptive mechanism.

If all observations had almost identical LID:

```text
LID(x1) ≈ LID(x2) ≈ ... ≈ LID(xn)
```

then using a point-dependent parameter would provide little information.

Instead, the observed local variation provides motivation for studying:

```text
α(x) = f(LID(x))
```

at observation level.

---

## 16. Figures

Three main figures were generated and stored in:

```text
lid_analysis/figures/
```

### `lid_evolucao_dimensionalidade.png`

Shows mean and median LID as dimensionality increases.

It highlights both:

```text
Dimensionality ↑
        ↓
Estimated LID ↑
```

and the increasing difference between total dimensionality and estimated LID in high-dimensional regimes.

---

### `lid_quartis_dimensionalidade.png`

Shows:

```text
Q25
Median
Q75
```

for each dimensionality.

This demonstrates both the global increase in LID and the presence of point-level variation.

---

### `lid_estabilidade_seeds.png`

Shows the coefficient of variation across the ten seeds.

This figure supports the conclusion that the LID trend is stable across repeated experimental runs.

---

## 17. Current Conclusions

The current experiment establishes that, within the controlled synthetic ADERH baseline:

1. MLE-LID can be computed reliably for every training observation.
2. Estimated LID increases monotonically with dimensionality.
3. The trend occurs in all ten random seeds.
4. Variability between seeds remains low.
5. Point-level LID distributions contain meaningful local variation.
6. Absolute dimensionality is increasingly underestimated in high-dimensional regimes.

Therefore, the current evidence supports using MLE-LID as a:

> **stable relative indicator of local dimensional complexity**

for the next stage of the project.

---

## 18. What Has Not Yet Been Established

The current analysis does **not yet demonstrate**:

```text
LID ↑
    ↓
ADERH performance ↓
```

nor does it demonstrate:

```text
LID ↑
    ↓
More Random Pairing
```

The second relationship is a working research hypothesis, not an established consequence of LID.

The next experiments must determine whether LID is actually associated with the geometric failure mode observed in ADERH.

---

## 19. Next Experimental Stage

The next analysis will investigate:

```text
LID ↔ Relative Contrast

LID ↔ Uncovered Fraction

LID ↔ ROC-AUC

LID ↔ Average Precision
```

The objective is to determine whether regions or configurations with greater local dimensional complexity are associated with:

- stronger distance concentration;
- lower hypersphere coverage;
- reduced anomaly discrimination;
- lower ADERH predictive performance.

Only after this relationship is understood should LID be used to define the adaptive pairing policy.

Further work will then include:

- sensitivity analysis for `k`;
- Local Pairing;
- Fixed Hybrid Pairing;
- definition of `α(x) = f(LID(x))`;
- LID-Aware Adaptive Pairing;
- ablation studies;
- interpretability analysis.

---

## 20. References

- Levina, E., & Bickel, P. J. (2004). *Maximum Likelihood Estimation of Intrinsic Dimension*. NeurIPS.
- Amsaleg, L. et al. (2015). *Estimating Local Intrinsic Dimensionality*. KDD.
- Houle, M. E., Schubert, E., & Zimek, A. (2018). *On the Correlation Between Local Intrinsic Dimensionality and Outlierness*.
- Anderberg et al. (2024). *Dimensionality-Aware Outlier Detection*. SDM.