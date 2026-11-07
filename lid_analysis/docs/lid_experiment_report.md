# LID Experiment Report

## 1. Objective

This document describes the implementation, validation, experimental analysis, and sensitivity study of the **Local Intrinsic Dimensionality (LID)** component.

The purpose of this stage is to determine whether LID provides a meaningful point-level measure of local dimensional complexity that can later be used to adapt ADERH's pair-selection strategy.

The future adaptive mechanism is conceptually based on:

```text
α(x) = f(LID(x))
```

where `α(x)` will determine the balance between Local Pairing and Random Pairing.

The LID stage therefore has three main objectives:

1. implement and validate an established LID estimator;
2. determine whether LID is associated with the dimensional failure mode observed in ADERH;
3. verify that the conclusions are robust to the neighbourhood size `k`.

---

## 2. Why LID Is Relevant to the Project

The controlled ADERH baseline previously established that increasing dimensionality is associated with:

- decreasing distance contrast;
- increasing hypersphere non-coverage;
- deteriorating ROC-AUC;
- deteriorating Average Precision;
- eventual collapse of anomaly-score discrimination.

However, total dimensionality is a **global property of the dataset**.

The proposed adaptive ADERH method requires a signal that can be calculated at the level of individual observations.

LID provides such a signal:

```text
Global dimensionality experiment
        ↓
ADERH degradation observed
        ↓
Need a local geometric indicator
        ↓
LID(x)
```

Each observation receives its own estimate of local dimensional complexity.

This creates the possibility of later defining:

```text
LID(x)
   ↓
α(x)
   ↓
Adaptive Pair Selection
```

---

## 3. Estimator Selection

The selected estimator is the **Maximum Likelihood Estimator of Local Intrinsic Dimensionality (MLE-LID)**.

For an observation `x`, the estimator considers the distances to its `k` nearest neighbours:

```text
r1 ≤ r2 ≤ ... ≤ rk
```

and estimates dimensionality from the local growth pattern of these distances.

The reference configuration is:

```text
Estimator:       MLE-LID
Distance:        Euclidean
k:               20
Self-neighbour:  excluded
Supervision:     unsupervised
Preprocessing:   MinMaxScaler fitted on training data only
```

The value `k = 20` is used as a reference configuration rather than as an assumed optimal value.

The estimator choice allows the project to focus its contribution on the **adaptive ADERH pairing mechanism**, rather than introducing a new intrinsic-dimensionality estimator.

---

## 4. MLE-LID Implementation

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

The implementation:

- validates input dimensionality;
- rejects NaN and infinite values;
- checks whether `k` is valid;
- excludes the observation itself from its neighbourhood;
- applies numerical stability safeguards;
- detects degenerate neighbourhoods.

The `reference` parameter also allows LID to be estimated for observations relative to a separate reference dataset.

---

## 5. Unit Testing

Technical validation is implemented in:

```text
lid_analysis/tests/test_lid_unit.py
```

The tests answer:

> Does the implementation behave technically as expected?

They verify:

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

This confirms the technical correctness of the implementation.

---

## 6. Synthetic Validation

A second validation stage is implemented in:

```text
lid_analysis/tests/test_lid_synthetic.py
```

Synthetic Gaussian datasets with known dimensionalities are generated:

```text
2D, 5D, 10D, 20D
```

The objective is to verify that:

- estimated LID increases as dimensionality increases;
- estimates remain positive and finite;
- low-dimensional estimates remain reasonably close to the known dimensionality.

Execution:

```bash
python -m pytest lid_analysis/tests/test_lid_synthetic.py -v
```

Result:

```text
3 passed
```

The synthetic validation therefore confirms that MLE-LID responds in the expected direction before being applied to the ADERH experimental setting.

---

## 7. Integration with the ADERH Baseline

The main LID experiment is implemented in:

```text
lid_analysis/analyze_lid_baseline.py
```

Rather than creating a new dimensionality experiment, the analysis deliberately reuses:

```text
baseline_experiment/
```

including:

- `make_nested_datasets()`;
- `DEFAULT_DIMENSIONS`;
- `FULL_SEEDS`;
- `SMOKE_DIMENSIONS`;
- `_split_indices()`;
- `scale_train_test()`.

This guarantees direct comparability between LID and the previously obtained ADERH baseline results.

---

## 8. Experimental Protocol

The controlled baseline contains:

```text
2000 observations
5% anomalies
2 informative dimensions
additional independent Gaussian noise dimensions
```

The analysed dimensionalities are:

```text
2, 5, 10, 20, 50, 100, 200, 500
```

with:

```text
seeds 0–9
```

giving:

```text
8 dimensions × 10 seeds = 80 configurations
```

For each seed:

1. the same nested synthetic datasets are regenerated;
2. the same stratified 70/30 split is reproduced;
3. the same indices are reused across dimensionalities;
4. MinMaxScaler is fitted only on the training split;
5. LID is calculated for all scaled training observations.

The training set contains:

```text
1400 observations
```

per configuration.

Therefore, the reference `k = 20` experiment produces:

```text
80 × 1400 = 112,000 point-level LID estimates
```

---

## 9. Use of Ground-Truth Labels

Ground-truth labels are used only to reproduce the same stratified train/test split as the ADERH baseline.

They are not supplied to the LID estimator.

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

## 10. Main LID Results

The aggregated results for the reference configuration `k = 20` are:

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

The observed relationship is:

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

This establishes a highly consistent monotonic relationship in the controlled experiment.

---

## 11. Stability Across Random Seeds

The reference `k = 20` estimator shows low variability across the ten seeds.

Approximate standard deviations of Mean LID include:

```text
2D    → 0.012
10D   → 0.050
50D   → 0.145
100D  → 0.346
200D  → 0.793
500D  → 1.976
```

Absolute variability increases because LID itself becomes larger.

However, relative variability remains low, with the coefficient of variation remaining approximately below 2%.

Therefore, the dimensional trend is stable and is not driven by a particular random seed.

---

## 12. Behaviour in High Dimensions

MLE-LID should not be interpreted as an exact reconstruction of total dimensionality.

In lower-dimensional settings, estimates are relatively close to the dimensionality of the synthetic construction:

```text
2D  → Mean LID ≈ 2.27
5D  → Mean LID ≈ 5.50
10D → Mean LID ≈ 9.79
```

At higher dimensions:

```text
20D  → Mean LID ≈ 16.24
50D  → Mean LID ≈ 29.32
100D → Mean LID ≈ 44.14
200D → Mean LID ≈ 65.03
500D → Mean LID ≈ 107.49
```

the estimator increasingly under-recovers the total dimensionality.

The appropriate interpretation in this project is therefore:

> **LID is a relative indicator of local dimensional complexity.**

The relevant information is the ordering and local variation of LID rather than exact equality with the total number of dimensions.

---

## 13. Point-Level Variation

The complete point-level distributions are stored in:

```text
full_lid_points.csv
```

Observations within the same dimensionality do not all receive identical LID values.

In general:

```text
Mean LID > Median LID
```

indicating some right-skewness in the distributions.

This local variation is important for the future adaptive method.

If:

```text
LID(x1) ≈ LID(x2) ≈ ... ≈ LID(xn)
```

then a point-dependent adaptive policy would contain little information.

Instead, the observed variation motivates studying:

```text
α(x) = f(LID(x))
```

at observation level.

---

# LID ↔ ADERH Relationship Analysis

## 14. Research Question

After validating the estimator, the next question was:

> **When LID increases, does the geometry used by ADERH become less informative?**

The analysis is implemented in:

```text
lid_analysis/analyze_lid_aderh_relationship.py
```

It does not rerun ADERH.

Instead, it combines:

```text
LID results
        +
existing ADERH baseline results
        ↓
merge by dimensionality + seed
```

This produces direct comparisons for the same 80 experimental configurations.

---

## 15. Analysed Relationships

Four relationships were investigated:

```text
LID ↔ Relative Contrast
LID ↔ Uncovered Fraction
LID ↔ ROC-AUC
LID ↔ Average Precision
```

These represent two complementary aspects.

### Geometry

```text
Relative Contrast
Uncovered Fraction
```

### Detection Performance

```text
ROC-AUC
Average Precision
```

This distinction is important because the objective is not merely to show that ADERH performance decreases.

The analysis also investigates whether the geometric conditions used by ADERH deteriorate at the same time.

---

## 16. Interpretation of the Metrics

### Relative Contrast

Relative Contrast measures how distinguishable nearby observations are from more distant observations.

High values indicate clearer distance structure.

Conceptually:

```text
High Relative Contrast
        ↓
near and far are clearly distinguishable
```

When Relative Contrast approaches 1:

```text
nearest distances ≈ other distances
```

and distance information becomes less discriminative.

---

### Uncovered Fraction

Uncovered Fraction represents the proportion of test observations that are not covered by any ADERH hypersphere.

Therefore:

```text
Uncovered Fraction ↑
        ↓
hypersphere coverage deteriorates
```

High values indicate that the geometric representation created by ADERH is failing to cover increasing portions of the data.

---

### ROC-AUC

ROC-AUC measures the overall ability of ADERH anomaly scores to rank anomalies above normal observations.

```text
ROC-AUC ≈ 1.0
→ strong discrimination

ROC-AUC ≈ 0.5
→ approximately random discrimination
```

---

### Average Precision

Average Precision summarizes precision/recall performance and is particularly useful when anomalies represent a small minority of the dataset.

In the controlled baseline:

```text
anomalies = 5%
```

Therefore, Average Precision provides an important view of how well ADERH identifies the minority anomaly class.

---

## 17. Relationship Results

Using values aggregated across the ten seeds for each dimensionality:

| Relationship | Spearman ρ | Direction |
|---|---:|---|
| Mean LID ↔ Relative Contrast | -1.000 | Negative |
| Mean LID ↔ Uncovered Fraction | +0.976 | Positive |
| Mean LID ↔ ROC-AUC | -1.000 | Negative |
| Mean LID ↔ Average Precision | -1.000 | Negative |

All four relationships are statistically very strong in this controlled setting.

More importantly, their direction is also reproduced across all ten random seeds.

```text
Relative Contrast:
10/10 seeds → negative relationship

Uncovered Fraction:
10/10 seeds → positive relationship

ROC-AUC:
10/10 seeds → negative relationship

Average Precision:
10/10 seeds → negative relationship
```

---

## 18. Global Interpretation

The observed pattern is:

```text
LID ↑
 │
 ├── Relative Contrast ↓
 │
 ├── Uncovered Fraction ↑
 │
 ├── ROC-AUC ↓
 │
 └── Average Precision ↓
```

This supports the following interpretation:

```text
Higher dimensional complexity
        ↓
Higher estimated LID
        ↓
Distance structure becomes less discriminative
        ↓
ADERH geometric representation deteriorates
        ↓
Anomaly-detection performance deteriorates
```

The four metrics therefore provide complementary evidence.

Relative Contrast shows deterioration in distance geometry.

Uncovered Fraction shows deterioration in hypersphere coverage.

ROC-AUC and Average Precision show the corresponding loss in anomaly-detection performance.

---

## 19. Non-Linearity of the Failure Mode

The relationship is strongly monotonic but should not be interpreted as necessarily linear.

The graphs show different phases of degradation.

Relative Contrast decreases strongly at relatively low dimensionalities and then approaches values close to 1.

Uncovered Fraction remains low during the earlier dimensional regimes but increases sharply at higher dimensionalities.

At approximately 200D and above, coverage is almost completely lost.

ROC-AUC and Average Precision also progressively deteriorate until the detector approaches non-informative behaviour.

This indicates that ADERH degradation is better interpreted as a changing geometric regime rather than as a simple linear decrease.

---

## 20. Important Causal Limitation

The observed correlations should not be interpreted as:

```text
LID ↑
causes
ADERH performance ↓
```

Dimensionality is a common experimental driver:

```text
              Dimensionality ↑
               /          \
              ↓            ↓
           LID ↑      ADERH degradation
```

Therefore, the experiment demonstrates:

> **association and consistency of trends**

rather than a causal effect of LID on ADERH.

The result that matters for the proposed extension is that LID consistently tracks the geometric regime in which ADERH becomes less effective.

---

# Sensitivity Analysis of k

## 21. Motivation

MLE-LID depends on the number of nearest neighbours used to estimate local dimensionality.

Therefore, conclusions obtained using only:

```text
k = 20
```

could potentially be an artefact of that specific neighbourhood size.

A sensitivity analysis was therefore performed using:

```text
k = 5, 10, 20, 30
```

The analysis is implemented in:

```text
lid_analysis/analyze_k_sensitivity.py
```

The objective is not to identify the value of `k` that produces the strongest correlation.

Instead, the question is:

> **Do the scientific conclusions remain stable when the local neighbourhood size changes?**

---

## 22. Sensitivity Protocol

For each value of `k`, LID is calculated using:

- the same dataset generator;
- the same dimensionalities;
- the same random seeds;
- the same train/test splits;
- the same MinMax preprocessing;
- the same training observations.

ADERH is not rerun.

The existing ADERH baseline metrics are reused.

Therefore:

```text
Same ADERH experiment
        ↑
        │
LID k=5
LID k=10
LID k=20
LID k=30
```

This isolates the effect of changing the LID neighbourhood size.

---

## 23. Sensitivity Results

The main result is that the direction of every LID ↔ ADERH relationship remains unchanged for all tested values of `k`.

| Metric | k=5 | k=10 | k=20 | k=30 |
|---|---:|---:|---:|---:|
| Relative Contrast | -1.000 | -1.000 | -1.000 | -1.000 |
| Uncovered Fraction | +0.976 | +0.976 | +0.976 | +0.976 |
| ROC-AUC | -1.000 | -1.000 | -1.000 | -1.000 |
| Average Precision | -1.000 | -1.000 | -1.000 | -1.000 |

In addition:

```text
all four relationships
        ×
all four k values
        ×
all ten random seeds
```

maintain the expected direction.

Therefore:

> **The main LID–ADERH conclusions are robust to the tested neighbourhood sizes.**

---

## 24. Effect of k on Absolute LID

Robustness of the relationship does not mean that `k` has no effect on the estimator.

Changing `k` changes the absolute magnitude and variability of MLE-LID estimates.

Conceptually:

```text
smaller k
   ↓
more local neighbourhood
   ↓
greater estimate variability
```

while larger neighbourhoods provide more smoothing.

Therefore:

```text
LID_k=5(x) ≠ LID_k=20(x)
```

in general.

However, the important finding is that the **relative dimensional ordering and the relationships with ADERH degradation remain stable**.

Thus, the correct conclusion is not:

> Changing `k` has no effect.

It is:

> **Changing `k` affects the numerical LID estimates, but does not alter the main scientific conclusions in the tested range.**

---

## 25. Why k Is Not Selected by Correlation Strength

This sensitivity study must not be used to choose:

```text
best k = k with strongest LID ↔ ADERH correlation
```

That would make the estimator selection dependent on the outcome the experiment is intended to explain.

Instead, `k = 20` remains a reference configuration.

The sensitivity experiment is used only to evaluate whether conclusions are robust to alternative reasonable neighbourhood sizes.

This analysis can later be reported as part of the project's **ablation/sensitivity study**.

---

## 26. Reproducibility Note

The sensitivity-analysis implementation can reuse the existing `k = 20` results generated during the original LID analysis.

For strict package-environment comparability across all four values of `k`, the reference experiment can also be recomputed using:

```bash
python -m lid_analysis.analyze_k_sensitivity --force-recompute-k20
```

This ensures that:

```text
k=5
k=10
k=20
k=30
```

are all generated under exactly the same current software environment.

---

# Generated Artifacts

## 27. Main LID Outputs

```text
lid_analysis/results/
├── full_lid_points.csv
├── full_lid_by_seed.csv
├── full_lid_summary.csv
└── full_lid_metadata.json
```

---

## 28. LID ↔ ADERH Outputs

```text
lid_analysis/results/relationships/
├── lid_aderh_merged.csv
├── relationship_by_dimension.csv
├── correlations_by_seed.csv
├── correlations_by_dimension.csv
└── relationship_summary.json
```

---

## 29. k-Sensitivity Outputs

```text
lid_analysis/results/k_sensitivity/
├── full_k5_*
├── full_k10_*
├── full_k30_*
├── k_sensitivity_lid_by_seed.csv
├── k_sensitivity_merged.csv
├── k_sensitivity_by_dimension.csv
├── k_sensitivity_correlations.csv
├── k_sensitivity_correlations_by_seed.csv
├── k_direction_consistency.csv
├── k_robustness_summary.csv
└── k_sensitivity_summary.json
```

---

# Figures

## 30. Main LID Figures

```text
lid_analysis/figures/
├── lid_evolucao_dimensionalidade.png
├── lid_quartis_dimensionalidade.png
└── lid_estabilidade_seeds.png
```

These show:

- LID growth with dimensionality;
- point-level distribution and quartiles;
- stability across random seeds.

---

## 31. LID ↔ ADERH Figures

```text
lid_analysis/figures/relationships/
├── lid_vs_relative_contrast.png
├── lid_vs_uncovered_fraction.png
├── lid_vs_roc_auc.png
└── lid_vs_average_precision.png
```

These figures visualize the relationship between increasing Mean LID and ADERH geometric/performance degradation.

---

## 32. k-Sensitivity Figures

```text
lid_analysis/figures/k_sensitivity/
├── lid_by_k_vs_dimensionality.png
└── correlations_by_k.png
```

The first shows how LID estimates change across dimensionality for different neighbourhood sizes.

The second shows whether the LID ↔ ADERH correlation structure remains stable across `k`.

---

# Overall Conclusions

## 33. What Has Been Established

Within the controlled ADERH dimensionality baseline, the current evidence establishes that:

1. MLE-LID can be calculated reliably for every training observation.
2. Estimated LID increases monotonically with dimensional complexity.
3. This trend is reproduced across all ten random seeds.
4. Point-level LID contains local variation.
5. Higher LID is associated with lower Relative Contrast.
6. Higher LID is associated with greater hypersphere non-coverage.
7. Higher LID is associated with lower ROC-AUC.
8. Higher LID is associated with lower Average Precision.
9. These relationships remain stable for `k = 5, 10, 20, 30`.

Therefore, the current evidence supports treating LID as a:

> **stable local geometric signal associated with the dimensional regime in which ADERH loses geometric informativeness and anomaly-detection performance.**

---

## 34. What Has Not Been Established

The experiments do not establish that:

```text
LID ↑
causes
ADERH degradation
```

and they do not yet establish that:

```text
LID ↑
        ↓
More Random Pairing
        ↓
Better ADERH
```

The mapping between LID and pairing behaviour remains a research hypothesis that must now be tested experimentally.

## 35. References

- Levina, E., & Bickel, P. J. (2004). *Maximum Likelihood Estimation of Intrinsic Dimension*. NeurIPS.
- Amsaleg, L. et al. (2015). *Estimating Local Intrinsic Dimensionality*. KDD.
- Houle, M. E., Schubert, E., & Zimek, A. (2018). *On the Correlation Between Local Intrinsic Dimensionality and Outlierness*.
- Anderberg et al. (2024). *Dimensionality-Aware Outlier Detection*. SDM.