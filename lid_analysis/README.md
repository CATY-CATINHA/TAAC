# LID Analysis

This directory contains the **Local Intrinsic Dimensionality (LID)** component of the project.

The objective is to estimate the local dimensional complexity of each observation and investigate whether LID can guide the pair-selection mechanism of ADERH.

The future adaptive mechanism is conceptually based on:

```text
α(x) = f(LID(x))
```

where:

- higher `α(x)` will correspond to more Local Pairing;
- lower `α(x)` will correspond to more Random Pairing.

The work in this module establishes whether LID provides a meaningful and robust geometric signal before it is integrated into the adaptive pairing mechanism.

---

## LID Estimator

The selected estimator is the **Maximum Likelihood Estimator of Local Intrinsic Dimensionality (MLE-LID)**.

Reference configuration:

- **Estimator:** MLE-LID
- **Distance metric:** Euclidean
- **Reference neighbourhood size:** `k = 20`
- **Self-neighbour:** excluded
- **Labels used for LID estimation:** no
- **Preprocessing:** MinMaxScaler fitted on training data only

The value `k = 20` is used as a reference configuration rather than as an assumed optimal value.

A sensitivity analysis was also performed with:

```text
k ∈ {5, 10, 20, 30}
```

---

## Directory Structure

```text
lid_analysis/
│
├── __init__.py
├── lid.py
├── analyze_lid_baseline.py
├── analyze_lid_aderh_relationship.py
├── analyze_k_sensitivity.py
├── README.md
│
├── tests/
│   ├── test_lid_unit.py
│   └── test_lid_synthetic.py
│
├── results/
│   ├── full_lid_points.csv
│   ├── full_lid_by_seed.csv
│   ├── full_lid_summary.csv
│   ├── full_lid_metadata.json
│   │
│   ├── relationships/
│   │   ├── lid_aderh_merged.csv
│   │   ├── relationship_by_dimension.csv
│   │   ├── correlations_by_seed.csv
│   │   ├── correlations_by_dimension.csv
│   │   └── relationship_summary.json
│   │
│   └── k_sensitivity/
│       ├── full_k5_*
│       ├── full_k10_*
│       ├── full_k30_*
│       ├── k_sensitivity_by_dimension.csv
│       ├── k_sensitivity_correlations.csv
│       ├── k_sensitivity_correlations_by_seed.csv
│       ├── k_direction_consistency.csv
│       ├── k_robustness_summary.csv
│       └── k_sensitivity_summary.json
│
├── figures/
│   ├── lid_evolucao_dimensionalidade.png
│   ├── lid_quartis_dimensionalidade.png
│   ├── lid_estabilidade_seeds.png
│   │
│   ├── relationships/
│   │   ├── lid_vs_relative_contrast.png
│   │   ├── lid_vs_uncovered_fraction.png
│   │   ├── lid_vs_roc_auc.png
│   │   └── lid_vs_average_precision.png
│   │
│   └── k_sensitivity/
│       ├── lid_by_k_vs_dimensionality.png
│       └── correlations_by_k.png
│
└── docs/
    └── lid_experiment_report.md
```

---

## Implementation

The main estimator is implemented in:

```text
lid.py
```

Main function:

```python
estimate_lid(X, k=20, reference=None)
```

Conceptually:

```text
Input observations
        ↓
k-nearest neighbours
        ↓
MLE-LID estimation
        ↓
One LID value per observation
```

The implementation includes input validation, self-neighbour exclusion, numerical stability checks, and detection of invalid neighbourhoods.

---

## Validation

Two complementary validation stages are provided.

### Unit Tests

```text
tests/test_lid_unit.py
```

They verify:

- one LID value per observation;
- positive and finite estimates;
- reproducibility;
- correct handling of invalid `k`.

Run with:

```bash
python -m pytest lid_analysis/tests/test_lid_unit.py -v
```

Current result:

```text
4 passed
```

### Synthetic Validation

```text
tests/test_lid_synthetic.py
```

Synthetic Gaussian datasets are used to verify that estimated LID behaves consistently as dimensionality increases.

Current result:

```text
3 passed
```

Run with:

```bash
python -m pytest lid_analysis/tests/test_lid_synthetic.py -v
```

---

## ADERH Baseline Integration

LID is evaluated on the **same controlled datasets used in the ADERH dimensionality baseline**.

The experiment reuses:

- the same synthetic generator;
- the same dimensionalities;
- the same random seeds;
- the same train/test splits;
- the same MinMax preprocessing.

The analysed dimensionalities are:

```text
2, 5, 10, 20, 50, 100, 200, 500
```

with:

```text
10 random seeds
```

giving:

```text
8 dimensions × 10 seeds = 80 configurations
```

LID is estimated on all training observations without using anomaly labels.

---

## Running the LID Analysis

Install the baseline package from the project root:

```bash
python -m pip install -e baseline_experiment
```

Run the main `k = 20` experiment:

```bash
python -m lid_analysis.analyze_lid_baseline full
```

---

## Main LID Results

Aggregated results for the reference configuration `k = 20`:

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
Increasing dimensional complexity
        ↓
Increasing estimated LID
```

The trend is monotonic across all ten random seeds.

MLE-LID is therefore treated in this project as a:

> **relative indicator of local dimensional complexity**

rather than an exact estimator of total dimensionality.

---

## LID ↔ ADERH Relationship

The relationship analysis is implemented in:

```text
analyze_lid_aderh_relationship.py
```

Run with:

```bash
python -m lid_analysis.analyze_lid_aderh_relationship
```

The analysis investigates:

```text
LID ↔ Relative Contrast
LID ↔ Uncovered Fraction
LID ↔ ROC-AUC
LID ↔ Average Precision
```

Results using dimensionality-level means:

| Relationship | Spearman ρ |
|---|---:|
| Mean LID ↔ Relative Contrast | -1.000 |
| Mean LID ↔ Uncovered Fraction | +0.976 |
| Mean LID ↔ ROC-AUC | -1.000 |
| Mean LID ↔ Average Precision | -1.000 |

The expected direction is also reproduced across all ten random seeds.

Overall:

```text
LID ↑
 │
 ├── Relative Contrast ↓
 ├── Uncovered Fraction ↑
 ├── ROC-AUC ↓
 └── Average Precision ↓
```

This shows that increasing LID is consistently associated with the dimensional regime in which ADERH's geometry becomes less informative and detection performance deteriorates.

These relationships are associations and should not be interpreted as evidence that LID itself causes ADERH degradation.

---

## Sensitivity to k

Because MLE-LID depends on the number of nearest neighbours used, a sensitivity analysis was performed with:

```text
k = 5, 10, 20, 30
```

The experiment is implemented in:

```text
analyze_k_sensitivity.py
```

Run with:

```bash
python -m lid_analysis.analyze_k_sensitivity
```

The analysis shows that changing `k` affects the **absolute magnitude of LID estimates**, but does not change the main scientific conclusions.

Across all tested values of `k`:

```text
LID ↔ Relative Contrast     negative
LID ↔ Uncovered Fraction    positive
LID ↔ ROC-AUC               negative
LID ↔ Average Precision     negative
```

The direction of all four relationships remains stable across all tested neighbourhood sizes and random seeds.

Therefore:

> **The observed relationship between LID and ADERH degradation is robust to the tested choice of neighbourhood size.**

The sensitivity analysis is used as a robustness/ablation analysis and not to select `k` according to the strongest correlation.

---

## Key Findings

The LID analysis currently establishes that:

- MLE-LID can be estimated reliably at observation level;
- estimated LID increases consistently with dimensional complexity;
- this trend is stable across random seeds;
- point-level LID contains local variation;
- increasing LID is associated with decreasing distance contrast;
- increasing LID is associated with increasing hypersphere non-coverage;
- increasing LID is associated with decreasing ROC-AUC and Average Precision;
- these conclusions remain stable for `k = 5, 10, 20, 30`.

Conceptually:

```text
Higher dimensional complexity
        ↓
Higher estimated LID
        ↓
Less informative distance geometry
        ↓
ADERH degradation
```

## References

- Levina, E., & Bickel, P. J. (2004). *Maximum Likelihood Estimation of Intrinsic Dimension*. NeurIPS.
- Amsaleg, L. et al. (2015). *Estimating Local Intrinsic Dimensionality*. KDD.
- Houle, M. E., Schubert, E., & Zimek, A. (2018). *On the Correlation Between Local Intrinsic Dimensionality and Outlierness*.
- Anderberg et al. (2024). *Dimensionality-Aware Outlier Detection*. SDM.