# LID Analysis

This directory contains the **Local Intrinsic Dimensionality (LID)** component of the project.

The objective is to estimate the local dimensional complexity of each observation and investigate whether LID can later guide the pair-selection mechanism of ADERH.

The future adaptive mechanism is conceptually based on:

```text
α(x) = f(LID(x))
```

where:

- higher `α(x)` will correspond to more Local Pairing;
- lower `α(x)` will correspond to more Random Pairing.

At this stage, the focus is on implementing, validating, and analysing LID before integrating it into the pairing mechanism.

---

## LID Estimator

The selected estimator is the **Maximum Likelihood Estimator of Local Intrinsic Dimensionality (MLE-LID)**.

Current configuration:

- **Estimator:** MLE-LID
- **Distance metric:** Euclidean
- **Initial neighbourhood size:** `k = 20`
- **Self-neighbour:** excluded
- **Labels used for LID estimation:** no
- **Preprocessing:** MinMaxScaler fitted on training data only

The value `k = 20` is used as the initial reference configuration and is not assumed to be universally optimal.

A later sensitivity analysis will evaluate:

```text
k ∈ {10, 20, 30, 50}
```

---

## Directory Structure

```text
lid_analysis/
│
├── __init__.py
├── lid.py
├── analyze_lid_baseline.py
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
│   └── full_lid_metadata.json
│
├── figures/
│   ├── lid_evolucao_dimensionalidade.png
│   ├── lid_quartis_dimensionalidade.png
│   └── lid_estabilidade_seeds.png
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

Two complementary validation files are provided.

### Unit tests

```text
tests/test_lid_unit.py
```

These verify the technical correctness of the implementation:

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

### Synthetic validation

```text
tests/test_lid_synthetic.py
```

Synthetic Gaussian datasets are used to verify that estimated LID increases when known dimensionality increases.

Current tested dimensions:

```text
2D, 5D, 10D, 20D
```

Run with:

```bash
python -m pytest lid_analysis/tests/test_lid_synthetic.py -v
```

Current result:

```text
3 passed
```

---

## ADERH Baseline Integration

The main experiment is implemented in:

```text
analyze_lid_baseline.py
```

It reuses the experimental protocol from:

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

This ensures that LID is analysed on the same datasets, splits, seeds, and preprocessing used by the original ADERH baseline.

The complete experiment evaluates:

```text
Dimensions: 2, 5, 10, 20, 50, 100, 200, 500
Seeds:      0, 1, 2, 3, 4, 5, 6, 7, 8, 9

Total: 8 × 10 = 80 configurations
```

LID is estimated using **all training observations** and does not use anomaly labels.

---

## Running the Analysis

Install the baseline package from the project root:

```bash
python -m pip install -e baseline_experiment
```

Run the smoke experiment first:

```bash
python -m lid_analysis.analyze_lid_baseline smoke
```

Then run the complete experiment:

```bash
python -m lid_analysis.analyze_lid_baseline full
```

---

## Generated Outputs

### `full_lid_points.csv`

Contains the complete point-level LID distribution.

Used for:

- distribution analysis;
- histograms and boxplots;
- point-level diagnostics;
- future adaptive pairing.

### `full_lid_by_seed.csv`

Contains one row for each dimensionality/seed configuration, including:

- mean;
- median;
- standard deviation;
- minimum;
- Q25;
- Q75;
- maximum.

Used mainly to analyse stability across seeds.

### `full_lid_summary.csv`

Aggregates the ten seeds for each dimensionality.

Used to analyse the overall relationship between dimensionality and estimated LID.

### `full_lid_metadata.json`

Stores the experiment configuration and environment information required for reproducibility.

---

## Main Results

Aggregated results for `k = 20`:

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

Main observation:

```text
Increasing dimensional complexity
        ↓
Increasing estimated LID
```

The trend is monotonic across all ten random seeds.

---

## Key Findings

- estimated LID increases strongly with dimensionality;
- the trend is consistent across all ten seeds;
- variability between seeds remains low;
- point-level LID values show local variation;
- MLE-LID increasingly underestimates absolute dimensionality at very high dimensions.

Therefore, in this project LID is currently interpreted as a:

> **relative indicator of local dimensional complexity**

rather than an exact estimator of the total dimensionality.

---

## References

- Levina, E., & Bickel, P. J. (2004). *Maximum Likelihood Estimation of Intrinsic Dimension*. NeurIPS.
- Amsaleg, L. et al. (2015). *Estimating Local Intrinsic Dimensionality*. KDD.
- Houle, M. E., Schubert, E., & Zimek, A. (2018). *On the Correlation Between Local Intrinsic Dimensionality and Outlierness*.
- Anderberg et al. (2024). *Dimensionality-Aware Outlier Detection*. SDM.
