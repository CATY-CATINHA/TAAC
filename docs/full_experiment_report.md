# Full experiment report: distance concentration in ADERH 1.0.0

## Status and scope

The full controlled experiment was completed on 2026-10-11 with 80 independent
configurations: eight dimensionalities and ten seeds. The official
`aderh==1.0.0` package was used unchanged, with its documented defaults and an
explicit `random_state`.

This experiment provides a reproducible empirical demonstration of a
high-dimensional failure mode under the protocol below. It does not claim that
ADERH must fail on every high-dimensional dataset or under every possible
hyperparameter setting.

## Controlled protocol

- 2,000 observations per seed, with exactly 5% anomalies.
- Two informative coordinates fixed within each seed.
- Independent `N(0, 1)` coordinates appended to reach total dimensionalities
  `[2, 5, 10, 20, 50, 100, 200, 500]`.
- Identical observations, labels, column prefixes, and stratified train/test
  indices across dimensionalities within a seed.
- `MinMaxScaler` fitted on training data only. Because it is feature-wise,
  appending columns does not alter the scaled values of existing columns.
- Unsupervised ADERH fitting; labels used only for splitting and evaluation.
- Official defaults: `n_estimators=256`, `n=18`, `contamination=0.1`.
- Ten seeds: `0` through `9`.

The only systematic within-seed intervention is therefore the addition of
irrelevant coordinates.

## Main results

Values are mean ± population standard deviation across the ten seeds.

| Dimensions | Relative Contrast | ROC-AUC | Average Precision | Uncovered test fraction |
|---:|---:|---:|---:|---:|
| 2 | 17.0685 ± 0.4271 | 0.9993 ± 0.0010 | 0.9833 ± 0.0244 | 0.0000 ± 0.0000 |
| 5 | 3.8457 ± 0.0375 | 0.9945 ± 0.0032 | 0.8958 ± 0.0706 | 0.0000 ± 0.0000 |
| 10 | 2.2594 ± 0.0134 | 0.9839 ± 0.0074 | 0.7978 ± 0.0764 | 0.0000 ± 0.0000 |
| 20 | 1.6829 ± 0.0055 | 0.9506 ± 0.0234 | 0.6061 ± 0.0797 | 0.0002 ± 0.0005 |
| 50 | 1.3520 ± 0.0031 | 0.8780 ± 0.0323 | 0.3023 ± 0.0777 | 0.0158 ± 0.0130 |
| 100 | 1.2271 ± 0.0030 | 0.6990 ± 0.0342 | 0.0840 ± 0.0092 | 0.4725 ± 0.0804 |
| 200 | 1.1512 ± 0.0026 | 0.5018 ± 0.0014 | 0.0502 ± 0.0001 | 0.9967 ± 0.0026 |
| 500 | 1.0901 ± 0.0017 | 0.5000 ± 0.0000 | 0.0500 ± 0.0000 | 1.0000 ± 0.0000 |

The Average Precision random-ranking baseline is the test prevalence, 0.05.

## Evidence chain

### 1. Distances concentrate

Mean Relative Contrast decreases from 17.0685 at two dimensions to 1.0901 at
500 dimensions, a 93.61% reduction. A value approaching one means that a random
reference distance and a nearest-neighbor distance have become similar in
relative magnitude. Relative Contrast decreases strictly with every added
dimensionality step in all 10 seeds.

### 2. Predictive separation degrades with the same intervention

ROC-AUC and Average Precision are non-increasing across all eight
dimensionalities in every one of the 10 seeds. Across dimensionality-level
means, Spearman correlation is 1.0 between Relative Contrast and ROC-AUC and
also 1.0 between Relative Contrast and Average Precision. Seed-level
correlations between Relative Contrast and either predictive metric range from
0.9940 to 1.0.

These correlations are descriptive rather than an independent causal test:
both quantities vary with dimensionality. Their value is that they establish a
consistent association across every controlled replicate.

### 3. The ADERH scoring mechanism collapses

ADERH initializes every per-estimator contribution to one. A covered point gets
a contribution strictly below one; therefore a final score exactly equal to
one identifies a point uncovered by every sphere in every ensemble member.

- At 2, 5, and 10 dimensions, each test set has 600 distinct scores and no
  exactly-uncovered observations.
- At 100 dimensions, 47.25% of test observations are uncovered on average:
  45.49% of normals and 80.67% of anomalies.
- At 200 dimensions, 99.67% of all test observations are uncovered; all
  anomalies are uncovered, but so are 99.65% of normals. Only two distinct
  scores remain on average.
- At 500 dimensions, all 6,000 test observations across all seeds receive the
  identical score `1.0`. Score standard deviation and the mean anomaly–normal
  score gap are both exactly zero.

Constant scores mathematically force ROC-AUC to 0.5 and Average Precision to
the class prevalence. This exactly matches all ten 500-dimensional runs.

## Conclusion

For this controlled synthetic regime, the high-dimensionality limitation is
empirically demonstrated and mechanistically characterized:

1. appending only irrelevant independent dimensions drives Relative Contrast
   toward one;
2. predictive ranking degrades consistently across all seeds;
3. hypersphere coverage collapses;
4. the original ADERH scoring function saturates at one for every observation,
   erasing all anomaly/normal separation.

Because informative coordinates, row order, split indices, preprocessing of
existing columns, algorithm implementation, and hyperparameters are held fixed,
the observed within-seed change is attributable to the appended irrelevant
dimensions under this protocol. The evidence supports distance concentration
as the geometric explanation and loss of hypersphere coverage as the immediate
ADERH failure mechanism.

## Limitations

- The conclusion is specific to this synthetic distribution, sample size,
  contamination, MinMax preprocessing, and ADERH 1.0.0 defaults.
- Relative Contrast co-varies with dimensionality; this experiment does not
  independently manipulate Relative Contrast.
- No hyperparameter search was performed, intentionally. The result describes
  the original default baseline, not the best achievable tuned configuration.
- Internal Pitch and NDensity values are not exposed by the official package.
  The coverage conclusion relies on the verified official score semantics, not
  on modified or instrumented ADERH code.
- NDensity is recomputed from the batch passed to `decision_function`, so ADERH
  scores are batch-composition dependent. Every configuration here scores its
  complete fixed 600-row test split in one call, keeping this behavior
  controlled across dimensionalities.
- Runtime values are machine- and load-dependent and should be interpreted as
  descriptive measurements only.

## Reproducibility and artifacts

```powershell
.venv\Scripts\python -m pytest
& .\.venv\Scripts\aderh-experiment.exe full
& .\.venv\Scripts\aderh-analyze.exe
```

- Raw configurations: `results/full_raw_results.csv`
- Means and standard deviations: `results/full_summary.csv`
- Environment and protocol: `results/full_metadata.json`
- Reproducible checks: `results/full_analysis.json`
- Per-seed trends: `results/full_seed_trends.csv`
- Figures: `figures/full_*.png` and `figures/full_*.pdf`
