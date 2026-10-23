# LID-Adaptive Pairing for ADERH

Extension of **ADERH — Anomaly Detection by an Ensemble of Random Pairs of Hyperspheres** for improved robustness in high-dimensional data.

## Project Overview

ADERH is an unsupervised anomaly detection method based on ensembles of hyperspheres generated from random pairs of observations.

The original paper identifies **high dimensionality and distance concentration** as an important limitation: as the intrinsic dimensionality of the data increases, local neighborhoods may become less informative and the geometric cues used by ADERH may lose discriminative power.

This project investigates whether **Local Intrinsic Dimensionality (LID)** can be used to adapt the pair-selection mechanism of ADERH.

The main idea is to balance:

- **Local pairing**, which better preserves local structure;
- **Random pairing**, which preserves the original multi-scale behavior of ADERH.

Instead of using a fixed balance between both strategies, we aim to make the pairing policy adaptive to the local geometric complexity of the data.

---

## Proposed Method

The proposed method follows this pipeline:

```text
Input Data
    ↓
Estimate Local Intrinsic Dimensionality
    ↓
Determine Local / Random Pairing Balance
    ↓
Adaptive Pair Selection
    ↓
ADERH Hyperspheres
    ↓
Pitch + NDensity
    ↓
Anomaly Score
```

The adaptive pairing parameter is conceptually defined as:

```text
α(x) = f(LID(x))
```

where:

- higher `α(x)` means more local pairing;
- lower `α(x)` means more random pairing.

The exact function `f` will be determined and evaluated experimentally.

---

## Research Question

> **Can Local Intrinsic Dimensionality guide ADERH's pair selection to balance local geometric fidelity and multi-scale random coverage in high-dimensional data?**

---

## Baseline Experiment

Before introducing the proposed method, a controlled experiment was conducted using the original ADERH implementation.

The experiment progressively adds irrelevant dimensions while keeping the informative structure fixed.

The results show that, under this controlled setting:

- distance contrast decreases as dimensionality increases;
- ADERH predictive performance deteriorates;
- hypersphere coverage collapses;
- at very high dimensionality, anomaly scores become indistinguishable.

The complete baseline experiment, implementation, results, and documentation are available in:

```text
baseline_experiment/
```

---

## Experimental Roadmap

The project is organized into the following stages:

### 1. ADERH Baseline

Reproduce and characterize the original high-dimensional failure mode.

### 2. LID Analysis

Estimate **Local Intrinsic Dimensionality** and study its relationship with:

- distance contrast;
- hypersphere coverage;
- ADERH predictive performance.

### 3. Local Pairing

Replace fully random partners with locally selected neighbors as an intermediate baseline.

### 4. Hybrid Pairing

Combine local and random pairing using a fixed parameter `α`.

### 5. LID-Adaptive Pairing

Define:

```text
α(x) = f(LID(x))
```

so that the pairing strategy adapts to local geometric complexity.

### 6. Evaluation and Ablation Studies

Compare:

- Original ADERH;
- Local Pairing ADERH;
- Fixed Hybrid Pairing ADERH;
- LID-Adaptive Pairing ADERH.

### 7. Explainability / Interpretability

Analyze how:

- LID;
- adaptive pairing decisions;
- local/random pairing balance;
- feature contributions;

relate to the final anomaly score.

---

## Evaluation

The main evaluation metrics and diagnostics will include:

- **ROC-AUC**;
- **Average Precision / AUC-PR**;
- **runtime**;
- **stability across random seeds**;
- **distance contrast**;
- **hypersphere coverage**;
- performance across different **ambient dimensionalities**;
- performance across different **intrinsic dimensionalities**.

Where appropriate, statistical comparisons will also be performed.

---

## Repository Structure

```text
.
├── baseline_experiment/    # Original ADERH dimensionality baseline
├── lid_analysis/           # LID estimation and diagnostic analysis
├── pairing/                # Local, hybrid, and adaptive pairing methods
├── experiments/            # Main experimental pipelines
├── xai/                    # Explainability and interpretability analyses
├── results/                # Experimental results
├── figures/                # Generated figures and plots
├── docs/                   # Project documentation
└── README.md
```

---

## Selected Paper

**Durani, W., Leiber, C., Durani, K., Plant, C., & Böhm, C. (2025).**  
*Anomaly Detection by an Ensemble of Random Pairs of Hyperspheres.*  
**NeurIPS 2025**

Paper:  
https://proceedings.neurips.cc/paper_files/paper/2025/hash/f2b6bc2104f6f4481eef101f6a32c085-Abstract-Conference.html

Official ADERH implementation:  
https://github.com/Walid10010/ADERH

---
