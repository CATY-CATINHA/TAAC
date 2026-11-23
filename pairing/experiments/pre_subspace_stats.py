"""Recompute paired effects from both complete pre-subspace run tables."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .pre_subspace import CANDIDATE, DEFAULT_DIMENSIONS, MANIFEST, MODELS, NEW_SEEDS, OUTPUT, REAL_SEEDS

METRICS = ("average_precision", "roc_auc")
BOOTSTRAP_SEED = 20261123
BOOTSTRAP_REPEATS = 20000


def holm(p_values):
    p = np.asarray(p_values)
    order = np.argsort(p)
    adjusted = np.empty_like(p, dtype=float)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def effect(pairs):
    delta = (pairs[CANDIDATE] - pairs["original"]).to_numpy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.choice(delta, size=(BOOTSTRAP_REPEATS, len(delta)), replace=True)
    low, high = np.quantile(sampled.mean(axis=1), [0.025, 0.975])
    # Ignore differences below the precision retained for tie handling.
    signed = delta.round(12)
    wins = int((signed > 0).sum())
    losses = int((signed < 0).sum())
    p = binomtest(wins, wins + losses).pvalue if wins + losses else 1.0
    return dict(n_units=len(delta), original=pairs["original"].mean(),
                pre_subspace=pairs[CANDIDATE].mean(), delta_mean=delta.mean(),
                delta_median=np.median(delta), ci95_low=low, ci95_high=high,
                wins=wins, ties=len(delta) - wins - losses, losses=losses, sign_p=p)


def validate_grid(frame, datasets, splits, seeds):
    keys = ["dataset", "split", "seed", "model"]
    expected = pd.MultiIndex.from_product([datasets, splits, seeds, MODELS], names=keys)
    actual = pd.MultiIndex.from_frame(frame[keys])
    if not actual.is_unique or not expected.sort_values().equals(actual.sort_values()):
        raise ValueError("Incomplete, extra or duplicate paired observations")
    scores = frame[list(METRICS)].to_numpy()
    if not np.isfinite(scores).all() or not ((0 <= scores) & (scores <= 1)).all():
        raise ValueError("Invalid metric in raw observations")


def summarize(folder):
    real = pd.read_csv(folder / "real_runs.csv")
    synthetic = pd.read_csv(folder / "synthetic_runs.csv")
    manifest = pd.read_csv(MANIFEST).set_index("dataset")
    validate_grid(real, manifest.index, range(3), REAL_SEEDS)
    validate_grid(synthetic, [f"{d}D" for d in DEFAULT_DIMENSIONS], [0], NEW_SEEDS)
    real_rows, synthetic_rows, by_dataset = [], [], []
    for metric in METRICS:
        paired = real.pivot(index=["dataset", "split", "seed"], columns="model", values=metric)
        means = paired.groupby(level="dataset").mean()
        split_means = paired.groupby(level=["dataset", "split"]).mean()
        split_deltas = split_means[CANDIDATE] - split_means["original"]
        real_rows.append(dict(metric=metric, **effect(means)))
        for dataset, pair in means.iterrows():
            paper = manifest.loc[dataset, f"paper_{metric}"]
            by_dataset.append(dict(dataset=dataset, metric=metric, paper=paper,
                                   original=pair["original"], pre_subspace=pair[CANDIDATE],
                                   original_minus_paper=pair["original"] - paper,
                                   delta=pair[CANDIDATE] - pair["original"],
                                   split_delta_min=split_deltas.loc[dataset].min(),
                                   split_delta_max=split_deltas.loc[dataset].max()))
        for d in DEFAULT_DIMENSIONS:
            rows = synthetic[synthetic.dataset == f"{d}D"]
            pairs = rows.pivot(index="seed", columns="model", values=metric)
            synthetic_rows.append(dict(dimensionality=d, metric=metric, **effect(pairs)))
    for rows, filename in ((real_rows, "real_effects"), (synthetic_rows, "synthetic_effects")):
        table = pd.DataFrame(rows)
        table["sign_p_holm"] = holm(table.sign_p)
        table.to_csv(folder / f"{filename}.csv", index=False)
    pd.DataFrame(by_dataset).to_csv(folder / "real_by_dataset.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    summarize(parser.parse_args().output)
