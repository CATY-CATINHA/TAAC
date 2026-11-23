"""Run and compare the four frozen policies in the first LID gate experiment."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from pairing.subspace_pairing.lid_gate import LIDGateADERH
from .experiments import git_revision, scale_train_test
from .pre_subspace import (
    CACHE, CANDIDATE, DEFAULT_DIMENSIONS, HERE, MANIFEST, MODEL_CONFIGS,
    REAL_SEEDS, metric_row, run_suite,
)
from .pre_subspace_stats import METRICS, holm, paired_difference, validate_grid

GATE = "lid_gate_v1"
AMBIENT = "ambient_gate_10"
CONTROLS = ("original", CANDIDATE, AMBIENT)
POLICIES = (*CONTROLS, GATE)
SEEDS = range(30, 50)
DEFAULT_OUTPUT = Path("pairing/experiments/results/current/lid_gate_v1")


def evaluate(dataset, X, y, train, test, split, seeds):
    X_train, X_test, _ = scale_train_test(X[train], X[test])
    for seed in seeds:
        gate = LIDGateADERH(**MODEL_CONFIGS["original"]["parameters"], random_state=seed)
        gate.fit(X_train)
        chosen = gate.selected_model_
        other = CANDIDATE if chosen == "original" else "original"
        config = MODEL_CONFIGS[other]
        other_model = config["class"](**config["parameters"], random_state=seed)
        scores = {
            chosen: gate.decision_function(X_test),
            other: other_model.fit(X_train).decision_function(X_test),
        }
        # Each hard gate returns the exact scores of one frozen branch.
        routes = {"original": "original", CANDIDATE: CANDIDATE,
                  AMBIENT: CANDIDATE if X.shape[1] > 10 else "original", GATE: chosen}
        for name, branch in routes.items():
            row = metric_row(dataset, X.shape[1], len(train), y[test], split, seed,
                             name, scores[branch])
            yield dict(**row, selected_model=branch,
                       median_train_lid=gate.median_lid_ if name == GATE else np.nan,
                       n_valid_lid=gate.n_valid_lid_ if name == GATE else None,
                       route_reason=gate.route_reason_ if name == GATE else None)


def paired_effects(runs, suite):
    rows = []
    groups = runs.groupby("dataset") if suite == "synthetic" else [("panel", runs)]
    for dataset, group in groups:
        for metric in METRICS:
            scores = group.pivot(index=["dataset", "split", "seed"], columns="model", values=metric)
            if suite == "real":
                scores = scores.groupby(level="dataset").mean()
            for control in CONTROLS:
                delta = (scores[GATE] - scores[control]).to_numpy()
                rows.append(dict(dataset=dataset, metric=metric, control=control,
                                 control_mean=scores[control].mean(), gate_mean=scores[GATE].mean(),
                                 **paired_difference(delta)))
    effects = pd.DataFrame(rows)
    effects["sign_p_holm"] = holm(effects.sign_p)
    return effects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=["synthetic", "real"])
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    suite = args.suite
    args.output.mkdir(parents=True, exist_ok=True)
    revision = git_revision()
    runs = pd.DataFrame(run_suite(suite, args.cache, evaluate=evaluate, synthetic_seeds=SEEDS))
    if suite == "real":
        validate_grid(runs, pd.read_csv(MANIFEST).dataset, range(3), REAL_SEEDS, models=POLICIES)
        # Check both frozen policies against the actual archived checkpoint.
        baseline = pd.read_csv(HERE / "results/pre_subspace/real_runs.csv")
        keys = ["dataset", "split", "seed", "model"]
        metrics = [*METRICS, "uncovered_fraction"]
        now = runs[runs.model.isin(CONTROLS[:2])].set_index(keys)[metrics].sort_index()
        then = baseline.set_index(keys)[metrics].sort_index()
        pd.testing.assert_frame_equal(now, then, rtol=0, atol=1e-12)
    else:
        validate_grid(runs, [f"{d}D" for d in DEFAULT_DIMENSIONS], [0], SEEDS, models=POLICIES)
    effects = paired_effects(runs, suite)
    runs.to_csv(args.output / f"{suite}_runs.csv", index=False)
    effects.to_csv(args.output / f"{suite}_effects.csv", index=False)
    info = dict(revision=revision, suite=suite, policies=POLICIES, lid_k=20, threshold=10,
                model_seeds=list(REAL_SEEDS if suite == "real" else SEEDS),
                cp0_revision="a15791d", python=platform.python_version(),
                packages={p: version(p) for p in ("aderh", "numpy", "scipy", "scikit-learn", "pandas")},
                manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest())
    (args.output / f"{suite}_metadata.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main()
