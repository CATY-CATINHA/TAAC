"""Paired retest of the frozen random_subspace_10_shuffled detector."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from .data_generation import DEFAULT_DIMENSIONS, make_nested_datasets
from .experiments import MODEL_CONFIGS, scale_train_test, split_indices

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "data/adbench_manifest.csv"
CACHE = Path("pairing/experiments/data/adbench")
OUTPUT = Path("pairing/experiments/results/current/pre_subspace")
ADBENCH_REV = "3dac8221081e190f157d78e93bfa8867f90d0965"
SOURCE = f"https://raw.githubusercontent.com/Minqi824/ADBench/{ADBENCH_REV}/adbench/datasets/Classical/"
CANDIDATE = "random_subspace_10_shuffled"
MODELS = ("original", CANDIDATE)
REAL_SEEDS = (0, 1, 2, 100, 1000)
NEW_SEEDS = range(10, 30)


def verified_data(row, cache, download=False):
    path = cache / row.file
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} is missing. Run pre_subspace fetch first.")
        cache.mkdir(parents=True, exist_ok=True)
        with urlopen(SOURCE + row.file, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != row.sha256:
            raise ValueError(f"Unexpected download checksum for {path.name}")
        with NamedTemporaryFile(dir=cache, delete=False) as temp:
            temp.write(content)
        Path(temp.name).replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != row.sha256:
        raise ValueError(f"Unexpected cached checksum for {path.name}")
    with np.load(path, allow_pickle=False) as archive:
        X, y = archive["X"], archive["y"]
    if (X.shape != (row.n, row.d) or y.shape != (row.n,)
            or not np.isin(y, [0, 1]).all() or y.sum() != row.anomalies
            or not np.isfinite(X).all()):
        raise ValueError(f"Data disagree with the manifest for {row.dataset}")
    return X, y


def paired_rows(dataset, X, y, train, test, split, seeds):
    X_train, X_test, _ = scale_train_test(X[train], X[test])
    for seed in seeds:
        for name in MODELS:
            config = MODEL_CONFIGS[name]
            detector = config["class"](**config["parameters"], random_state=seed)
            detector.fit(X_train)
            scores = detector.decision_function(X_test)
            yield dict(dataset=dataset, dimensionality=X.shape[1], split=split,
                       seed=seed, model=name, n_train=len(train), n_test=len(test),
                       test_anomalies=int(y[test].sum()),
                       average_precision=average_precision_score(y[test], scores),
                       roc_auc=roc_auc_score(y[test], scores),
                       uncovered_fraction=np.mean(scores == 1))


def run_suite(suite, cache):
    if suite == "synthetic":
        for seed in NEW_SEEDS:
            print(f"Synthetic seed {seed}", flush=True)
            for d, data in make_nested_datasets(DEFAULT_DIMENSIONS, random_state=seed).items():
                train, test = split_indices(data.y, random_state=seed)
                yield from paired_rows(f"{d}D", data.X, data.y, train, test, 0, [seed])
    else:
        for row in pd.read_csv(MANIFEST).itertuples():
            X, y = verified_data(row, cache)
            print(row.dataset, flush=True)
            splitter = StratifiedShuffleSplit(n_splits=3, test_size=0.30, random_state=0)
            for split, (train, test) in enumerate(splitter.split(X, y)):
                yield from paired_rows(row.dataset, X, y, train, test, split, REAL_SEEDS)


def wdbc_reference(cache):
    """Explain the WDBC reference gap by changing scaling and model seeds."""
    row = pd.read_csv(MANIFEST).query("dataset == 'WDBC'").iloc[0]
    X, y = verified_data(row, cache)
    public_script_seeds = (0, 1, 2, 1000, 10000)
    # Match the two transformations in the public authors' script.
    global_X = MinMaxScaler().fit_transform(MinMaxScaler().fit_transform(X))
    config = MODEL_CONFIGS["original"]
    rows = []
    splitter = StratifiedShuffleSplit(n_splits=3, test_size=0.30, random_state=0)
    for split, (train, test) in enumerate(splitter.split(X, y)):
        train_X, test_X, _ = scale_train_test(X[train], X[test])
        for scope, a, b in (("train", train_X, test_X),
                            ("global_twice", global_X[train], global_X[test])):
            for seed in sorted(set(REAL_SEEDS) | set(public_script_seeds)):
                model = config["class"](**config["parameters"], random_state=seed)
                scores = model.fit(a).decision_function(b)
                rows.append(dict(scaler=scope, split=split, seed=seed,
                                 paper_seed=seed in REAL_SEEDS, script_seed=seed in public_script_seeds,
                                 average_precision=average_precision_score(y[test], scores),
                                 roc_auc=roc_auc_score(y[test], scores)))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["fetch", "synthetic", "real", "wdbc-reference"])
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.action == "fetch":
        for row in pd.read_csv(MANIFEST).itertuples():
            verified_data(row, args.cache, download=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == "wdbc-reference":
        wdbc_reference(args.cache).to_csv(args.output / "wdbc_reference.csv", index=False)
        return
    results = pd.DataFrame(run_suite(args.action, args.cache))
    results.to_csv(args.output / f"{args.action}_runs.csv", index=False)
    info = dict(suite=args.action, python=platform.python_version(),
                packages={p: version(p) for p in ("aderh", "numpy", "scipy", "scikit-learn", "pandas")},
                models={m: MODEL_CONFIGS[m]["parameters"] for m in MODELS},
                model_seeds=list(REAL_SEEDS if args.action == "real" else NEW_SEEDS),
                split="stratified 70/30; real: three splits with RNG 0; synthetic: one split per data seed",
                scaler="MinMaxScaler fit to training only", adbench_revision=ADBENCH_REV,
                manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest())
    (args.output / f"{args.action}_metadata.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main()
