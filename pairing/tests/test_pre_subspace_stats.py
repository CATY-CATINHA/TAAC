from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pairing.experiments import pre_subspace_stats as stats
from pairing.experiments.pre_subspace import CANDIDATE, verified_data
from pairing.experiments.pre_subspace_stats import effect, holm, validate_grid


def test_holm_restores_original_order_and_is_monotone():
    np.testing.assert_allclose(holm([0.04, 0.001, 0.03, 0.9]), [0.09, 0.004, 0.09, 0.9])


def test_effect_uses_pairs_and_handles_all_ties():
    pairs = pd.DataFrame({"original": [0.1, 0.7, 0.5], CANDIDATE: [0.1, 0.7, 0.5]})
    tied = effect(pairs)
    assert tied["ties"] == 3 and tied["sign_p"] == 1
    assert tied["ci95_low"] == tied["ci95_high"] == 0
    pairs[CANDIDATE] += 0.1
    gain = effect(pairs)
    assert gain["wins"] == 3 and gain["sign_p"] == 0.25
    np.testing.assert_allclose([gain["ci95_low"], gain["ci95_high"]], 0.1)


def test_incomplete_or_duplicate_grid_rejected():
    frame = pd.DataFrame(dict(dataset=["50D"]*2, split=[0, 0], seed=[10, 10],
                              model=["original", CANDIDATE], average_precision=[.5, .6],
                              roc_auc=[.7, .8]))
    validate_grid(frame, ["50D"], [0], [10])
    for invalid in (frame.iloc[:1], pd.concat([frame, frame])):
        with pytest.raises(ValueError, match="paired"):
            validate_grid(invalid, ["50D"], [0], [10])


def test_fetch_verifies_download_and_rechecks_cache(tmp_path, monkeypatch):
    buffer = BytesIO()
    X = np.ones((3, 2))
    y = np.array([0, 1, 0])
    np.savez(buffer, X=X, y=y)
    blob = buffer.getvalue()
    row = SimpleNamespace(file="example.npz", dataset="example", n=3, d=2,
                          anomalies=1, sha256=sha256(blob).hexdigest())

    def source(url, timeout):
        return BytesIO(blob)

    monkeypatch.setattr("pairing.experiments.pre_subspace.urlopen", source)
    loaded_X, loaded_y = verified_data(row, tmp_path, download=True)
    np.testing.assert_array_equal(loaded_X, X)
    np.testing.assert_array_equal(loaded_y, y)
    (tmp_path / row.file).write_bytes(b"invalid")
    with pytest.raises(ValueError, match="checksum"):
        verified_data(row, tmp_path)


def test_real_repeated_splits_contribute_one_vote_per_dataset(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"dataset": ["a", "b"], "paper_roc_auc": [0.5, 0.5],
                  "paper_average_precision": [0.5, 0.5]}).to_csv(manifest, index=False)
    monkeypatch.setattr(stats, "MANIFEST", manifest)
    monkeypatch.setattr(stats, "REAL_SEEDS", [7, 8])
    monkeypatch.setattr(stats, "DEFAULT_DIMENSIONS", [2])
    monkeypatch.setattr(stats, "NEW_SEEDS", [10, 11])
    rows = []
    for dataset, gain in (("a", 0.2), ("b", -0.1)):
        for split in range(3):
            for seed in [7, 8]:
                for model in stats.MODELS:
                    score = 0.5 + (gain if model == CANDIDATE else 0)
                    rows.append(dict(dataset=dataset, split=split, seed=seed, model=model,
                                     average_precision=score, roc_auc=score))
    real = pd.DataFrame(rows)
    real.to_csv(tmp_path / "real_runs.csv", index=False)
    synthetic = real[(real.dataset == "a") & (real.split == 0)].copy()
    synthetic["dataset"] = "2D"
    synthetic["seed"] += 3
    synthetic.to_csv(tmp_path / "synthetic_runs.csv", index=False)
    stats.summarize(tmp_path)
    global_effect = pd.read_csv(tmp_path / "real_effects.csv")
    assert global_effect.n_units.tolist() == [2, 2]
    assert global_effect.wins.tolist() == global_effect.losses.tolist() == [1, 1]
    np.testing.assert_allclose(global_effect.delta_mean, 0.05)
