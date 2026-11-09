"""Regression tests for binary-SVM sequence selection."""

import gzip
import importlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.svm import SVC

from benchmarks.comparison_methods.data import CandidatePool
from benchmarks.comparison_methods.friedman import (
    SequenceKernel,
    binary_labels,
    entropy,
    method,
    read_fasta,
    run,
    select_entropy,
    top_targets,
)

shared = importlib.import_module("benchmarks.comparison_methods.run")
REFERENCE = Path(__file__).with_name("fixtures") / "friedman_binary_reference.json"
NATIVE = pytest.mark.skipif(
    shutil.which("c++") is None, reason="C++17 compiler required"
)


@pytest.fixture(scope="module")
def reference():
    return json.loads(REFERENCE.read_text())


@NATIVE
@pytest.mark.parametrize("seed", [0, 9, 60, 89])
def test_reference_trajectories_labels_and_selection(reference, seed):
    ids, scores = np.asarray(reference["ids"]), np.asarray(reference["scores"])
    labels = binary_labels(scores, ids)
    np.testing.assert_array_equal(labels, reference["labels"])
    kernel = SequenceKernel(reference["sequences"])
    try:
        observed = run(kernel, ids, scores, labels, seed, 5, stop_on_hit=False)
    finally:
        kernel.close()
    expected = reference["trajectories"][str(seed)]
    for key, value in expected.items():
        if key == "history":
            continue
        assert observed[key] == value
    for actual, entry in zip(observed["history"], expected["history"], strict=True):
        for key, value in entry.items():
            if key == "acquisition_entropies" and value is not None:
                np.testing.assert_allclose(actual[key], value, rtol=2e-12, atol=2e-12)
            else:
                assert actual[key] == value


def direct_similarity(x, y):
    complement = str.maketrans("ACGT", "TGCA")
    total = 0.0
    for i in range(len(x) - 5):
        a = x[i : i + 6]
        for j in range(len(y) - 5):
            b = y[j : j + 6]
            similarity = int(a == b) + int(a == b.translate(complement)[::-1])
            total += np.exp(-((i - j) ** 2) / 200) * similarity
    return total


@NATIVE
def test_kernel_matches_full_position_formula_and_palindrome_weight():
    long = "ACGTTGCA" * 52
    sequences = ["ACGTAC", "GTACGT", "ATGCAT", "A" * 20, long, long[7:] + "GATTACA"]
    kernel = SequenceKernel(sequences)
    try:
        raw = np.array(
            [[direct_similarity(x, y) for y in sequences] for x in sequences]
        )
        expected = raw / np.sqrt(raw.diagonal()[:, None] * raw.diagonal()[None, :])
        np.testing.assert_allclose(
            kernel.gram(np.arange(6), np.arange(6)), expected, rtol=2e-13, atol=2e-13
        )
        np.testing.assert_allclose(kernel.norm[2] ** 2, 2)
        np.testing.assert_allclose(
            kernel.gram([5, 1], [3, 0, 4]), expected[np.ix_([5, 1], [3, 0, 4])]
        )
    finally:
        kernel.close()
    with pytest.raises(RuntimeError, match="closed"):
        kernel.gram([0], [0])


def test_binary_labels_use_score_then_circuit_id():
    ids, scores = np.arange(103)[::-1], np.zeros(103)
    labels = binary_labels(scores, ids)
    assert np.bincount(labels).tolist() == [93, 10]
    assert sorted(ids[labels == 1]) == list(range(93, 103))
    order = np.random.default_rng(71).permutation(103)
    np.testing.assert_array_equal(
        binary_labels(scores[order], ids[order]), labels[order]
    )
    with pytest.raises(ValueError, match="unique"):
        binary_labels(scores, np.zeros(103))


def test_inclusive_endpoint_uses_floor_and_all_cutoff_ties():
    scores = np.arange(250, dtype=float)
    scores[-3:] = 999
    target, cutoff = top_targets(scores)
    assert cutoff == 999 and target.sum() == 3
    assert top_targets(np.arange(199))[0].sum() == 1


class SpyKernel:
    def gram(self, rows, cols):
        return np.asarray(rows)[:, None] == np.asarray(cols)[None, :]


def test_one_class_fallback_does_not_advance_probability_calibration_rng(monkeypatch):
    ids, scores = np.arange(200), np.arange(200, dtype=float)
    labels = binary_labels(scores, ids)
    seed = next(
        s
        for s in range(100)
        if len(set(labels[np.random.default_rng(s).choice(200, 12, replace=False)]))
        == 1
    )
    seeds, training = [], []

    def svc(**kwargs):
        seeds.append(kwargs["random_state"])
        model = SVC(**kwargs)
        fit = model.fit

        def recorded_fit(x, y):
            training.append(np.asarray(y).copy())
            return fit(x, y)

        model.fit = recorded_fit
        return model

    monkeypatch.setattr(method, "SVC", svc)
    result = run(SpyKernel(), ids, scores, labels, seed, 5, stop_on_hit=False)
    assert result["history"][1]["fitted_classes"] == [0]
    assert result["history"][1]["acquisition_entropies"] == [0] * 12
    expected_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    assert seeds == [int(expected_rng.integers(0, 2**31 - 1)) for _ in seeds]
    assert seeds
    for observed_labels in training:
        round_index = len(observed_labels) // 12
        selected_ids = sum(
            (row["circuit_ids"] for row in result["history"][:round_index]), []
        )
        np.testing.assert_array_equal(observed_labels, labels[selected_ids])


def test_entropy_uses_shannon_bits_and_seeded_phenotype_independent_ties():
    probabilities = np.array([[1, 0], [0.5, 0.5], [0.5, 0.5]])
    np.testing.assert_allclose(entropy(probabilities), [0, 1, 1])
    pool = np.array([4, 8, 12])
    rng = np.random.default_rng(np.random.SeedSequence([9, 1]))
    selected, values = select_entropy(pool, probabilities, rng, 2)
    keys = np.random.default_rng(np.random.SeedSequence([9, 1])).random(3)
    np.testing.assert_array_equal(
        selected, pool[np.lexsort((keys, -entropy(probabilities)))[:2]]
    )
    np.testing.assert_allclose(values, [1, 1])


def test_compressed_numeric_fasta_preserves_full_cassettes_and_id_alignment(tmp_path):
    path = tmp_path / "circuits.fasta.gz"
    with gzip.open(path, "wt") as f:
        f.write(">20 description\nacgtac\ngt\n>10\nATGCAT\n>99\nNNN\n")
    sequences = read_fasta(path, {10, 20})
    assert sequences == {20: "ACGTACGT", 10: "ATGCAT"}


@pytest.mark.parametrize(
    "text,wanted",
    [
        ("ACGTAC\n>0\nACGTAC\n", {0}),
        (">0\nACGTAC\n>0\nACGTAC\n", {0}),
        (">0\nACGTAC\n", {1}),
        (">0\nACGTAN\n", {0}),
    ],
)
def test_missing_ambiguous_or_duplicate_fasta_circuits_rejected(tmp_path, text, wanted):
    path = tmp_path / "invalid.fasta"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_fasta(path, wanted)


def toy_pool(task="IS"):
    return CandidatePool(
        task,
        "toy",
        np.arange(100),
        np.zeros(100),
        pd.DataFrame(),
        np.zeros((100, 1), dtype=int),
        {},
    )


@NATIVE
@pytest.mark.parametrize("task", ["IS", "OR", "AND"])
def test_shared_friedman_dispatch_defaults_to_gzip_and_stops_at_inclusive_hit(
    tmp_path, monkeypatch, task
):
    pool = toy_pool(task)
    monkeypatch.setattr(shared, "load_pool", lambda *args: pool)
    path = (
        tmp_path
        / "sequences"
        / f"{'IS' if task == 'IS' else 'logic'}_candidates.fasta.gz"
    )
    path.parent.mkdir()
    with gzip.open(path, "wt") as f:
        for ident in pool.ids[::-1]:
            f.write(f">{ident}\nACGTACGT\n")
    rows, summary = shared.run(
        "friedman", task, "toy", tmp_path, seed=0, batches=3, output=tmp_path / "out"
    )
    assert len(rows) == 12
    assert summary.iloc[0].first_inclusive_top1pct_round == 1
    assert summary.iloc[0].batches_evaluated == 1
    stem = f"friedman_{task}_toy_seed0"
    assert {p.name for p in (tmp_path / "out").iterdir()} == {
        f"{stem}_selections.csv",
        f"{stem}_summary.csv",
    }
    assert list(rows.columns) == [
        "method",
        "task",
        "dataset_name",
        "seed",
        "batch",
        "candidate_order",
        "circuit_id",
        "objective_score",
    ]


@pytest.mark.parametrize("name", ["merzbacher", "kobiela"])
def test_existing_methods_keep_ranked_stopping_even_when_inclusive_target_was_hit(
    tmp_path, monkeypatch, name
):
    pool = toy_pool()
    monkeypatch.setattr(shared, "load_pool", lambda *args: pool)
    fake = SimpleNamespace(
        select_batch=lambda pool, observed, available, *args, **kwargs: available[:12]
    )
    monkeypatch.setattr(shared.importlib, "import_module", lambda *args: fake)
    rows, summary = shared.run(
        name, "IS", "toy", tmp_path, seed=0, batches=3, output=tmp_path / name
    )
    assert summary.iloc[0].first_inclusive_top1pct_round == 1
    assert summary.iloc[0].batches_evaluated == 3
    assert len(rows) == 36


@pytest.mark.parametrize("batches", [0, -1])
def test_friedman_requires_a_positive_batch_budget(tmp_path, monkeypatch, batches):
    monkeypatch.setattr(shared, "load_pool", lambda *args: toy_pool())
    with pytest.raises(ValueError, match="positive batch budget"):
        shared.run(
            "friedman", "IS", "toy", tmp_path, batches=batches, output=tmp_path / "out"
        )
