import gzip
import importlib

import numpy as np
import pandas as pd
import pytest

from benchmarks.comparison_methods.data import CandidatePool
from benchmarks.comparison_methods.yin import load_inputs, log_score, run

shared = importlib.import_module("benchmarks.comparison_methods.run")


@pytest.mark.parametrize(
    "task,expected",
    [
        ("IS", [3, -1]),
        ("OR", [1, -1]),
        ("AND", [-2, 2]),
    ],
)
def test_expression_objectives(task, expected):
    expression = np.array([[2, 5, 3, 3], [4, 3, 5, 7]], dtype=float)
    if task == "IS":
        expression = expression[:, :2]
    np.testing.assert_array_equal(log_score(expression, task), expected)


def test_only_revealed_expression_is_given_to_predictor():
    n, seed = 60, 3
    initial = np.random.default_rng(seed).choice(n, 12, replace=False)
    unseen = np.setdiff1d(np.arange(n), initial)
    codes = np.zeros((n, 8), dtype=np.uint8)
    expression = np.arange(n * 2).reshape(n, 2).astype(float)
    scores = np.zeros(n)
    scores[unseen[-1]] = 100
    calls = []

    def predictor(dna, observed, targets, remaining, *args, **kwargs):
        np.testing.assert_array_equal(observed, initial)
        np.testing.assert_array_equal(targets, expression[initial])
        np.testing.assert_array_equal(remaining, unseen)
        calls.append(targets.copy())
        return np.column_stack([np.zeros(len(remaining)), remaining])

    first = run(codes, expression, scores, "IS", seed, 2, predictor=predictor)
    expression[unseen] += 100000
    second = run(codes, expression, scores, "IS", seed, 2, predictor=predictor)
    assert first == second
    np.testing.assert_array_equal(calls[0], calls[1])
    assert first["history"][1]["indices"] == unseen[::-1][:12].tolist()


def test_inclusive_cutoff_ties_stop_without_fitting():
    def unexpected(*args, **kwargs):
        pytest.fail("An initial hit must not trigger CNN training")

    result = run(
        np.zeros((100, 8), np.uint8),
        np.zeros((100, 4)),
        np.ones(100),
        "AND",
        predictor=unexpected,
    )
    assert result["rounds_to_top"] == 1
    assert len(result["history"]) == 1


def test_pool_exhaustion_uses_small_final_batch_without_repeated_ids():
    n, seed = 37, 4
    initial = np.random.default_rng(seed).choice(n, 12, replace=False)
    target = int(np.setdiff1d(np.arange(n), initial)[-1])
    scores = np.zeros(n)
    scores[target] = 1

    def predictor(codes, observed, targets, remaining, *args, **kwargs):
        values = -remaining.astype(float)
        values[remaining == target] = -10000
        return np.column_stack([np.zeros(len(remaining)), values])

    result = run(
        np.zeros((n, 8), np.uint8),
        np.zeros((n, 2)),
        scores,
        "IS",
        seed,
        10,
        predictor=predictor,
    )
    batches = [row["indices"] for row in result["history"]]
    assert list(map(len, batches)) == [12, 12, 12, 1]
    assert len(set(sum(batches, []))) == n
    assert result["rounds_to_top"] == 4


@pytest.mark.parametrize("task", ["IS", "OR", "AND"])
def test_runner_loads_fasta_in_pool_order_and_reports_initial_hit(
    tmp_path, monkeypatch, task
):
    n = 12
    columns = ["Basal Exp (GFP, au)"] + (
        ["Induced Exp (GFP, au)"]
        if task == "IS"
        else [
            "4-OHT Induced Exp (GFP, au)",
            "GZV Induced Exp (GFP, au)",
            "Dual Induced Exp (GFP, au)",
        ]
    )
    frame = pd.DataFrame({column: np.full(n, 2**i) for i, column in enumerate(columns)})
    pool = CandidatePool(
        task, "toy", np.arange(n)[::-1], np.ones(n), frame, np.zeros((n, 1), int), {}
    )
    folder = tmp_path / "sequences"
    folder.mkdir()
    path = folder / f"{'IS' if task == 'IS' else 'logic'}_candidates.fasta.gz"
    with gzip.open(path, "wt") as stream:
        for ident in range(n):
            stream.write(f">{ident}\n{'ACGTACG' if ident == 11 else 'ACGTAC'}\n")
    codes, expression = load_inputs(pool, tmp_path)
    np.testing.assert_array_equal(codes[0], [0, 1, 2, 3, 0, 1, 2])
    np.testing.assert_array_equal(codes[1], [0, 1, 2, 3, 0, 1, 4])
    np.testing.assert_allclose(expression, np.tile(np.arange(len(columns)), (n, 1)))
    monkeypatch.setattr(shared, "load_pool", lambda *args: pool)
    rows, summary = shared.run("yin", task, "toy", tmp_path, output=tmp_path / "out")
    assert len(rows) == 12
    assert summary.iloc[0].first_inclusive_top1pct_round == 1
    assert summary.iloc[0].batches_evaluated == 1
    assert {p.name for p in (tmp_path / "out").iterdir()} == {
        f"yin_{task}_toy_seed0_selections.csv",
        f"yin_{task}_toy_seed0_summary.csv",
    }


def test_bootstrap_split_and_reverse_complement_batches():
    pytest.importorskip("tensorflow")
    from benchmarks.comparison_methods.yin.model import DNABatches, bootstrap_split

    train, valid = bootstrap_split(12, 17)
    assert len(train) == 12 and len(valid) > 0
    assert not set(train).intersection(valid)
    assert set(train).union(valid) == set(range(12))
    codes = np.array([[0, 1, 2, 3, 4], [1, 1, 0, 2, 4]], np.uint8)
    targets = np.array([[2, 3], [4, 5]], float)
    batches = DNABatches(codes, np.arange(2), targets, augment=True, batch_size=4)
    dna, labels = batches[0]
    np.testing.assert_array_equal(dna[2:], dna[:2, ::-1, ::-1])
    np.testing.assert_array_equal(labels, np.vstack([targets, targets]))


def test_real_cnn_training_and_greedy_selection():
    pytest.importorskip("tensorflow")
    rng = np.random.default_rng(20)
    codes = rng.integers(0, 4, size=(25, 32), dtype=np.uint8)
    expression = rng.normal(size=(25, 2))
    initial = np.random.default_rng(0).choice(25, 12, replace=False)
    scores = np.zeros(25)
    scores[np.setdiff1d(np.arange(25), initial)[-1]] = 1
    result = run(
        codes,
        expression,
        scores,
        "IS",
        total_rounds=2,
        settings={
            "members": 2,
            "epochs": 1,
            "batch_size": 8,
            "prediction_batch_size": 8,
        },
    )
    assert len(result["history"]) == 2
    assert len(set(sum((entry["indices"] for entry in result["history"]), []))) == 24
