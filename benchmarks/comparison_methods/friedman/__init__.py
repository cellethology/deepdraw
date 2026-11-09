"""Friedman binary-SVM selection on a measured DNA sequence pool."""

from pathlib import Path

import numpy as np
import pandas as pd

from ..data import read_fasta
from .kernel import SequenceKernel
from .method import (
    binary_labels,
    entropy,
    run,
    select_entropy,
    top_targets,
)


def selection_rows(pool, seed, history):
    return [
        {
            "method": "friedman",
            "task": pool.task,
            "dataset_name": pool.dataset_name,
            "seed": seed,
            "batch": entry["total_rounds"],
            "candidate_order": order,
            "circuit_id": ident,
            "objective_score": score,
        }
        for entry in history
        for order, (ident, score) in enumerate(
            zip(entry["circuit_ids"], entry["phenotypes"], strict=True)
        )
    ]


def run_pool(pool, data_dir, seed, batches, output, sequences_fasta=None):
    if len(pool.ids) < 12 or batches < 1:
        raise ValueError(
            "At least twelve candidates and a positive batch budget required"
        )
    if sequences_fasta is None:
        library = "IS" if pool.task == "IS" else "logic"
        sequences_fasta = (
            Path(data_dir).expanduser() / "sequences" / f"{library}_candidates.fasta.gz"
        )
    sequences = read_fasta(sequences_fasta, set(map(int, pool.ids)))
    labels = binary_labels(pool.scores, pool.ids)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"friedman_{pool.task}_{pool.dataset_name}_seed{seed}"

    def progress(history):
        pd.DataFrame(selection_rows(pool, seed, history)).to_csv(
            output / f"{stem}_selections.csv", index=False
        )

    kernel = SequenceKernel([sequences[int(ident)] for ident in pool.ids])
    try:
        result = run(
            kernel, pool.ids, pool.scores, labels, seed, batches, progress=progress
        )
    finally:
        kernel.close()
    count = max(1, int(len(pool.scores) * 0.01))
    ranked_ids = set(map(int, pool.ids[np.argsort(pool.scores)[-count:]]))
    first_ranked = next(
        (
            row["total_rounds"]
            for row in result["history"]
            if ranked_ids.intersection(row["circuit_ids"])
        ),
        None,
    )
    rows = pd.DataFrame(selection_rows(pool, seed, result["history"]))
    summary = pd.DataFrame(
        [
            {
                "method": "friedman",
                "task": pool.task,
                "dataset_name": pool.dataset_name,
                "seed": seed,
                "first_ranked_top1pct_round": first_ranked,
                "first_inclusive_top1pct_round": result["rounds_to_top"],
                "batches_evaluated": len(result["history"]),
                "circuits_evaluated": len(rows),
            }
        ]
    )
    summary.to_csv(output / f"{stem}_summary.csv", index=False)
    return rows, summary


__all__ = [
    "SequenceKernel",
    "binary_labels",
    "entropy",
    "read_fasta",
    "run",
    "run_pool",
    "select_entropy",
    "top_targets",
]
