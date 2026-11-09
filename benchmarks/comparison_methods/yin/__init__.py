"""Yin-inspired bootstrap CNN selection on a measured DNA pool."""

from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_inputs
from .method import log_score, run


def run_pool(
    pool, data_dir, seed, batches, output, sequences_fasta=None, settings=None
):
    if len(pool.ids) < 12 or batches < 1:
        raise ValueError(
            "At least twelve candidates and a positive batch budget required"
        )
    codes, expression = load_inputs(pool, data_dir, sequences_fasta)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"yin_{pool.task}_{pool.dataset_name}_seed{seed}"

    def selection_rows(history):
        return pd.DataFrame(
            [
                {
                    "method": "yin",
                    "task": pool.task,
                    "dataset_name": pool.dataset_name,
                    "seed": seed,
                    "batch": entry["round"],
                    "candidate_order": order,
                    "circuit_id": int(pool.ids[index]),
                    "objective_score": float(pool.scores[index]),
                }
                for entry in history
                for order, index in enumerate(entry["indices"])
            ]
        )

    def progress(history):
        selection_rows(history).to_csv(output / f"{stem}_selections.csv", index=False)

    result = run(
        codes,
        expression,
        pool.scores,
        pool.task,
        seed,
        batches,
        settings=settings,
        progress=progress,
    )
    count = max(1, int(len(pool.ids) * 0.01))
    ranked = set(np.argsort(pool.scores)[-count:].tolist())
    first_ranked = next(
        (
            entry["round"]
            for entry in result["history"]
            if ranked.intersection(entry["indices"])
        ),
        None,
    )
    rows = selection_rows(result["history"])
    summary = pd.DataFrame(
        [
            {
                "method": "yin",
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


__all__ = ["load_inputs", "log_score", "run", "run_pool"]
