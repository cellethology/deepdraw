"""Run a comparison method on one deposited design space."""

import argparse
import hashlib
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_pool


def fitting_seeds(task, space, seed, batch):
    identity = hashlib.sha256(f"{task}|{space}".encode()).digest()[:16]
    words = np.frombuffer(identity, dtype="<u4").tolist()
    return np.random.SeedSequence([seed, batch, *words]).generate_state(2)


def run(
    method,
    task,
    space,
    data_dir,
    seed=0,
    batches=None,
    output="benchmark_output",
    settings=None,
    sequences_fasta=None,
):
    if method not in ("merzbacher", "kobiela", "friedman", "yin"):
        raise ValueError("Unknown comparison method")
    pool = load_pool(task, space, data_dir)
    if method == "yin":
        from .yin import run_pool

        return run_pool(
            pool,
            data_dir,
            seed,
            200 if batches is None else batches,
            output,
            sequences_fasta=sequences_fasta,
            settings=settings,
        )
    if method == "friedman":
        from .friedman import run_pool

        return run_pool(
            pool,
            data_dir,
            seed,
            200 if batches is None else batches,
            output,
            sequences_fasta=sequences_fasta,
        )
    module = method
    if method == "kobiela":
        module += ".single_input" if task == "IS" else ".logic"
    implementation = importlib.import_module("." + module, __package__)
    batches = (100 if method == "kobiela" else 200) if batches is None else batches
    if batches < 1 or len(pool.ids) < 12:
        raise ValueError(
            "Positive batch budget and at least twelve candidates required"
        )
    initial_seed = (
        np.random.SeedSequence(seed).spawn(4)[0] if method == "kobiela" else seed
    )
    initial = np.random.default_rng(initial_seed).choice(
        len(pool.ids), 12, replace=False
    )
    tie_rng = np.random.default_rng(np.random.SeedSequence([seed, 1729]))
    count = max(1, int(len(pool.scores) * 0.01))
    ranked = np.zeros(len(pool.ids), dtype=bool)
    ranked[np.argsort(pool.scores)[-count:]] = True
    inclusive = pool.scores >= np.sort(pool.scores)[-count]
    observed, rows = [], []
    available_mask = np.ones(len(pool.ids), dtype=bool)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{method}_{task}_{space}_seed{seed}"
    first_ranked = first_inclusive = None
    for batch in range(batches):
        available = np.flatnonzero(available_mask)
        if not len(available):
            break
        if batch == 0:
            selected = initial
        else:
            arguments = (pool, np.asarray(observed), available, seed, batch, tie_rng)
            if method == "kobiela":
                streams = fitting_seeds(task, space, seed, batch)
                options = dict(settings or {})
                options.setdefault("inference_seed", int(streams[0]))
                options.setdefault("selection_seed", int(streams[1]))
                selected = implementation.select_batch(*arguments, settings=options)
            else:
                selected = implementation.select_batch(*arguments)
        selected = np.asarray(selected, dtype=np.int64)
        if (
            len(selected) != min(12, len(available))
            or len(np.unique(selected)) != len(selected)
            or not np.isin(selected, available).all()
        ):
            raise ValueError("Selected batch must contain distinct available circuits")
        available_mask[selected] = False
        observed.extend(selected.tolist())
        if first_ranked is None and ranked[selected].any():
            first_ranked = batch + 1
        if first_inclusive is None and inclusive[selected].any():
            first_inclusive = batch + 1
        rows.extend(
            {
                "method": method,
                "task": task,
                "dataset_name": space,
                "seed": seed,
                "batch": batch + 1,
                "candidate_order": order,
                "circuit_id": int(pool.ids[index]),
                "objective_score": float(pool.scores[index]),
            }
            for order, index in enumerate(selected)
        )
        pd.DataFrame(rows).to_csv(output / f"{stem}_selections.csv", index=False)
        if first_ranked is not None:
            break
    summary = pd.DataFrame(
        [
            {
                "method": method,
                "task": task,
                "dataset_name": space,
                "seed": seed,
                "first_ranked_top1pct_round": first_ranked,
                "first_inclusive_top1pct_round": first_inclusive,
                "batches_evaluated": rows[-1]["batch"],
                "circuits_evaluated": len(rows),
            }
        ]
    )
    summary.to_csv(output / f"{stem}_summary.csv", index=False)
    return pd.DataFrame(rows), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("merzbacher", "kobiela", "friedman", "yin"))
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Downloaded Zenodo data directory"
    )
    parser.add_argument("--task", choices=("IS", "OR", "AND"), required=True)
    parser.add_argument("--space", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batches", type=int)
    parser.add_argument("--output", type=Path, default=Path("benchmark_output"))
    parser.add_argument(
        "--sequences-fasta",
        type=Path,
        help="Friedman/Yin: override the deposited complete-cassette FASTA (.gz supported)",
    )
    args = parser.parse_args()
    _, summary = run(**vars(args))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
