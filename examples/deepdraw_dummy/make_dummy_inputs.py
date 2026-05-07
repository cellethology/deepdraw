"""Regenerate the tiny Deepdraw example inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
POOL_PATH = ROOT / "design_pool.csv"
EMBEDDINGS_PATH = ROOT / "embeddings.npz"
MEASUREMENTS_PATH = ROOT / "measurements.csv"
NUM_DESIGNS = 60
ROUND0_POOL_INDICES = [52, 3, 10, 17, 24, 31, 38, 45, 56, 20, 30, 42]


def main() -> None:
    pool = pd.DataFrame(
        {
            "variant_id": [f"variant_{idx:02d}" for idx in range(NUM_DESIGNS)],
            "sequence": [_sequence_for(idx) for idx in range(NUM_DESIGNS)],
            "design_note": [f"starter design {idx}" for idx in range(NUM_DESIGNS)],
        }
    )
    pool.to_csv(POOL_PATH, index=False)

    positions = np.linspace(0.0, 1.0, len(pool), dtype=np.float32)
    gc_fraction = pool["sequence"].map(_gc_fraction).to_numpy(dtype=np.float32)
    embeddings = np.column_stack(
        [
            positions,
            gc_fraction,
            np.sin(positions * np.pi).astype(np.float32),
            np.cos(positions * np.pi).astype(np.float32),
        ]
    ).astype(np.float32)
    np.savez_compressed(
        EMBEDDINGS_PATH,
        embeddings=embeddings,
        ids=pool["variant_id"].to_numpy(),
    )

    measurements = pd.DataFrame(
        {
            "deepdraw_pool_index": ROUND0_POOL_INDICES,
            "deepdraw_id": [f"variant_{idx:02d}" for idx in ROUND0_POOL_INDICES],
            "Expression": [_dummy_expression(idx) for idx in ROUND0_POOL_INDICES],
        }
    )
    measurements.to_csv(MEASUREMENTS_PATH, index=False)


def _gc_fraction(sequence: str) -> float:
    gc_count = sequence.count("G") + sequence.count("C")
    return gc_count / len(sequence)


def _sequence_for(idx: int) -> str:
    alphabet = "ACGT"
    value = idx
    suffix = []
    for _ in range(8):
        suffix.append(alphabet[value % len(alphabet)])
        value //= len(alphabet)
    return "ATGCGTAC" + "".join(suffix)


def _dummy_expression(idx: int) -> float:
    return round(1.0 + idx * 0.31, 2)


if __name__ == "__main__":
    main()
