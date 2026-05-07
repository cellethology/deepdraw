"""Regenerate the tiny Deepdraw example inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
POOL_PATH = ROOT / "design_pool.csv"
EMBEDDINGS_PATH = ROOT / "embeddings.npz"
MEASUREMENTS_PATH = ROOT / "measurements_round0.csv"
NUM_DESIGNS = 24


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
            "deepdraw_pool_index": [7, 14, 1, 18, 21, 10, 2, 5, 9, 13, 23, 0],
            "deepdraw_id": [
                "variant_07",
                "variant_14",
                "variant_01",
                "variant_18",
                "variant_21",
                "variant_10",
                "variant_02",
                "variant_05",
                "variant_09",
                "variant_13",
                "variant_23",
                "variant_00",
            ],
            "Expression": [
                3.5,
                5.8,
                1.3,
                6.6,
                7.2,
                4.4,
                1.7,
                2.8,
                4.1,
                5.5,
                7.8,
                1.0,
            ],
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


if __name__ == "__main__":
    main()
