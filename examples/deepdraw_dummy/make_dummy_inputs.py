"""Regenerate the tiny Deepdraw example inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
POOL_PATH = ROOT / "design_pool.csv"
EMBEDDINGS_PATH = ROOT / "embeddings.npz"
MEASUREMENTS_PATH = ROOT / "measurements_round0.csv"


def main() -> None:
    pool = pd.DataFrame(
        {
            "variant_id": [f"variant_{idx:02d}" for idx in range(12)],
            "sequence": [
                "ATGCGTACGTTAGCGA",
                "ATGCGTACGATAGCAA",
                "ATGCGTACGCTAGCTA",
                "ATGCGTACGGTAGCGT",
                "ATGCGTACGTTAGTTA",
                "ATGCGTACGAAAGCGA",
                "ATGCGTACGCCAGCAA",
                "ATGCGTACGGGAGCTA",
                "ATGCGTACGTAAGCGT",
                "ATGCGTACGCAAGTTA",
                "ATGCGTACGGAAGCGA",
                "ATGCGTACGTCAGCAA",
            ],
            "design_note": [f"starter design {idx}" for idx in range(12)],
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
            "deepdraw_pool_index": [1, 5, 9, 8],
            "deepdraw_id": ["variant_01", "variant_05", "variant_09", "variant_08"],
            "Expression": [1.2, 2.8, 5.4, 4.7],
        }
    )
    measurements.to_csv(MEASUREMENTS_PATH, index=False)


def _gc_fraction(sequence: str) -> float:
    gc_count = sequence.count("G") + sequence.count("C")
    return gc_count / len(sequence)


if __name__ == "__main__":
    main()
