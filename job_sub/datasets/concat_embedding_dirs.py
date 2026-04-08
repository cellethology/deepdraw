#!/usr/bin/env python3
"""
Concatenate two embedding NPZ files inside each child directory of a root path.

Each child directory is expected to contain:
  - 166k_enformer_center_default.npz
  - evo2_meanpool_block28_updated_default.npz

The script verifies that both files contain identical `ids` arrays in identical
order, concatenates the embeddings row-wise, and writes a new NPZ file back into
the same child directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

FIRST_FILENAME = "166k_enformer_center_default.npz"
SECOND_FILENAME = "evo2_meanpool_block28_updated_default.npz"
DEFAULT_OUTPUT_FILENAME = "concat_166k_enformer_evo2.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read each child directory of a root folder, load the two expected "
            "embedding NPZ files, verify matching ids, and write a concatenated NPZ."
        )
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Directory whose immediate subdirectories will be processed.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_FILENAME,
        help=(
            "Filename for the concatenated NPZ written inside each child directory. "
            f"Defaults to {DEFAULT_OUTPUT_FILENAME}."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def load_embedding_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "embeddings" not in data or "ids" not in data:
        raise ValueError(
            f"{path} must contain 'embeddings' and 'ids'. Found keys: {list(data.keys())}"
        )

    embeddings = np.asarray(data["embeddings"])
    ids = np.asarray(data["ids"])

    if embeddings.ndim != 2:
        raise ValueError(f"{path} embeddings must be 2D, got shape {embeddings.shape}")
    if ids.ndim != 1:
        raise ValueError(f"{path} ids must be 1D, got shape {ids.shape}")
    if embeddings.shape[0] != ids.shape[0]:
        raise ValueError(
            f"{path} has {embeddings.shape[0]} embeddings but {ids.shape[0]} ids"
        )

    return embeddings, ids


def concatenate_directory(child_dir: Path, output_name: str, overwrite: bool) -> None:
    first_path = child_dir / FIRST_FILENAME
    second_path = child_dir / SECOND_FILENAME
    output_path = child_dir / output_name

    if not first_path.exists():
        raise FileNotFoundError(f"Missing {first_path}")
    if not second_path.exists():
        raise FileNotFoundError(f"Missing {second_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    first_embeddings, first_ids = load_embedding_npz(first_path)
    second_embeddings, second_ids = load_embedding_npz(second_path)

    if first_ids.shape != second_ids.shape:
        raise ValueError(
            f"ID shape mismatch in {child_dir}: {first_ids.shape} vs {second_ids.shape}"
        )
    if not np.array_equal(first_ids, second_ids):
        raise ValueError(
            f"ID mismatch in {child_dir}: the two files do not have identical ids "
            "in identical order."
        )

    merged_embeddings = np.concatenate([first_embeddings, second_embeddings], axis=1)
    np.savez(output_path, embeddings=merged_embeddings, ids=first_ids)

    print(
        f"{child_dir.name}: wrote {output_path.name} with shape {merged_embeddings.shape}"
    )


def main() -> None:
    args = parse_args()
    root_dir = args.root_dir.expanduser().resolve()

    if not root_dir.is_dir():
        raise NotADirectoryError(f"{root_dir} is not a directory")

    child_dirs = sorted(path for path in root_dir.iterdir() if path.is_dir())
    if not child_dirs:
        raise RuntimeError(f"No child directories found in {root_dir}")

    for child_dir in child_dirs:
        concatenate_directory(
            child_dir=child_dir,
            output_name=args.output_name,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
