#!/usr/bin/env python3
"""
Concatenate two embedding NPZ files by matching sample ids.

By default, embeddings are concatenated as-is. Optional L2 normalization can be
enabled before concatenation, and optional PCA can be applied after
concatenation to reach a target explained variance ratio.

Usage examples:
  python utils/concat_embedding.py a.npz b.npz out.npz
  python utils/concat_embedding.py a.npz b.npz out.npz --normalize
  python utils/concat_embedding.py a.npz b.npz out.npz --normalize --pca-var 0.95
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "embeddings" not in data or "ids" not in data:
        raise ValueError(
            f"{path} must contain 'embeddings' and 'ids' arrays. "
            f"Found keys: {list(data.keys())}"
        )
    embeddings = np.asarray(data["embeddings"])
    ids = np.asarray(data["ids"])
    if ids.ndim != 1:
        raise ValueError(f"{path} ids must be 1D, got shape {ids.shape}")
    if embeddings.shape[0] != ids.shape[0]:
        raise ValueError(
            f"{path} embeddings/ids length mismatch: "
            f"{embeddings.shape[0]} vs {ids.shape[0]}"
        )
    return embeddings, ids


def _ensure_unique(ids: np.ndarray, label: str) -> None:
    unique_count = np.unique(ids).size
    if unique_count != ids.size:
        raise ValueError(f"{label} ids contain duplicates ({ids.size - unique_count}).")


def _l2_normalize(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, eps)


def _apply_pca_variance(
    embeddings: np.ndarray, target_variance: float
) -> tuple[np.ndarray, int]:
    if not 0.0 < target_variance <= 1.0:
        raise ValueError("target_variance must be in (0, 1].")
    mean = np.mean(embeddings, axis=0, keepdims=True)
    centered = embeddings - mean
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    if s.size == 0:
        return centered, 0
    var = (s**2) / max(embeddings.shape[0] - 1, 1)
    total_var = float(np.sum(var))
    if total_var <= 0:
        return centered, 1
    explained_ratio = var / total_var
    cumulative = np.cumsum(explained_ratio)
    n_components = int(np.searchsorted(cumulative, target_variance) + 1)
    components = vt[:n_components]
    return centered @ components.T, n_components


def concat_embeddings(
    path_a: Path,
    path_b: Path,
    output_path: Path,
    normalize: bool = False,
    pca_variance: float | None = None,
) -> None:
    emb_a, ids_a = _load_npz(path_a)
    emb_b, ids_b = _load_npz(path_b)
    _ensure_unique(ids_a, f"{path_a}")
    _ensure_unique(ids_b, f"{path_b}")

    ids_b_set = set(ids_b.tolist())
    mask_a = np.isin(ids_a, ids_b)
    ids_common = ids_a[mask_a]
    if ids_common.size == 0:
        raise ValueError("No overlapping ids found between the two files.")

    index_b = {int(id_val): idx for idx, id_val in enumerate(ids_b)}
    idx_a = np.nonzero(mask_a)[0]
    idx_b = np.array([index_b[int(id_val)] for id_val in ids_common], dtype=int)

    emb_a_sel = emb_a[idx_a]
    emb_b_sel = emb_b[idx_b]
    if normalize:
        emb_a_sel = _l2_normalize(emb_a_sel)
        emb_b_sel = _l2_normalize(emb_b_sel)
    emb_concat = np.concatenate([emb_a_sel, emb_b_sel], axis=1)
    if pca_variance is not None:
        emb_concat, n_components = _apply_pca_variance(emb_concat, pca_variance)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, embeddings=emb_concat, ids=ids_common)

    print(f"Saved concatenated embeddings to {output_path}")
    print(f"File A: {path_a} ({ids_a.size} ids, {emb_a.shape[1]} dims)")
    print(f"File B: {path_b} ({ids_b.size} ids, {emb_b.shape[1]} dims)")
    print(f"Overlap: {ids_common.size} ids, output shape: {emb_concat.shape}")
    if pca_variance is not None:
        print(f"PCA retained {n_components} components for {pca_variance:.2f} variance")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate two embedding NPZ files by matching ids."
    )
    parser.add_argument("embedding_a", type=Path, help="Path to first NPZ file.")
    parser.add_argument("embedding_b", type=Path, help="Path to second NPZ file.")
    parser.add_argument("output", type=Path, help="Path for output NPZ file.")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Enable L2 normalization before concatenation.",
    )
    parser.add_argument(
        "--pca-var",
        type=float,
        default=None,
        help="Target explained variance ratio for PCA (e.g., 0.95).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    concat_embeddings(
        args.embedding_a,
        args.embedding_b,
        args.output,
        normalize=args.normalize,
        pca_variance=args.pca_var,
    )


if __name__ == "__main__":
    main()
