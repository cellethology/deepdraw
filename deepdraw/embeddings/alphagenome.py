"""AlphaGenome embedding extraction for Deepdraw design pools."""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_INSTALL_COMMAND = (
    'python -m pip install "deepdraw-opt[alphagenome]" '
    '"https://github.com/google-deepmind/alphagenome_research/archive/'
    'dad09dd11a480fb8c16ae703a606f55a6e3e968b.zip"'
)

logger = logging.getLogger(__name__)

VALID_DNA_CHARS = set("ACGTRYSWKMBDHVN")
SEQUENCE_COLUMN_CANDIDATES = (
    "sequence",
    "Sequence",
    "seq",
    "Seq",
    "dna_sequence",
    "DNA_sequence",
)
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = "~/.cache"


@dataclass(frozen=True)
class SequenceRecord:
    """A design ID and DNA sequence to embed."""

    sample_id: str
    sequence: str


@dataclass(frozen=True)
class _AlphaGenomeRuntime:
    hk: Any
    jax: Any
    jnp: Any
    jmp: Any
    huggingface_hub: Any
    core_model: Any
    research_dna_model: Any
    metadata_lib: Any


def read_design_pool_sequences(
    pool_csv: str | Path,
    *,
    sequence_column: str | None = None,
    id_column: str | None = None,
    validate: bool = True,
) -> list[SequenceRecord]:
    """Read IDs and DNA sequences from a Deepdraw design-pool CSV."""

    pool_path = Path(pool_csv).expanduser().resolve()
    if not pool_path.exists():
        raise FileNotFoundError(f"Design pool CSV does not exist: {pool_path}")

    pool_df = pd.read_csv(pool_path)
    resolved_sequence_column = _resolve_sequence_column(pool_df, sequence_column)
    if id_column is not None and id_column not in pool_df.columns:
        raise ValueError(f"id_column '{id_column}' was not found in {pool_path}.")

    records: list[SequenceRecord] = []
    ids: list[str] = []
    for idx, row in pool_df.iterrows():
        sample_id = str(row[id_column]) if id_column is not None else str(idx)
        sequence = str(row[resolved_sequence_column])
        if validate:
            sequence = normalize_dna_sequence(sequence, sample_id=sample_id)
        else:
            sequence = _strip_sequence_whitespace(sequence).upper()
        records.append(SequenceRecord(sample_id=sample_id, sequence=sequence))
        ids.append(sample_id)

    _ensure_unique_ids(ids, label="design pool")
    logger.info(
        "Read %d sequences from %s (sequence column: %s; id: %s).",
        len(records),
        pool_path,
        resolved_sequence_column,
        id_column if id_column is not None else "row index",
    )
    return records


def read_fasta_sequences(
    fasta_path: str | Path,
    *,
    validate: bool = True,
) -> list[SequenceRecord]:
    """Read DNA sequences from a FASTA file without requiring Biopython."""

    path = Path(fasta_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"FASTA file does not exist: {path}")

    records: list[SequenceRecord] = []
    header: str | None = None
    chunks: list[str] = []

    def flush_record() -> None:
        if header is None:
            return
        sample_id = header.split()[0]
        sequence = "".join(chunks)
        if validate:
            sequence = normalize_dna_sequence(sequence, sample_id=sample_id)
        else:
            sequence = _strip_sequence_whitespace(sequence).upper()
        records.append(SequenceRecord(sample_id=sample_id, sequence=sequence))

    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush_record()
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"Empty FASTA header found in {path}.")
                chunks = []
            else:
                if header is None:
                    raise ValueError(
                        f"FASTA sequence encountered before header in {path}."
                    )
                chunks.append(line)
        flush_record()

    if not records:
        raise ValueError(f"No FASTA records found in {path}.")
    _ensure_unique_ids([record.sample_id for record in records], label="FASTA")
    logger.info("Read %d sequences from %s", len(records), path)
    return records


def normalize_dna_sequence(sequence: str, *, sample_id: str = "") -> str:
    """Normalize and validate a DNA sequence."""

    normalized = _strip_sequence_whitespace(sequence).upper()
    if not normalized:
        raise ValueError(f"Empty sequence found{_format_sample_suffix(sample_id)}.")

    invalid_chars = sorted(set(normalized) - VALID_DNA_CHARS)
    if invalid_chars:
        raise ValueError(
            f"Invalid DNA characters {invalid_chars}"
            f"{_format_sample_suffix(sample_id)}."
        )
    return normalized


def embed_design_pool(
    *,
    pool_csv: str | Path,
    output_path: str | Path,
    sequence_column: str | None = None,
    id_column: str | None = None,
    model_version: str = "all_folds",
    batch_size: int = 1,
    pooling: str = "mean",
    resolution: int = 1,
    species: str = "human",
    pad_to_multiple: bool = True,
    validate: bool = True,
    device: str | None = None,
) -> None:
    """Embed a Deepdraw design-pool CSV with AlphaGenome."""

    records = read_design_pool_sequences(
        pool_csv,
        sequence_column=sequence_column,
        id_column=id_column,
        validate=validate,
    )
    embed_sequences(
        records,
        output_path=output_path,
        model_version=model_version,
        batch_size=batch_size,
        pooling=pooling,
        resolution=resolution,
        species=species,
        pad_to_multiple=pad_to_multiple,
        device=device,
    )


def embed_fasta(
    *,
    fasta_path: str | Path,
    output_path: str | Path,
    model_version: str = "all_folds",
    batch_size: int = 1,
    pooling: str = "mean",
    resolution: int = 1,
    species: str = "human",
    pad_to_multiple: bool = True,
    validate: bool = True,
    device: str | None = None,
) -> None:
    """Embed a FASTA file with AlphaGenome."""

    records = read_fasta_sequences(fasta_path, validate=validate)
    embed_sequences(
        records,
        output_path=output_path,
        model_version=model_version,
        batch_size=batch_size,
        pooling=pooling,
        resolution=resolution,
        species=species,
        pad_to_multiple=pad_to_multiple,
        device=device,
    )


def embed_sequences(
    records: list[SequenceRecord],
    *,
    output_path: str | Path,
    model_version: str = "all_folds",
    batch_size: int = 1,
    pooling: str = "mean",
    resolution: int = 1,
    species: str = "human",
    pad_to_multiple: bool = True,
    device: str | None = None,
) -> None:
    """Run AlphaGenome and save embeddings to a Deepdraw-compatible NPZ."""

    if not records:
        raise ValueError("No sequences were provided for AlphaGenome embedding.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size}.")
    if pooling not in {"mean", "none"}:
        raise ValueError("pooling must be one of: 'mean', 'none'.")
    if resolution not in {1, 128}:
        raise ValueError("resolution must be one of: 1, 128.")

    runtime = _load_alphagenome_runtime()
    device_obj = _auto_device(runtime, device)
    if _is_cpu_device(device_obj):
        _patch_cpu_attention_precision(runtime)
    model = _create_model_from_huggingface(
        runtime,
        model_version=model_version,
        device=device_obj,
    )
    _adapt_model_precision_for_device(runtime, model, device_obj)
    forward = _build_forward(
        runtime,
        force_float32_compute=getattr(device_obj, "platform", None) == "cpu",
    )
    organism_index = _get_organism_index(species)
    multiple_of = 2048 if pad_to_multiple else 1

    logger.info(
        "Embedding %d sequences with AlphaGenome %s at %dbp resolution.",
        len(records),
        model_version,
        resolution,
    )
    all_embeddings: list[np.ndarray] = []
    embedding_lengths: list[int] = []

    total_batches = (len(records) + batch_size - 1) // batch_size
    for batch_num, start in enumerate(range(0, len(records), batch_size), start=1):
        batch_records = records[start : start + batch_size]
        logger.info(
            "AlphaGenome batch %d/%d (%d sequences).",
            batch_num,
            total_batches,
            len(batch_records),
        )
        batch_embeddings, batch_lengths = _embed_batch(
            runtime=runtime,
            model=model,
            forward=forward,
            records=batch_records,
            organism_index=organism_index,
            multiple_of=multiple_of,
            resolution=resolution,
            pooling=pooling,
            device=device_obj,
        )
        all_embeddings.extend(batch_embeddings)
        embedding_lengths.extend(batch_lengths)

    ids = np.asarray([record.sample_id for record in records], dtype=object)
    sequence_lengths_bp = np.asarray([len(record.sequence) for record in records])
    _save_embeddings(
        embeddings=all_embeddings,
        ids=ids,
        sequence_lengths_bp=sequence_lengths_bp,
        embedding_lengths=np.asarray(embedding_lengths, dtype=np.int64),
        output_path=output_path,
        pooling=pooling,
        resolution=resolution,
        model_version=model_version,
        species=species,
    )


def _embed_batch(
    *,
    runtime: _AlphaGenomeRuntime,
    model: Any,
    forward: Any,
    records: list[SequenceRecord],
    organism_index: int,
    multiple_of: int,
    resolution: int,
    pooling: str,
    device: Any | None,
) -> tuple[list[np.ndarray], list[int]]:
    onehot_arrays = [
        np.asarray(model._one_hot_encoder.encode(record.sequence), dtype=np.float32)
        for record in records
    ]
    padded = _pad_onehot_to_multiple(onehot_arrays, multiple_of)
    with _default_device(runtime, device):
        onehot = runtime.jnp.array(padded, dtype=_model_input_dtype(runtime, model))
        org_idx = runtime.jnp.array(
            [organism_index] * len(records),
            dtype=runtime.jnp.int32,
        )

        embeds, _ = forward.apply(
            model._params,
            model._state,
            None,
            onehot,
            org_idx,
            model._metadata,
        )

    if resolution == 1:
        embeddings = embeds.embeddings_1bp
        mask_factor = 1
    else:
        embeddings = embeds.embeddings_128bp
        mask_factor = 128

    lengths_bp = np.asarray(
        [len(record.sequence) for record in records], dtype=np.int64
    )
    max_len = int(embeddings.shape[1]) * mask_factor
    base_mask = np.zeros((len(records), max_len, 1), dtype=np.float32)
    for idx, seq_len in enumerate(lengths_bp):
        base_mask[idx, : int(seq_len), 0] = 1.0
    with _default_device(runtime, device):
        pool_mask = runtime.jnp.array(base_mask)
        if mask_factor != 1:
            pool_mask = _downsample_padding_mask(runtime, pool_mask, mask_factor)

        if pooling == "mean":
            pooled = _mean_pool(runtime, embeddings, pool_mask)
            return [np.asarray(row, dtype=np.float32) for row in np.asarray(pooled)], [
                1
            ] * len(records)

        pool_lengths = np.asarray(pool_mask.sum(axis=1)).reshape(len(records))

    output_embeddings: list[np.ndarray] = []
    output_lengths: list[int] = []
    for idx in range(len(records)):
        embedding_len = int(pool_lengths[idx])
        output_embeddings.append(np.asarray(embeddings[idx, :embedding_len, :]))
        output_lengths.append(embedding_len)
    return output_embeddings, output_lengths


def _load_alphagenome_runtime() -> _AlphaGenomeRuntime:
    try:
        import haiku as hk
        import huggingface_hub
        import jax
        import jax.numpy as jnp
        import jmp
        from alphagenome_research.model import dna_model as research_dna_model
        from alphagenome_research.model import model as core_model
        from alphagenome_research.model.metadata import metadata as metadata_lib
    except ImportError as exc:
        raise RuntimeError(
            "AlphaGenome embedding extraction needs its optional dependencies. "
            f"Install them in a Python 3.11 environment with `{_INSTALL_COMMAND}`, "
            "then run `deepdraw embed-alphagenome ...`."
        ) from exc

    return _AlphaGenomeRuntime(
        hk=hk,
        jax=jax,
        jnp=jnp,
        jmp=jmp,
        huggingface_hub=huggingface_hub,
        core_model=core_model,
        research_dna_model=research_dna_model,
        metadata_lib=metadata_lib,
    )


def _create_model_from_huggingface(
    runtime: _AlphaGenomeRuntime,
    *,
    model_version: str,
    device: Any | None,
) -> Any:
    repo_id = f"google/alphagenome-{model_version.replace('_', '-').lower()}"
    offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}
    try:
        checkpoint_path = runtime.huggingface_hub.snapshot_download(
            repo_id=repo_id,
            local_files_only=offline,
            cache_dir=_get_hf_cache_dir(),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not download or load {repo_id}. AlphaGenome model weights are "
            "gated on Hugging Face; accept access to the model and authenticate "
            "with `hf auth login` or "
            "set HF_TOKEN. If you are running offline, make sure the model is "
            "already present in the Hugging Face cache."
        ) from exc
    organism_settings = {
        runtime.research_dna_model.Organism.HOMO_SAPIENS: (
            runtime.research_dna_model.OrganismSettings(
                metadata=runtime.metadata_lib.load(
                    runtime.research_dna_model.Organism.HOMO_SAPIENS
                )
            )
        ),
        runtime.research_dna_model.Organism.MUS_MUSCULUS: (
            runtime.research_dna_model.OrganismSettings(
                metadata=runtime.metadata_lib.load(
                    runtime.research_dna_model.Organism.MUS_MUSCULUS
                )
            )
        ),
    }
    with _default_device(runtime, device):
        return runtime.research_dna_model.create(
            checkpoint_path,
            organism_settings=organism_settings,
            device=device,
        )


def _get_hf_cache_dir() -> str:
    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_hub_cache:
        return os.path.expanduser(hf_hub_cache)

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(os.path.expanduser(hf_home), "hub")

    base = os.environ.get(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR)
    return os.path.expanduser(os.path.join(base, "alphagenome"))


def _build_forward(
    runtime: _AlphaGenomeRuntime,
    *,
    force_float32_compute: bool = False,
) -> Any:
    policy_name = (
        "params=float32,compute=float32,output=float32"
        if force_float32_compute
        else "params=float32,compute=bfloat16,output=bfloat16"
    )
    policy = runtime.jmp.get_policy(policy_name)

    def forward(dna_onehot: Any, organism_index: Any, output_metadata: Any) -> Any:
        with runtime.hk.mixed_precision.push_policy(
            runtime.core_model.AlphaGenome,
            policy,
        ):
            model = runtime.core_model.AlphaGenome(output_metadata=output_metadata)
            _, embeds = model(dna_onehot, organism_index)
        return embeds

    return runtime.hk.transform_with_state(forward)


def _pad_onehot_to_multiple(
    onehot_arrays: list[np.ndarray],
    multiple_of: int,
) -> np.ndarray:
    max_len = max(arr.shape[0] for arr in onehot_arrays)
    if multiple_of > 1:
        remainder = max_len % multiple_of
        if remainder:
            max_len += multiple_of - remainder

    padded = np.zeros((len(onehot_arrays), max_len, 4), dtype=np.float32)
    for idx, arr in enumerate(onehot_arrays):
        padded[idx, : arr.shape[0], :] = arr
    return padded


def _mean_pool(runtime: _AlphaGenomeRuntime, embeddings: Any, padding_mask: Any) -> Any:
    if hasattr(padding_mask, "astype"):
        padding_mask = padding_mask.astype(embeddings.dtype)
    summed = (embeddings * padding_mask).sum(axis=1)
    denominator = padding_mask.sum(axis=1).clip(min=1.0)
    return summed / denominator


def _adapt_model_precision_for_device(
    runtime: _AlphaGenomeRuntime,
    model: Any,
    device: Any | None,
) -> None:
    if not _is_cpu_device(device):
        return

    def cast_cpu_float(value: Any) -> Any:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and dtype == getattr(runtime.jnp, "bfloat16", None):
            return value.astype(runtime.jnp.float32)
        return value

    model._params = runtime.jax.tree.map(cast_cpu_float, model._params)
    model._state = runtime.jax.tree.map(cast_cpu_float, model._state)


def _patch_cpu_attention_precision(runtime: _AlphaGenomeRuntime) -> None:
    """Relax AlphaGenome's BF16 attention preset on CPU-only JAX backends."""

    if getattr(runtime.jnp.einsum, "_deepdraw_cpu_precision_patch", False):
        return

    original_einsum = runtime.jnp.einsum
    bf16_preset = runtime.jax.lax.DotAlgorithmPreset.BF16_BF16_F32

    def einsum_without_cpu_bf16_preset(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("precision") == bf16_preset:
            kwargs = dict(kwargs)
            kwargs["precision"] = None
        return original_einsum(*args, **kwargs)

    einsum_without_cpu_bf16_preset._deepdraw_cpu_precision_patch = True
    runtime.jnp.einsum = einsum_without_cpu_bf16_preset


def _model_input_dtype(runtime: _AlphaGenomeRuntime, model: Any) -> Any:
    dtype = _first_floating_dtype(runtime, model._params)
    if dtype is not None:
        return dtype
    return getattr(runtime.jnp, "bfloat16", runtime.jnp.float32)


def _first_floating_dtype(runtime: _AlphaGenomeRuntime, tree: Any) -> Any | None:
    if not hasattr(runtime, "jax"):
        return None
    leaves = runtime.jax.tree.leaves(tree)
    for leaf in leaves:
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and runtime.jnp.issubdtype(dtype, runtime.jnp.floating):
            return dtype
    return None


def _downsample_padding_mask(
    runtime: _AlphaGenomeRuntime,
    padding_mask: Any,
    factor: int,
) -> Any:
    if factor <= 1:
        return padding_mask
    batch_size, seq_len, _ = padding_mask.shape
    if seq_len % factor != 0:
        raise ValueError(
            f"Padding length {seq_len} must be divisible by downsample factor {factor}."
        )
    reduced = padding_mask.reshape(batch_size, seq_len // factor, factor, 1)
    return reduced.max(axis=2)


def _auto_device(runtime: _AlphaGenomeRuntime, requested: str | None) -> Any:
    if requested is not None:
        return _get_device(runtime, requested)

    devices = runtime.jax.devices()
    for preferred in ("gpu", "tpu"):
        for device in devices:
            if device.platform.lower() == preferred:
                return device
    logger.warning("No GPU/TPU found; falling back to CPU.")
    return _get_device(runtime, "cpu")


def _get_device(runtime: _AlphaGenomeRuntime, requested: str) -> Any:
    requested = requested.lower()
    try:
        backend_devices = runtime.jax.devices(requested)
    except Exception:
        backend_devices = []
    if backend_devices:
        return backend_devices[0]

    for device in runtime.jax.devices():
        platform = device.platform.lower()
        if platform == requested:
            return device
    if requested == "gpu" and any(
        device.platform.lower() == "metal" for device in runtime.jax.devices()
    ):
        raise ValueError(
            "A JAX Metal device is visible, but Deepdraw's AlphaGenome embedding "
            "path is not supported on Apple Metal. Use --device cpu on Apple "
            "Silicon, or run on a CUDA GPU."
        )
    raise ValueError(
        f"Requested device '{requested}' not available: {runtime.jax.devices()}"
    )


def _is_cpu_device(device: Any | None) -> bool:
    return getattr(device, "platform", None) == "cpu"


def _default_device(runtime: _AlphaGenomeRuntime, device: Any | None) -> Any:
    if device is None:
        return nullcontext()
    return runtime.jax.default_device(device)


def _get_organism_index(species: str) -> int:
    mapping = {
        "human": 0,
        "mouse": 1,
    }
    key = species.lower()
    if key not in mapping:
        raise ValueError("species must be one of: human, mouse.")
    return mapping[key]


def _save_embeddings(
    *,
    embeddings: list[np.ndarray],
    ids: np.ndarray,
    sequence_lengths_bp: np.ndarray,
    embedding_lengths: np.ndarray,
    output_path: str | Path,
    pooling: str,
    resolution: int,
    model_version: str,
    species: str,
) -> None:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if pooling == "mean":
        embeddings_array = np.stack(embeddings).astype(np.float32)
    else:
        embeddings_array = _pad_embeddings(embeddings)

    np.savez_compressed(
        output,
        embeddings=embeddings_array,
        ids=ids,
        sequence_lengths_bp=sequence_lengths_bp,
        embedding_lengths=embedding_lengths,
        pooling=np.asarray(pooling),
        resolution=np.asarray(resolution, dtype=np.int32),
        model=np.asarray(f"alphagenome-{model_version}"),
        species=np.asarray(species),
    )
    logger.info(
        "Saved AlphaGenome embeddings with shape %s to %s",
        embeddings_array.shape,
        output,
    )


def _pad_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    max_len = max(arr.shape[0] for arr in embeddings)
    dim = embeddings[0].shape[1]
    padded = np.zeros((len(embeddings), max_len, dim), dtype=np.float32)
    for idx, arr in enumerate(embeddings):
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D unpooled embedding; got shape {arr.shape}.")
        if arr.shape[1] != dim:
            raise ValueError("All embeddings must have the same feature dimension.")
        padded[idx, : arr.shape[0], :] = arr
    return padded


def _resolve_sequence_column(
    pool_df: pd.DataFrame,
    requested: str | None,
) -> str:
    if requested:
        if requested not in pool_df.columns:
            raise ValueError(f"sequence_column '{requested}' was not found.")
        return requested
    for candidate in SEQUENCE_COLUMN_CANDIDATES:
        if candidate in pool_df.columns:
            return candidate
    raise ValueError(
        "Could not infer the sequence column. Pass --sequence-column explicitly."
    )


def _ensure_unique_ids(ids: list[str], *, label: str) -> None:
    unique_count = len(set(ids))
    if unique_count != len(ids):
        raise ValueError(f"{label} ids contain {len(ids) - unique_count} duplicates.")


def _strip_sequence_whitespace(sequence: str) -> str:
    return "".join(str(sequence).split())


def _format_sample_suffix(sample_id: str) -> str:
    return f" in sequence '{sample_id}'" if sample_id else ""
