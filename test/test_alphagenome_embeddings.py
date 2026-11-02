from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import deepdraw.embeddings.alphagenome as alphagenome
from deepdraw import cli
from deepdraw.embeddings.alphagenome import (
    embed_design_pool,
    embed_fasta,
    embed_sequences,
    normalize_dna_sequence,
    read_design_pool_sequences,
    read_fasta_sequences,
)


def test_read_design_pool_sequences_uses_stable_ids(tmp_path):
    pool_path = tmp_path / "designs.csv"
    pd.DataFrame(
        {
            "variant_id": ["v1", "v2"],
            "sequence": ["atgc", "NNNN"],
        }
    ).to_csv(pool_path, index=False)

    records = read_design_pool_sequences(
        pool_path,
        sequence_column="sequence",
        id_column="variant_id",
    )

    assert [record.sample_id for record in records] == ["v1", "v2"]
    assert [record.sequence for record in records] == ["ATGC", "NNNN"]


def test_read_fasta_sequences_parses_multiline_records(tmp_path):
    fasta_path = tmp_path / "designs.fasta"
    fasta_path.write_text(">v1 first design\nATG\nC\n>v2\nNNNN\n")

    records = read_fasta_sequences(fasta_path)

    assert [record.sample_id for record in records] == ["v1", "v2"]
    assert [record.sequence for record in records] == ["ATGC", "NNNN"]


def test_normalize_dna_sequence_rejects_invalid_characters():
    with pytest.raises(ValueError, match="Invalid DNA characters"):
        normalize_dna_sequence("ATGX", sample_id="bad")


@pytest.mark.parametrize(
    "entry_point", ["pool", "fasta", "sequences", "cli", "explicit_128"]
)
def test_embedding_defaults_and_resolution_override(monkeypatch, tmp_path, entry_point):
    pool_path = tmp_path / "designs.csv"
    output_path = tmp_path / "embeddings.npz"
    pd.DataFrame(
        {
            "variant_id": ["v1", "v2"],
            "sequence": ["ATGCATGC", "ATGCNNNN"],
        }
    ).to_csv(pool_path, index=False)

    class FakeEncoder:
        def encode(self, sequence):
            return np.ones((len(sequence), 4), dtype=np.float32)

    class FakeForward:
        def apply(self, params, state, rng, onehot, organism_index, metadata):
            assert np.all(organism_index == 0)
            batch_size, padded_len, _ = onehot.shape
            fine = np.full((batch_size, padded_len, 3), 1000.0, dtype=np.float32)
            fine[:, :8, :] = 1.0
            coarse = np.full(
                (batch_size, padded_len // 128, 3), 1000.0, dtype=np.float32
            )
            coarse[:, :1, :] = 3.0
            embeddings = SimpleNamespace(
                embeddings_1bp=fine,
                embeddings_128bp=coarse,
            )
            return embeddings, {}

    fake_model = SimpleNamespace(
        _one_hot_encoder=FakeEncoder(),
        _params={},
        _state={},
        _metadata={},
    )
    monkeypatch.setattr(
        alphagenome,
        "_load_alphagenome_runtime",
        lambda: SimpleNamespace(jnp=np),
    )
    monkeypatch.setattr(alphagenome, "_auto_device", lambda runtime, device: None)
    monkeypatch.setattr(
        alphagenome,
        "_create_model_from_huggingface",
        lambda runtime, model_version, device: fake_model,
    )
    monkeypatch.setattr(
        alphagenome,
        "_build_forward",
        lambda runtime, **kwargs: FakeForward(),
    )

    if entry_point == "fasta":
        fasta = tmp_path / "designs.fasta"
        fasta.write_text(">v1\nATGCATGC\n>v2\nATGCNNNN\n")
        embed_fasta(fasta_path=fasta, output_path=output_path)
    elif entry_point == "sequences":
        records = read_design_pool_sequences(pool_path, id_column="variant_id")
        embed_sequences(records, output_path=output_path)
    elif entry_point == "cli":
        cli.main(
            [
                "embed-alphagenome",
                "--pool-csv",
                str(pool_path),
                "--id-column",
                "variant_id",
                "--output",
                str(output_path),
            ]
        )
    else:
        options = {"resolution": 128} if entry_point == "explicit_128" else {}
        embed_design_pool(
            pool_csv=pool_path,
            output_path=output_path,
            sequence_column="sequence",
            id_column="variant_id",
            **options,
        )

    data = np.load(output_path, allow_pickle=True)
    assert data["ids"].tolist() == ["v1", "v2"]
    assert data["embeddings"].shape == (2, 3)
    assert data["pooling"].item() == "mean"
    assert data["species"].item() == "human"
    assert data["model"].item() == "alphagenome-all_folds"
    assert data["resolution"].item() == (128 if entry_point == "explicit_128" else 1)
    np.testing.assert_array_equal(
        data["embeddings"],
        np.full((2, 3), 3.0 if entry_point == "explicit_128" else 1.0),
    )
