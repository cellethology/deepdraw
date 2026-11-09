from pathlib import Path

import numpy as np

from ..data import read_fasta


def load_inputs(pool, data_dir, sequences_fasta=None):
    library = "IS" if pool.task == "IS" else "logic"
    if sequences_fasta is None:
        sequences_fasta = (
            Path(data_dir).expanduser() / "sequences" / f"{library}_candidates.fasta.gz"
        )
    sequences = read_fasta(sequences_fasta, set(map(int, pool.ids)))
    width = max(map(len, sequences.values()))
    codes = np.full((len(pool.ids), width), 4, dtype=np.uint8)
    lookup = np.full(256, 4, dtype=np.uint8)
    lookup[np.frombuffer(b"ACGT", dtype=np.uint8)] = np.arange(4)
    for row, ident in enumerate(pool.ids):
        sequence = sequences[int(ident)].encode("ascii")
        codes[row, : len(sequence)] = lookup[np.frombuffer(sequence, dtype=np.uint8)]
    columns = ["Basal Exp (GFP, au)"]
    columns += (
        ["Induced Exp (GFP, au)"]
        if pool.task == "IS"
        else [
            "4-OHT Induced Exp (GFP, au)",
            "GZV Induced Exp (GFP, au)",
            "Dual Induced Exp (GFP, au)",
        ]
    )
    expression = pool.features[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(expression).all() or np.any(expression <= 0):
        raise ValueError("Expression measurements must be finite and positive")
    return codes, np.log2(expression)
