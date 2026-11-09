"""Candidate pools shared by the comparison methods."""

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

IS_COMPONENTS = (
    "SynTF Prom.",
    "AD",
    "IDR",
    "ZF Aff",
    "SynTF Term",
    "Syn. Term. Sp.",
    "No. of BS",
    "pMin",
    "Rep. Term. Sp.",
    "Orientation",
)
LOGIC_COMPONENTS = (
    "No. of Binding sites",
    "Binding site pattern",
    "Core prom",
    "Term spacing reporter",
    "Prom. SynTF A",
    "AD SynTF A",
    "IDR SynTF A",
    "ZF Affinity SynTF A",
    "Term spacing SynTF A",
    "Prom. SynTF B",
    "AD SynTF B",
    "IDR SynTF B",
    "ZF Affinity SynTF B",
    "Term spacing SynTF B",
    "Orientation",
)
LOGIC_PARTS = {
    "promoter_a": ("Prom. SynTF A", "Promoter (SynTF A/B)"),
    "promoter_b": ("Prom. SynTF B", "Promoter (SynTF A/B)"),
    "ad_a": ("AD SynTF A", "Activation Domain (SynTF A/B)"),
    "ad_b": ("AD SynTF B", "Activation Domain (SynTF A/B)"),
    "idp_a": ("IDR SynTF A", "IDR (SynTF A/B)"),
    "idp_b": ("IDR SynTF B", "IDR (SynTF A/B)"),
    "zf_a": ("ZF Affinity SynTF A", "ZF Affinity (SynTF A)"),
    "zf_b": ("ZF Affinity SynTF B", "ZF Affinity (SynTF B)"),
    "spacer_a": ("Term spacing SynTF A", "SynTF Term Spacer (SynTF A/B)"),
    "spacer_b": ("Term spacing SynTF B", "SynTF Term Spacer (SynTF A/B)"),
    "reporter_spacer": ("Term spacing reporter", "SynTF Term Spacer (SynTF A/B)"),
    "core_promoter": ("Core prom", "Core/Minimal Promoters"),
    "orientation": ("Orientation", "Orientation"),
}


@dataclass(eq=False)
class CandidatePool:
    task: str
    dataset_name: str
    ids: np.ndarray
    scores: np.ndarray
    features: pd.DataFrame
    components: np.ndarray
    component_names: dict
    circuits: list[dict] | None = None


def read_fasta(path, wanted):
    path = Path(path).expanduser()
    sequences, ident, parts = {}, None, []

    def finish():
        if ident in wanted:
            if ident in sequences:
                raise ValueError("Duplicate FASTA circuit ID")
            sequence = "".join(parts).upper()
            if len(sequence) < 6 or set(sequence) - set("ACGT"):
                raise ValueError(f"Invalid DNA for circuit ID {ident}")
            sequences[ident] = sequence

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        for line in stream:
            if line.startswith(">"):
                finish()
                ident, parts = int(line[1:].split()[0]), []
            elif ident in wanted:
                parts.append(line.strip())
            elif ident is None and line.strip():
                raise ValueError("FASTA sequence appears before its numeric ID header")
        finish()
    if set(sequences) != wanted:
        raise ValueError(
            f"{len(wanted - set(sequences))} required IDs absent from FASTA"
        )
    return sequences


def logic_circuits(frame, names):
    patterns = {
        1: "clustered_AB",
        2: "clustered_BA",
        3: "alternating_AB",
        4: "alternating_BA",
    }
    rows = []
    for values in frame.to_dict("records"):
        row = {
            field: names[group][str(int(values[column]))]
            for field, (column, group) in LOGIC_PARTS.items()
        }
        row["zf_a"] = "A:" + row["zf_a"]
        row["zf_b"] = "B:" + row["zf_b"]
        row["total_sites"] = {1: 2, 2: 8, 3: 12}[int(values["No. of Binding sites"])]
        row["pattern"] = patterns[int(values["Binding site pattern"])]
        rows.append(row)
    return rows


def load_pool(task, dataset_name, data_dir):
    if task not in ("IS", "OR", "AND"):
        raise ValueError("Unknown design task")
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Zenodo data directory does not exist: {data_dir}")
    folder = "IS" if task == "IS" else "logic"
    filename = (
        "166k_alphagenome_1bp_embedding.npz"
        if task == "IS"
        else "1m_alphagenome_1bp_embeddings.npz"
    )
    with np.load(data_dir / "embeddings" / folder / dataset_name / filename) as archive:
        ids = archive["ids"].astype(np.int64)
    library = "Rai_166k_data" if task == "IS" else "Rai_1M_data"
    measurement_dir = data_dir / "measurements" / library
    measurement_file = (
        "166k_Library_CLASSIC_Data.csv" if task == "IS" else "base_data_with_scores.csv"
    )
    frame = pd.read_csv(measurement_dir / measurement_file).iloc[ids].copy()
    frame = frame.reset_index(drop=True)
    component_names = json.loads((measurement_dir / "component_names.json").read_text())
    score_column = (
        "Fold Change (Induced/Basal)" if task == "IS" else task.lower() + "_score"
    )
    if task == "IS":
        combined = frame["SynTF Term & Sp."].astype(int) - 1
        frame["SynTF Term"] = combined // 3 + 1
        frame["Syn. Term. Sp."] = combined % 3 + 1
    columns = IS_COMPONENTS if task == "IS" else LOGIC_COMPONENTS
    components = np.column_stack(
        [pd.factorize(frame[name], sort=True)[0] for name in columns]
    )
    return CandidatePool(
        task=task,
        dataset_name=dataset_name,
        ids=ids,
        scores=frame[score_column].to_numpy(dtype=float),
        features=frame,
        components=components,
        component_names=component_names,
        circuits=logic_circuits(frame, component_names) if task != "IS" else None,
    )
